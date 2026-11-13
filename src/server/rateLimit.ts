// API の呼び出し回数の上限。公開URLに連続で送られたときに、Gemini や読み上げの費用がかさんだり、
// 利用枠（429）を使い切ってほかの人まで止まったりしないようにする。
// 数えるのはこのサーバー（Cloud Run ではインスタンス）ごとなので、上限はおおよそになる。
// インスタンスの数はデプロイ時に最大インスタンス数で抑える

import type { IncomingMessage } from 'node:http';

export type RateLimitGroup = 'gemini' | 'tts' | 'record';

export interface RateLimitRule {
  /** 1つの接続元が、時間の枠の中で呼べる回数 */
  perClient: number;
  /** サーバー全体で、時間の枠の中で呼べる回数 */
  global: number;
}

/** ふつうに遊ぶと Gemini を呼ぶAPIは1分に6〜8回ほど。読み上げは発言ごとに呼ぶので多め */
const DEFAULT_RATE_LIMITS: Record<RateLimitGroup, RateLimitRule> = {
  gemini: {
    perClient: readLimit('RATE_LIMIT_GEMINI_PER_CLIENT', 20),
    global: readLimit('RATE_LIMIT_GEMINI_GLOBAL', 120),
  },
  tts: {
    perClient: readLimit('RATE_LIMIT_TTS_PER_CLIENT', 60),
    global: readLimit('RATE_LIMIT_TTS_GLOBAL', 300),
  },
  /** 挑戦の始まりと終わりの記録。1回の挑戦で2回ほど */
  record: {
    perClient: readLimit('RATE_LIMIT_RECORD_PER_CLIENT', 20),
    global: readLimit('RATE_LIMIT_RECORD_GLOBAL', 300),
  },
};

const WINDOW_MS = 60_000;

/** Gemini を呼ぶAPIと、読み上げのAPI。ここにないAPIは数えない */
const GROUP_BY_PATH: Record<string, RateLimitGroup> = {
  '/api/chat': 'gemini',
  '/api/evaluate': 'gemini',
  '/api/owner-line': 'gemini',
  '/api/transcribe': 'gemini',
  '/api/decision': 'gemini',
  '/api/tts': 'tts',
  '/api/run-event': 'record',
};

function readLimit(name: string, fallback: number): number {
  const n = Number(process.env[name]);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback;
}

export function rateLimitGroupOf(method: string | undefined, path: string): RateLimitGroup | null {
  if (method === 'GET' && path === '/api/aggregate-tips') return 'record';
  if (method !== 'POST') return null;
  return GROUP_BY_PATH[path] ?? null;
}

/**
 * 接続元を見分けるキー。Cloud Run では X-Forwarded-For の最後に、Google の受付が見た接続元が付く。
 * 先頭は送る側が自由に書けるので使わない
 */
export function clientKeyOf(req: IncomingMessage): string {
  const forwarded = req.headers['x-forwarded-for'];
  const raw = Array.isArray(forwarded) ? forwarded.join(',') : forwarded;
  const parts = (raw ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  return (parts[parts.length - 1] ?? req.socket.remoteAddress ?? 'unknown').slice(0, 128);
}

const MAX_TRACKED_KEYS = 10_000;

/** 直近の時間の枠に呼んだ時刻を覚えておき、上限を超えたら断る（スライディングウィンドウ） */
export class RateLimiter {
  private readonly hits = new Map<string, number[]>();
  private readonly rules: Record<RateLimitGroup, RateLimitRule>;
  private readonly windowMs: number;
  private readonly now: () => number;

  constructor(
    rules: Record<RateLimitGroup, RateLimitRule> = DEFAULT_RATE_LIMITS,
    windowMs = WINDOW_MS,
    now: () => number = () => Date.now(),
  ) {
    this.rules = rules;
    this.windowMs = windowMs;
    this.now = now;
  }

  /** 呼んでよければ回数に数えて ok を返す。断るときは、あと何秒待てばよいかを返す */
  take(group: RateLimitGroup, clientKey: string): { ok: true } | { ok: false; retryAfterSec: number } {
    const rule = this.rules[group];
    const t = this.now();
    const globalKey = `${group}:global`;
    const clientHitKey = `${group}:client:${clientKey}`;
    const globalHits = this.recent(globalKey, t);
    const clientHits = this.recent(clientHitKey, t);
    const full =
      clientHits.length >= rule.perClient
        ? clientHits
        : globalHits.length >= rule.global
          ? globalHits
          : null;
    if (full) {
      const retryAfterSec = Math.max(1, Math.ceil((full[0] + this.windowMs - t) / 1000));
      return { ok: false, retryAfterSec };
    }
    clientHits.push(t);
    globalHits.push(t);
    this.remember(clientHitKey, clientHits);
    this.remember(globalKey, globalHits);
    return { ok: true };
  }

  private recent(key: string, t: number): number[] {
    const existing = this.hits.get(key);
    if (!existing) return [];
    const kept = existing.filter((at) => t - at < this.windowMs);
    if (kept.length === 0) this.hits.delete(key);
    else this.hits.set(key, kept);
    return kept;
  }

  private remember(key: string, list: number[]): void {
    if (!this.hits.has(key) && this.hits.size >= MAX_TRACKED_KEYS) {
      this.prune();
      while (this.hits.size >= MAX_TRACKED_KEYS) {
        const oldest = this.hits.keys().next().value;
        if (oldest === undefined) break;
        this.hits.delete(oldest);
      }
    }
    this.hits.set(key, list);
  }

  /** 古い記録を捨てる。接続元が増え続けても記録がたまらないように、定期的に呼ぶ */
  prune(): void {
    const t = this.now();
    for (const [key, list] of this.hits) {
      const kept = list.filter((at) => t - at < this.windowMs);
      if (kept.length === 0) this.hits.delete(key);
      else this.hits.set(key, kept);
    }
  }
}

// ============================================================================
// 1日あたりの上限（費用の非常停止）と、同時に処理する数の上限
// ============================================================================

/** Gemini を呼べる1日あたりの回数。日付は日本時間で区切る。サーバー（インスタンス）ごとに数える */
const DAILY_GEMINI_LIMIT = readLimit('RATE_LIMIT_GEMINI_DAILY', 2000);
/** 1つの接続元から同時に処理する、Gemini を呼ぶリクエストの数 */
export const MAX_CONCURRENT_PER_CLIENT = readLimit('RATE_LIMIT_CONCURRENT_PER_CLIENT', 2);

export class DailyLimitError extends Error {
  readonly statusCode = 429;
  readonly code = 'daily_limit';
  constructor() {
    super('本日の利用上限に達しました。明日になってから、もう一度お試しください。');
    this.name = 'DailyLimitError';
  }
}

function tokyoDateKey(now: number): string {
  return new Date(now).toLocaleDateString('sv-SE', { timeZone: 'Asia/Tokyo' });
}

/** 1日あたりの回数を数える。日本時間で日付が変わったら数え直す */
export class DailyCounter {
  private day = '';
  private count = 0;
  private readonly limit: number;
  private readonly now: () => number;

  constructor(limit: number, now: () => number = () => Date.now()) {
    this.limit = limit;
    this.now = now;
  }

  private roll(): void {
    const today = tokyoDateKey(this.now());
    if (today !== this.day) {
      this.day = today;
      this.count = 0;
    }
  }

  isExhausted(): boolean {
    this.roll();
    return this.count >= this.limit;
  }

  /** 1回分を数える。上限に達していれば DailyLimitError を投げる */
  consume(): void {
    this.roll();
    if (this.count >= this.limit) throw new DailyLimitError();
    this.count++;
  }
}

/** Gemini の呼び出しを数える、サーバー全体で1つの数え口 */
export const dailyGeminiCounter = new DailyCounter(DAILY_GEMINI_LIMIT);

/** 同時に処理している数を接続元ごとに数え、上限を超える分は断る */
export class ConcurrencyGate {
  private readonly inFlight = new Map<string, number>();
  private readonly max: number;

  constructor(max: number) {
    this.max = max;
  }

  tryEnter(key: string): boolean {
    const n = this.inFlight.get(key) ?? 0;
    if (n >= this.max) return false;
    this.inFlight.set(key, n + 1);
    return true;
  }

  leave(key: string): void {
    const n = (this.inFlight.get(key) ?? 1) - 1;
    if (n <= 0) this.inFlight.delete(key);
    else this.inFlight.set(key, n);
  }
}

// ============================================================================
// 本文の大きさと、呼び出し元の確認
// ============================================================================

/** 画像や音声を送るAPIだけ大きめにし、そのほかは小さく抑える */
const LARGE_BODY_PATHS = new Set(['/api/chat', '/api/evaluate', '/api/transcribe']);
export const LARGE_BODY_BYTES = 2 * 1024 * 1024;
export const SMALL_BODY_BYTES = 16 * 1024;

/** 静的配信と API 応答で共通して付与するセキュリティヘッダー */
export const COMMON_SECURITY_HEADERS: Record<string, string> = {
  'X-Content-Type-Options': 'nosniff',
  'X-Robots-Tag': 'noindex, nofollow, noarchive',
  'Referrer-Policy': 'strict-origin-when-cross-origin',
  'Cross-Origin-Resource-Policy': 'same-origin',
  'Permissions-Policy': 'camera=(), geolocation=(), payment=(), usb=()',
};

export function maxBodyBytesFor(path: string): number {
  return LARGE_BODY_PATHS.has(path) ? LARGE_BODY_BYTES : SMALL_BODY_BYTES;
}

const BOT_UA_PATTERN =
  /([a-z0-9_-]*bot\b|crawler|[a-z0-9_-]*spider\b|scrapy|\bcurl\b|\bwget\b|python-requests|python-urllib|\bhttpx\b|\baiohttp\b|go-http-client|\bjava\/|libwww-perl|\baxios\/|postmanruntime|\binsomnia\b)/i;

/**
 * クローラやスクリプトなどの自動プログラムによる呼び出しかどうかを判定する。
 * User-Agent が空、または既知のボット・HTTPクライアント文字列を含む場合は true を返す
 */
export function isBotUserAgent(req: IncomingMessage): boolean {
  const raw = req.headers['user-agent'];
  const ua = (Array.isArray(raw) ? raw.join(' ') : (raw ?? '')).trim();
  if (!ua) return true;
  return BOT_UA_PATTERN.test(ua);
}

/**
 * ブラウザから来たリクエストが、このサイト自身から送られたかを確かめる。
 * Origin がない直接の呼び出し（curl やスクリプト等）は Sec-Fetch-Site: same-origin がない限り断る。
 * このサイトの名前は Host と、前に立つ配信（Firebase Hosting など）が付ける X-Forwarded-Host から読む。
 * ほかに許す呼び出し元があれば ALLOWED_ORIGINS にカンマ区切りで書く
 */
export function isAllowedOrigin(req: IncomingMessage): boolean {
  const origin = req.headers.origin;
  if (!origin) {
    return req.headers['sec-fetch-site'] === 'same-origin';
  }
  let parsedOrigin: URL;
  try {
    parsedOrigin = new URL(origin);
  } catch {
    return false;
  }
  if (parsedOrigin.protocol !== 'https:' && parsedOrigin.protocol !== 'http:') {
    return false;
  }
  const allowed = (process.env.ALLOWED_ORIGINS ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  if (allowed.includes(parsedOrigin.origin)) return true;

  // 逆プロキシが HTTPS で受けている場合は HTTP からのダウングレード呼び出しを断る
  const rawProto = req.headers['x-forwarded-proto'];
  const lastProto = (Array.isArray(rawProto) ? rawProto.at(-1) : rawProto)
    ?.split(',')
    .at(-1)
    ?.trim()
    .toLowerCase();
  if (lastProto === 'https' && parsedOrigin.protocol !== 'https:') {
    return false;
  }

  // X-Forwarded-Host はクライアント側で先頭に偽装値を付け足せるため、プロキシが付与した末尾のみを見る
  const rawForwardedHost = req.headers['x-forwarded-host'];
  const lastForwardedHost = (
    Array.isArray(rawForwardedHost) ? rawForwardedHost.at(-1) : rawForwardedHost
  )
    ?.split(',')
    .at(-1)
    ?.trim()
    .toLowerCase();
  const directHost = req.headers.host?.trim().toLowerCase();
  const originHost = parsedOrigin.host.toLowerCase();
  return (
    originHost === directHost ||
    (Boolean(lastForwardedHost) && originHost === lastForwardedHost)
  );
}

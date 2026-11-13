import type { IncomingMessage, ServerResponse } from 'node:http';
import type {
  AppSpec,
  ChatHistoryEntry,
  ChatRequest,
  ComponentBoundsInput,
  DecisionStage,
  EvaluateRequest,
  OwnerLineRequest,
  LatestReviewRecord,
  RecordDecisionRequest,
  ReviewOutcome,
  TargetCohortId,
  TranscribeRequest,
  TtsRequest,
  VisibilityCondition,
} from '../shared/types.ts';
import { isSafeComponentId, isSafeStateKey } from '../shared/appRuntime.ts';
import { ACTIVE_THEME } from '../shared/initialTheme.ts';
import {
  AiNotConfiguredError,
  AiTimeoutError,
  chatWithStudio,
  evaluateAppWithPersonas,
  getAiConnectionStatus,
  recordHumanDecisionWithFunctionCalling,
  transcribeAndRefineUserAudio,
  voiceOwnerLine,
} from './gemini.ts';
import { getPublicAggregateTips } from './aggregateTips.ts';
import {
  synthesizeStudioSpeech,
  recordRunEvent,
  type RunEventInput,
} from './store.ts';
import {
  COMMON_SECURITY_HEADERS,
  ConcurrencyGate,
  DailyLimitError,
  LARGE_BODY_BYTES,
  MAX_CONCURRENT_PER_CLIENT,
  RateLimiter,
  clientKeyOf,
  dailyGeminiCounter,
  isAllowedOrigin,
  isBotUserAgent,
  maxBodyBytesFor,
  rateLimitGroupOf,
} from './rateLimit.ts';
import { describeErrorForLog } from './logSafe.ts';
import {
  PromptRejectedError,
  assertPromptAllowed,
  findPersonalInfoByRules,
} from './modelArmor.ts';

const apiRateLimiter = new RateLimiter();
const geminiConcurrency = new ConcurrencyGate(MAX_CONCURRENT_PER_CLIENT);
// 使われなくなった接続元の記録を捨てる。サーバーの終了を妨げないよう unref する
setInterval(() => apiRateLimiter.prune(), 5 * 60_000).unref();

const MAX_TEXT_INPUT_LENGTH = 2_000;
const VALID_DECISION_STAGES = new Set<DecisionStage>([
  'select_cohort',
  'apply_change',
  'review_disposition',
]);
const VALID_OWNER_SITUATIONS = new Set<OwnerLineRequest['situation']>([
  'change',
  'closing',
  'reply',
]);
const VALID_REVIEW_OUTCOMES = new Set<ReviewOutcome>([
  'achieved',
  'bothered',
  'abandoned',
  'regretted',
]);
const SAFE_DECISION_TARGET_RE = /^[A-Za-z0-9_:,-]{1,80}$/;

/** 発言の番号は 0 以上の整数だけを受け付ける。それ以外は番号なしとして扱う */
function sanitizeTurnIndex(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 && value < 10_000
    ? value
    : undefined;
}

const RUN_ID_RE = /^run-[0-9a-z-]{1,40}$/;
const COHORT_IDS = new Set<string>(ACTIVE_THEME.cohorts.map((c) => c.id));
const MAX_HISTORY_ENTRIES = 20;
const MAX_HISTORY_TEXT_LENGTH = 1_000;

/** 挑戦の番号は画面が作る形だけを受け付ける。記録に自由な文字列が入らないようにする */
function sanitizeRunId(value: unknown): string | undefined {
  return typeof value === 'string' && RUN_ID_RE.test(value) ? value : undefined;
}

/** 客層は題材に決まっているものだけを受け付ける */
function sanitizeCohortId(value: unknown): TargetCohortId | undefined {
  return typeof value === 'string' && COHORT_IDS.has(value) ? (value as TargetCohortId) : undefined;
}

export function sanitizeComponentIds(value: unknown): string[] {
  return Array.isArray(value) ? value.filter(isSafeComponentId).slice(0, 20) : [];
}

/** 部品の座標一覧は、数値が有限で形の正しいものだけを上限件数まで受け付ける */
export function sanitizeComponentBounds(value: unknown): ComponentBoundsInput[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const inCoordRange = (n: unknown): n is number =>
    typeof n === 'number' && Number.isFinite(n) && n >= -10_000 && n <= 100_000;
  return value
    .filter(
      (b): b is ComponentBoundsInput =>
        Boolean(b) &&
        typeof b === 'object' &&
        isSafeComponentId((b as ComponentBoundsInput).id) &&
        inCoordRange((b as ComponentBoundsInput).left) &&
        inCoordRange((b as ComponentBoundsInput).top) &&
        inCoordRange((b as ComponentBoundsInput).right) &&
        inCoordRange((b as ComponentBoundsInput).bottom) &&
        (b as ComponentBoundsInput).right >= (b as ComponentBoundsInput).left &&
        (b as ComponentBoundsInput).bottom >= (b as ComponentBoundsInput).top,
    )
    .slice(0, 60)
    .map((b) => ({
      id: b.id,
      left: b.left,
      top: b.top,
      right: b.right,
      bottom: b.bottom,
    }));
}

const BASE64_PAYLOAD_RE =
  /^(?:data:[a-z0-9.+-]+\/[a-z0-9.+-]+(?:;[a-z0-9.+-]+=[^;,]+)*;base64,)?[A-Za-z0-9+/=\s]+$/i;

/** 画像・音声の Base64 文字列は、形式と長さの上限を満たすものだけを通す */
export function sanitizeBase64(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > LARGE_BODY_BYTES) return undefined;
  return BASE64_PAYLOAD_RE.test(trimmed) ? trimmed : undefined;
}

/** 会話の履歴は件数と長さを抑え、形の正しいものだけを Gemini に渡す */
function sanitizeHistory(value: unknown): ChatHistoryEntry[] {
  if (!Array.isArray(value)) return [];
  return value
    .filter(
      (e): e is ChatHistoryEntry =>
        Boolean(e) && typeof e.speaker === 'string' && typeof e.text === 'string',
    )
    .slice(-MAX_HISTORY_ENTRIES)
    .map((e) => ({ speaker: e.speaker.slice(0, 20), text: e.text.slice(0, MAX_HISTORY_TEXT_LENGTH) }));
}

/** 直近のお客さんの反応の記録も、人数と行数と長さを抑える */
function sanitizeLatestReviews(value: unknown): LatestReviewRecord | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const v = value as Partial<LatestReviewRecord>;
  if (!Array.isArray(v.reviews)) return undefined;
  const text = (s: unknown, max: number) => (typeof s === 'string' ? s.slice(0, max) : '');
  const reviews = v.reviews
    .filter(
      (r): r is LatestReviewRecord['reviews'][number] =>
        Boolean(r) &&
        typeof r === 'object' &&
        VALID_REVIEW_OUTCOMES.has(r.outcome as ReviewOutcome),
    )
    .slice(0, 9)
    .map((r) => ({
      name: text(r.name, 40),
      outcome: r.outcome,
      actionTrace: Array.isArray(r.actionTrace)
        ? r.actionTrace.slice(0, 16).map((t) => text(t, 160))
        : [],
      frictionComponentIds: sanitizeComponentIds(r.frictionComponentIds),
    }));
  return {
    screenLabel: text(v.screenLabel, 40),
    isCurrentScreen: v.isCurrentScreen === true,
    reviews,
    ownerMetrics: Array.isArray(v.ownerMetrics)
      ? v.ownerMetrics.slice(0, 10).map((m) => ({
          label: text(m?.label, 40),
          count:
            typeof m?.count === 'number' && Number.isInteger(m.count) && m.count >= 0 && m.count <= 100
              ? m.count
              : 0,
          missing: m?.missing === true,
        }))
      : undefined,
  };
}

/**
 * 今の発言と会話の履歴に、個人情報やAIへの指示の乗っ取りが含まれていないかを調べる。
 * 履歴が長くても今の発言が切り落とされないよう先に単独で検査し、発言者名を偽装した履歴も全件検査する
 */
async function assertMessageAndHistoryAllowed(
  message: string,
  history: ChatHistoryEntry[],
  extraTexts: ReadonlyArray<string> = [],
): Promise<void> {
  for (const entry of history) {
    if (findPersonalInfoByRules(entry.text)) {
      throw new PromptRejectedError('personal_info');
    }
  }
  for (const extra of extraTexts) {
    if (findPersonalInfoByRules(extra)) {
      throw new PromptRejectedError('personal_info');
    }
  }
  if (message.trim()) {
    await assertPromptAllowed(message);
  }
  const recentHistory = history
    .slice(-12)
    .map((e) => e.text)
    .join('\n')
    .slice(-3800);
  if (recentHistory.trim()) {
    await assertPromptAllowed(recentHistory);
  }
}

/** 画面の仕様を直接書き換えて送られた場合でも、部品の文字やHTMLから個人情報が Gemini に渡らないようにする */
function assertSpecAllowed(spec: AppSpec): void {
  const target = JSON.stringify({
    title: spec.title,
    subtitle: spec.subtitle,
    state: spec.state,
    derived: spec.derived,
    components: spec.components,
  });
  if (findPersonalInfoByRules(target)) {
    throw new PromptRejectedError('personal_info');
  }
}

function isValidCondition(cond: unknown): cond is VisibilityCondition {
  return (
    Boolean(cond) &&
    typeof cond === 'object' &&
    typeof (cond as VisibilityCondition).stateKey === 'string' &&
    isSafeStateKey((cond as VisibilityCondition).stateKey)
  );
}

export function isValidAppSpec(value: unknown): value is AppSpec {
  if (!value || typeof value !== 'object') return false;
  const candidate = value as Partial<AppSpec>;
  if (typeof candidate.id !== 'string' || !candidate.id || candidate.id.length > 80) return false;
  if (typeof candidate.title !== 'string' || candidate.title.length > 120) return false;
  if (typeof candidate.subtitle !== 'string' || candidate.subtitle.length > 120) return false;
  if (!candidate.theme || typeof candidate.theme !== 'object') return false;
  if (!Array.isArray(candidate.components) || candidate.components.length > 60) return false;
  if (!Array.isArray(candidate.state) || candidate.state.length > 40) return false;
  if (
    candidate.derived !== undefined &&
    (!Array.isArray(candidate.derived) ||
      candidate.derived.length > 20 ||
      !candidate.derived.every(
        (d) =>
          Boolean(d) &&
          typeof d === 'object' &&
          typeof d.key === 'string' &&
          isSafeStateKey(d.key) &&
          typeof d.expression === 'string' &&
          d.expression.length <= 500,
      ))
  ) {
    return false;
  }
  const isOptionalBoundedStr = (v: unknown, max: number) =>
    v === undefined || (typeof v === 'string' && v.length <= max);
  return (
    candidate.components.every(
      (c) =>
        Boolean(c) &&
        typeof c === 'object' &&
        isSafeComponentId(c.id) &&
        typeof c.type === 'string' &&
        c.type.length <= 40 &&
        typeof c.label === 'string' &&
        c.label.length <= 500 &&
        isOptionalBoundedStr(c.badge, 120) &&
        isOptionalBoundedStr(c.caption, 500) &&
        isOptionalBoundedStr(c.priceLabel, 120) &&
        isOptionalBoundedStr(c.placeholder, 500) &&
        isOptionalBoundedStr(c.icon, 80) &&
        isOptionalBoundedStr(c.closeLabel, 120) &&
        (c.optionDetails === undefined ||
          (Array.isArray(c.optionDetails) &&
            c.optionDetails.length <= 20 &&
            c.optionDetails.every((d) => typeof d === 'string' && d.length <= 300))) &&
        (c.behavior === undefined ||
          (Boolean(c.behavior) &&
            typeof c.behavior === 'object' &&
            isOptionalBoundedStr(c.behavior.confirmText, 500))) &&
        (c.customHtml === undefined || (typeof c.customHtml === 'string' && c.customHtml.length <= 10_000)) &&
        (c.customCss === undefined || (typeof c.customCss === 'string' && c.customCss.length <= 5_000)) &&
        (c.options === undefined ||
          (Array.isArray(c.options) &&
            c.options.length <= 20 &&
            c.options.every((o) => typeof o === 'string' && o.length <= 200))) &&
        (c.rows === undefined ||
          (Array.isArray(c.rows) &&
            c.rows.length <= 20 &&
            c.rows.every(
              (r) =>
                Boolean(r) &&
                typeof r === 'object' &&
                typeof r.label === 'string' &&
                r.label.length <= 120 &&
                typeof r.value === 'string' &&
                r.value.length <= 120 &&
                (r.visibleWhen === undefined || isValidCondition(r.visibleWhen)),
            ))) &&
        Boolean(c.style) &&
        typeof c.style === 'object' &&
        Array.isArray(c.actions) &&
        c.actions.length <= 20 &&
        c.actions.every(
          (a) =>
            Boolean(a) &&
            typeof a === 'object' &&
            typeof a.type === 'string' &&
            a.type.length <= 40 &&
            isOptionalBoundedStr(a.message, 500) &&
            (a.targetKey === undefined ||
              (typeof a.targetKey === 'string' && isSafeStateKey(a.targetKey))),
        ) &&
        (c.stateKey === undefined ||
          (typeof c.stateKey === 'string' && isSafeStateKey(c.stateKey))) &&
        (c.visibleWhen === undefined || isValidCondition(c.visibleWhen)) &&
        (c.trapWhen === undefined || isValidCondition(c.trapWhen)),
    ) &&
    candidate.state.every(
      (s) =>
        Boolean(s) &&
        typeof s === 'object' &&
        typeof s.key === 'string' &&
        s.key.length <= 80 &&
        isSafeStateKey(s.key),
    )
  );
}

class PayloadTooLargeError extends Error {
  readonly statusCode = 413;
  constructor() {
    super('リクエストサイズが上限を超えています');
    this.name = 'PayloadTooLargeError';
  }
}

/** 本文を読む。大きさの上限はAPIごとに決め、画像や音声を送らないAPIは小さく抑える */
function readJsonBody<T>(req: IncomingMessage): Promise<T> {
  const maxBytes = maxBodyBytesFor((req.url ?? '').split('?')[0]);
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;

    req.on('data', (chunk: Buffer) => {
      total += chunk.length;
      if (total > maxBytes) {
        // 読むのをやめ、413 を返してから接続を閉じる。先に接続を切ると、画面に理由が届かない
        req.removeAllListeners('data');
        req.pause();
        reject(new PayloadTooLargeError());
        return;
      }
      chunks.push(chunk);
    });

    req.on('end', () => {
      try {
        const raw = Buffer.concat(chunks).toString('utf-8');
        resolve(JSON.parse(raw) as T);
      } catch (err) {
        reject(err);
      }
    });

    req.on('error', reject);
  });
}

function sendJson(
  res: ServerResponse,
  statusCode: number,
  payload: unknown,
  extraHeaders: Record<string, string> = {},
): void {
  const body = JSON.stringify(payload);
  res.writeHead(statusCode, {
    ...COMMON_SECURITY_HEADERS,
    ...extraHeaders,
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    'X-Frame-Options': 'DENY',
    Vary: 'Origin',
  });
  res.end(body);
}

/**
 * `/api/*` リクエストを処理する共通ハンドラ。
 * 処理した場合は true、対象外パスの場合は false を返す。
 */
export async function handleApiRequest(
  req: IncomingMessage,
  res: ServerResponse,
): Promise<boolean> {
  const url = (req.url ?? '').split('?')[0];
  if (!url.startsWith('/api/')) {
    return false;
  }

  // 自動スクリプトやクローラによる直接の API 呼び出しは断る
  if (isBotUserAgent(req)) {
    sendJson(res, 403, { error: '自動プログラムからの呼び出しはできません。' });
    return true;
  }

  // ほかのサイトのページや、Origin / Sec-Fetch-Site のない外部スクリプトからの呼び出しは断る
  if ((req.method !== 'GET' || url !== '/api/status') && !isAllowedOrigin(req)) {
    sendJson(res, 403, { error: 'このサイトの画面からだけ使えます。' });
    return true;
  }

  // Gemini や読み上げを呼ぶAPIは、本文を読む前に回数の上限を確かめる
  const limitGroup = rateLimitGroupOf(req.method, url);
  const clientKey = clientKeyOf(req);
  if (limitGroup) {
    if (limitGroup === 'gemini' && dailyGeminiCounter.isExhausted()) {
      const err = new DailyLimitError();
      sendJson(res, err.statusCode, { error: err.message, code: err.code });
      return true;
    }
    const taken = apiRateLimiter.take(limitGroup, clientKey);
    if (!taken.ok) {
      sendJson(
        res,
        429,
        { error: `混み合っています。${taken.retryAfterSec}秒ほど待ってから、もう一度送ってください。` },
        { 'Retry-After': String(taken.retryAfterSec) },
      );
      return true;
    }
  }

  // 1つの接続元から同時に処理する数を抑え、一度に大量に送られても利用枠を使い切らないようにする
  if (limitGroup === 'gemini') {
    if (!geminiConcurrency.tryEnter(clientKey)) {
      sendJson(res, 429, { error: '前の返事を待ってから、もう一度送ってください。' });
      return true;
    }
    try {
      return await routeApiRequest(req, res, url);
    } finally {
      geminiConcurrency.leave(clientKey);
    }
  }
  return routeApiRequest(req, res, url);
}

/** 確認を通ったリクエストを、それぞれのAPIに振り分ける */
async function routeApiRequest(
  req: IncomingMessage,
  res: ServerResponse,
  url: string,
): Promise<boolean> {

  try {
    if (req.method === 'GET' && url === '/api/status') {
      sendJson(res, 200, getAiConnectionStatus());
      return true;
    }

    if (req.method === 'GET' && url === '/api/aggregate-tips') {
      sendJson(res, 200, await getPublicAggregateTips());
      return true;
    }

    if (req.method === 'POST' && url === '/api/decision') {
      const body = await readJsonBody<RecordDecisionRequest>(req);
      if (
        !body?.stage ||
        !VALID_DECISION_STAGES.has(body.stage) ||
        typeof body.targetId !== 'string' ||
        !SAFE_DECISION_TARGET_RE.test(body.targetId) ||
        typeof body.chosenLabel !== 'string' ||
        !body.chosenLabel.trim()
      ) {
        sendJson(res, 400, { error: '有効な stage, targetId, chosenLabel が必要です' });
        return true;
      }
      const isAllowedDecision =
        body.stage === 'select_cohort'
          ? ACTIVE_THEME.cohorts.some((c) => c.id === body.targetId && c.label === body.chosenLabel)
          : body.stage === 'apply_change'
            ? body.chosenLabel === '何もしない' || body.chosenLabel === '手直し指示'
            : body.chosenLabel.length <= 40 && !findPersonalInfoByRules(body.chosenLabel);
      if (!isAllowedDecision) {
        sendJson(res, 400, { error: '有効な stage, targetId, chosenLabel が必要です' });
        return true;
      }
      if (body.stage === 'review_disposition') {
        await assertPromptAllowed(body.chosenLabel);
      }
      // 判断の記録に残すのは決まった値だけ。自由に書かれた理由は受け取らない
      const result = await recordHumanDecisionWithFunctionCalling({
        stage: body.stage,
        targetId: body.targetId,
        chosenLabel: body.chosenLabel,
        reason: '',
      });
      sendJson(res, 200, result);
      return true;
    }

    if (req.method === 'POST' && url === '/api/chat') {
      const body = await readJsonBody<ChatRequest>(req);
      if (!isValidAppSpec(body?.currentSpec) || typeof body.userMessage !== 'string') {
        sendJson(res, 400, { error: 'currentSpec と userMessage が必要です' });
        return true;
      }
      const userMessage = body.userMessage.slice(0, MAX_TEXT_INPUT_LENGTH);
      const history = sanitizeHistory(body.history);
      const latestReviews = sanitizeLatestReviews(body.latestReviews);
      const reviewTexts = latestReviews
        ? [
            latestReviews.screenLabel,
            ...latestReviews.reviews.flatMap((r) => [r.name, ...r.actionTrace]),
            ...(latestReviews.ownerMetrics?.map((m) => m.label) ?? []),
          ]
        : [];
      // Gemini に送る前に、個人情報やAIへの指示の乗っ取りが含まれていないかを調べる。
      // 履歴の発言も調べ、画面を通さずに送られた履歴から乗っ取られないようにする
      assertSpecAllowed(body.currentSpec);
      await assertMessageAndHistoryAllowed(userMessage, history, reviewTexts);
      const result = await chatWithStudio({
        currentSpec: body.currentSpec,
        runId: sanitizeRunId(body.runId),
        cohortId: sanitizeCohortId(body.cohortId),
        turnIndex: sanitizeTurnIndex(body.turnIndex),
        shareForAggregate: body.shareForAggregate === true,
        userMessage,
        history,
        latestReviews,
        targetComponentIds: sanitizeComponentIds(body.targetComponentIds),
        annotatedImageBase64: sanitizeBase64(body.annotatedImageBase64),
        mode: body.mode === 'circled' ? 'circled' : undefined,
        recipient: typeof body.recipient === 'string' ? body.recipient.slice(0, 40) : undefined,
        componentBounds: sanitizeComponentBounds(body.componentBounds),
      });
      sendJson(res, 200, result);
      return true;
    }

    if (req.method === 'POST' && url === '/api/run-event') {
      const body = await readJsonBody<Partial<RunEventInput>>(req);
      const reasons = ['cleared', 'reset', 'closed', 'idle'] as const;
      const isCount = (v: unknown) => typeof v === 'number' && Number.isInteger(v) && v >= 0 && v < 10_000;
      const runId = sanitizeRunId(body?.runId);
      if (!runId || (body.eventType !== 'start' && body.eventType !== 'end')) {
        sendJson(res, 400, { error: 'runId と eventType が必要です' });
        return true;
      }
      // 決まった値だけを取り出して送る。自由に書ける文字列は受け取らない
      await recordRunEvent({
        runId,
        eventType: body.eventType,
        endReason: reasons.find((r) => r === body.endReason),
        cohortId: sanitizeCohortId(body.cohortId),
        scenario: typeof body.scenario === 'string' && /^[a-z_:]{1,30}$/.test(body.scenario) ? body.scenario : undefined,
        turnCount: isCount(body.turnCount) ? body.turnCount : undefined,
        achievedCount: isCount(body.achievedCount) ? body.achievedCount : undefined,
        totalCustomers: isCount(body.totalCustomers) ? body.totalCustomers : undefined,
      });
      sendJson(res, 200, { ok: true });
      return true;
    }

    if (req.method === 'POST' && url === '/api/owner-line') {
      const body = await readJsonBody<OwnerLineRequest>(req);
      if (
        !body?.situation ||
        !VALID_OWNER_SITUATIONS.has(body.situation) ||
        !Array.isArray(body.facts) ||
        typeof body.fallback !== 'string'
      ) {
        sendJson(res, 400, { error: 'situation, facts, fallback が必要です' });
        return true;
      }
      const userMessage =
        typeof body.userMessage === 'string'
          ? body.userMessage.slice(0, MAX_TEXT_INPUT_LENGTH)
          : undefined;
      const history = sanitizeHistory(body.history);
      const facts = body.facts
        .filter((f): f is string => typeof f === 'string')
        .slice(0, 12)
        .map((f) => f.slice(0, 200));
      const fallback = body.fallback.slice(0, 200);
      await assertMessageAndHistoryAllowed(userMessage ?? '', history, [...facts, fallback]);
      sendJson(
        res,
        200,
        await voiceOwnerLine({
          situation: body.situation,
          facts,
          fallback,
          userMessage,
          history,
        }),
      );
      return true;
    }

    if (req.method === 'POST' && url === '/api/evaluate') {
      const body = await readJsonBody<EvaluateRequest>(req);
      if (!isValidAppSpec(body?.currentSpec)) {
        sendJson(res, 400, { error: 'currentSpec が必要です' });
        return true;
      }
      const history = sanitizeHistory(body.history);
      assertSpecAllowed(body.currentSpec);
      await assertMessageAndHistoryAllowed('', history);
      const result = await evaluateAppWithPersonas({
        currentSpec: body.currentSpec,
        runId: sanitizeRunId(body.runId),
        cohortId: sanitizeCohortId(body.cohortId),
        turnCount: sanitizeTurnIndex(body.turnCount),
        screenshotBase64: sanitizeBase64(body.screenshotBase64),
        componentBounds: sanitizeComponentBounds(body.componentBounds),
        history,
      });
      sendJson(res, 200, result);
      return true;
    }

    if (req.method === 'POST' && url === '/api/tts') {
      const body = await readJsonBody<TtsRequest>(req);
      if (typeof body?.text !== 'string' || !body.text.trim()) {
        sendJson(res, 400, { error: 'text が必要です' });
        return true;
      }
      const text = body.text.slice(0, 400);
      if (findPersonalInfoByRules(text)) {
        sendJson(res, 200, { fallbackBrowserTts: true });
        return true;
      }
      const result = await synthesizeStudioSpeech({
        text,
        speaker: typeof body.speaker === 'string' ? body.speaker.slice(0, 40) : 'assistant',
      });
      sendJson(res, 200, result);
      return true;
    }

    if (req.method === 'POST' && url === '/api/transcribe') {
      const body = await readJsonBody<TranscribeRequest>(req);
      const audioBase64 = sanitizeBase64(body?.audioBase64);
      if (!audioBase64 || !isValidAppSpec(body?.currentSpec)) {
        sendJson(res, 400, { error: 'audioBase64 と currentSpec が必要です' });
        return true;
      }
      assertSpecAllowed(body.currentSpec);
      const result = await transcribeAndRefineUserAudio({
        audioBase64,
        audioMimeType:
          typeof body.audioMimeType === 'string' ? body.audioMimeType.slice(0, 80) : undefined,
        currentSpec: body.currentSpec,
        targetComponentIds: sanitizeComponentIds(body.targetComponentIds),
        annotatedImageBase64: sanitizeBase64(body.annotatedImageBase64),
      });
      if (result.hasMeaningfulSpeech && result.cleanedText.trim()) {
        await assertPromptAllowed(result.cleanedText);
      }
      sendJson(res, 200, result);
      return true;
    }

    sendJson(res, 404, { error: `未対応のAPIエンドポイントです: ${req.method} ${url}` });
    return true;
  } catch (err) {
    if (err instanceof DailyLimitError) {
      sendJson(res, err.statusCode, { error: err.message, code: err.code });
      return true;
    }
    if (err instanceof PromptRejectedError) {
      // 止めた理由の種類だけを返す。発言そのものは返さず、ログにも残さない
      sendJson(res, err.statusCode, { error: err.message, code: err.code, reason: err.reason });
      return true;
    }
    if (err instanceof PayloadTooLargeError) {
      // 残りの本文は読まずに、返事を送ったら接続を閉じる
      sendJson(res, err.statusCode, { error: err.message }, { Connection: 'close' });
      return true;
    }
    if (err instanceof AiNotConfiguredError || err instanceof AiTimeoutError) {
      sendJson(res, err.statusCode, { error: err.message });
      return true;
    }
    if (err instanceof SyntaxError) {
      sendJson(res, 400, { error: 'JSONの形式が正しくありません' });
      return true;
    }
    // 原因は開発者向けにログへ残し、学習者には分かる言葉だけを返す
    console.error('[api] リクエスト処理に失敗しました:', describeErrorForLog(err));
    sendJson(res, 500, {
      error: 'お助けAIがうまく返事できませんでした。もう一度送ってください。',
    });
    return true;
  }
}

// みんなの集計のコツを、1時間ごとに作り直して保存し、開いた人には保存したものを返す。
// ウィンドウを開くたびに Gemini を呼ばないよう、サーバーのメモリと Cloud Firestore の aggregate_tips に置く。
// 古くなったコツは、まず保存したものを返してから裏で作り直す

import {
  ALL_AGGREGATE_TIP_SEGMENT,
  aggregateTipSegmentKey,
  buildAggregateTipFacts,
  composeAggregateTips,
  isPublicAggregateTips,
  type AggregateTipDrafter,
  type AggregateTipSegment,
  type PublicAggregateTips,
} from '../shared/aggregateTips.ts';
import { MIN_PUBLIC_RUN_COUNT } from '../shared/promptTelemetry.ts';
import type { PublicAggregateStats } from '../shared/types.ts';
import { draftAggregateTipsWithGemini } from './gemini.ts';
import {
  getPublicAggregateStats,
  isSafeFirestoreDocId,
  requestFirestoreJson,
} from './store.ts';

/** コツを作り直す間隔 */
export const AGGREGATE_TIPS_REFRESH_MS = 60 * 60 * 1000;
/** 集計を読めなかったときや、どのボックスも集計中のときは、早めに読み直す */
const AGGREGATE_TIPS_RETRY_MS = 5 * 60 * 1000;

/**
 * 保存したコツの決まり（プロンプト・タグの決め方・形）を変えたら上げる。古い保存は使われなくなる
 */
const AGGREGATE_TIPS_VERSION = 'v2';

export interface AggregateTipsDeps {
  /** 区分の集計を読む。今は全体だけ。客層で分けるときは、ここで絞り込んだ集計を返す */
  loadStats: (segment: AggregateTipSegment) => Promise<PublicAggregateStats>;
  drafter: AggregateTipDrafter | null;
  readStored: (key: string) => Promise<PublicAggregateTips | null>;
  writeStored: (key: string, value: PublicAggregateTips) => Promise<void>;
  now: () => number;
}

interface CacheEntry {
  value: PublicAggregateTips;
  refreshAt: number;
}

const memoryCache = new Map<string, CacheEntry>();
/** 同じ区分を同時に作り直さないよう、作り直し中のものを共有する */
const inFlight = new Map<string, Promise<PublicAggregateTips>>();

/** テスト用。メモリに置いたコツを消す */
export function clearAggregateTipsCache(): void {
  memoryCache.clear();
  inFlight.clear();
}

function hasGeneratedContent(value: PublicAggregateTips): boolean {
  return value.available && value.boxes.some((b) => b.status === 'ready');
}

function refreshAtFor(value: PublicAggregateTips, now: number): number {
  const parsed = Date.parse(value.generatedAt);
  const base = Number.isFinite(parsed) ? Math.min(parsed, now) : now;
  return base + (hasGeneratedContent(value) ? AGGREGATE_TIPS_REFRESH_MS : AGGREGATE_TIPS_RETRY_MS);
}

// ============================================================================
// Cloud Firestore
// ============================================================================

const FIRESTORE_COLLECTION = 'aggregate_tips';
const MAX_STORED_PAYLOAD_LENGTH = 32_000;

async function readStoredFromFirestore(key: string): Promise<PublicAggregateTips | null> {
  if (!isSafeFirestoreDocId(key)) return null;
  const res = await requestFirestoreJson<{
    fields?: { payload?: { stringValue?: string }; version?: { stringValue?: string } };
  }>(`/${FIRESTORE_COLLECTION}/${encodeURIComponent(key)}`);
  if (!res?.ok || !res.data) return null;
  try {
    if (res.data.fields?.version?.stringValue !== AGGREGATE_TIPS_VERSION) return null;
    const payload = res.data.fields?.payload?.stringValue;
    if (!payload || payload.length > MAX_STORED_PAYLOAD_LENGTH) return null;
    const parsed: unknown = JSON.parse(payload);
    return isPublicAggregateTips(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/** 保存するのは集計の値とコツの文だけ。発言の本文や個人を特定できる情報は入っていない */
async function writeStoredToFirestore(key: string, value: PublicAggregateTips): Promise<void> {
  if (!isSafeFirestoreDocId(key)) return;
  await requestFirestoreJson(`/${FIRESTORE_COLLECTION}/${encodeURIComponent(key)}`, {
    method: 'PATCH',
    body: {
      fields: {
        payload: { stringValue: JSON.stringify(value) },
        version: { stringValue: AGGREGATE_TIPS_VERSION },
        generatedAt: { timestampValue: value.generatedAt },
      },
    },
  });
}

const defaultDeps: AggregateTipsDeps = {
  loadStats: () => getPublicAggregateStats(),
  drafter: draftAggregateTipsWithGemini,
  readStored: readStoredFromFirestore,
  writeStored: writeStoredToFirestore,
  now: () => Date.now(),
};

// ============================================================================
// 作り直しと返し方
// ============================================================================

async function rebuildTips(
  key: string,
  segment: AggregateTipSegment,
  previous: PublicAggregateTips | undefined,
  deps: AggregateTipsDeps,
): Promise<PublicAggregateTips> {
  const running = inFlight.get(key);
  if (running) return running;
  const task = (async () => {
    let stats: PublicAggregateStats;
    try {
      stats = await deps.loadStats(segment);
    } catch {
      stats = {
        available: false,
        generatedAt: '',
        minRunsForDisplay: MIN_PUBLIC_RUN_COUNT,
        suppressed: true,
        cohorts: [],
      };
    }
    // 集計を一時的に読めないときは、前に作ったコツを出し続け、少しあとに読み直す
    if (!stats.available && previous && hasGeneratedContent(previous)) {
      memoryCache.set(key, { value: previous, refreshAt: deps.now() + AGGREGATE_TIPS_RETRY_MS });
      return previous;
    }
    const facts = buildAggregateTipFacts(stats);
    const boxes = await composeAggregateTips(facts, deps.drafter, { previous: previous?.boxes });
    const now = deps.now();
    const value: PublicAggregateTips = {
      available: stats.available,
      generatedAt: new Date(now).toISOString(),
      segment,
      minRunsForDisplay: stats.minRunsForDisplay,
      boxes,
    };
    memoryCache.set(key, { value, refreshAt: refreshAtFor(value, now) });
    if (hasGeneratedContent(value)) {
      await deps.writeStored(key, value);
    }
    return value;
  })();
  inFlight.set(key, task);
  try {
    return await task;
  } finally {
    inFlight.delete(key);
  }
}

/**
 * みんなの集計のコツを返す。
 * 新しいうちは保存したものをそのまま返す。古くなっていれば保存したものを返しつつ裏で作り直し、
 * まだ1つもなければその場で作る
 */
export async function getPublicAggregateTips(
  segment: AggregateTipSegment = ALL_AGGREGATE_TIP_SEGMENT,
  deps: AggregateTipsDeps = defaultDeps,
): Promise<PublicAggregateTips> {
  const key = aggregateTipSegmentKey(segment);
  const now = deps.now();
  let cached = memoryCache.get(key);
  if (cached && cached.refreshAt > now) return cached.value;

  if (!cached) {
    const stored = await deps.readStored(key);
    if (stored) {
      const currentNow = deps.now();
      cached = { value: stored, refreshAt: refreshAtFor(stored, currentNow) };
      memoryCache.set(key, cached);
      if (cached.refreshAt > currentNow) return stored;
    }
  }

  if (cached) {
    void rebuildTips(key, segment, cached.value, deps).catch(() => {});
    return cached.value;
  }
  return rebuildTips(key, segment, undefined, deps);
}

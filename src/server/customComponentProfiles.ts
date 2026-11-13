// 独自の部品の読み取り結果を、部品の定義ごとに1つに固定する。
// 同じ中身の部品なら、どのインスタンスでも、何回評価しても、同じ読み取り結果を使う。
// 置き場所はサーバーのメモリと Cloud Firestore の custom_component_profiles。
// Firestore には先に書いた結果だけを残し（あとから同じ鍵で書こうとしても上書きしない）、Cloud Run が複数台でも結果をそろえる

import { createHash } from 'node:crypto';
import { isSafeComponentId } from '../shared/appRuntime.ts';
import {
  isCustomComponent,
  isCustomComponentProfile,
  normalizeCustomComponentForProfile,
  pickCustomComponentProfile,
  type CustomComponentProfile,
} from '../shared/customComponentProfile.ts';
import type { AppComponent, AppSpec } from '../shared/types.ts';
import {
  isSafeFirestoreDocId,
  requestFirestoreJson,
} from './store.ts';

/**
 * 読み取りの決まり（プロンプト・スキーマ・モデル）を変えたら上げる。鍵に含めるので、古い読み取り結果は使われなくなる
 */
const CUSTOM_PROFILE_VERSION = 'v1';

/** 1つの部品を読み取る関数。Gemini を呼ぶものを gemini.ts から渡す。テストでは決まった値を返すものを渡す */
export type CustomComponentConverter = (comp: AppComponent) => Promise<CustomComponentProfile>;

export interface ResolvedCustomProfiles {
  /** 部品のID → 読み取り結果 */
  profiles: Record<string, CustomComponentProfile>;
  /** 読み取れなかった独自の部品のID。この回はHTMLに書かれた動きと文字だけで判定し、結果は保存しない */
  fallbackIds: string[];
}

const MEMORY_CACHE_LIMIT = 500;
const memoryCache = new Map<string, CustomComponentProfile>();
/** 同じ部品を同時に読み取らないよう、読み取り中のものを共有する */
const inFlight = new Map<string, Promise<CustomComponentProfile | null>>();

function customProfileCacheKey(comp: AppComponent, model: string): string {
  return createHash('sha256')
    .update(`${CUSTOM_PROFILE_VERSION}\n${model}\n${normalizeCustomComponentForProfile(comp)}`)
    .digest('hex');
}

function rememberInMemory(key: string, profile: CustomComponentProfile): void {
  if (memoryCache.has(key)) memoryCache.delete(key);
  memoryCache.set(key, profile);
  while (memoryCache.size > MEMORY_CACHE_LIMIT) {
    const oldest = memoryCache.keys().next().value;
    if (oldest === undefined) break;
    memoryCache.delete(oldest);
  }
}

/** テスト用。メモリの読み取り結果を消す */
export function clearCustomProfileMemoryCache(): void {
  memoryCache.clear();
  inFlight.clear();
}

// ============================================================================
// Cloud Firestore
// ============================================================================

const FIRESTORE_COLLECTION = 'custom_component_profiles';

type FsValue = { stringValue?: string; booleanValue?: boolean };

/** 保存するのは読み取り結果の値だけ。部品のHTMLや文字は保存しない */
function toFirestoreFields(profile: CustomComponentProfile): Record<string, FsValue> {
  return {
    purpose: { stringValue: profile.purpose },
    userChooses: { booleanValue: profile.userChooses },
    enabledByDefault: { booleanValue: profile.enabledByDefault },
    textSize: { stringValue: profile.textSize },
    tapTargetSize: { stringValue: profile.tapTargetSize },
    hasHiddenCondition: { booleanValue: profile.hasHiddenCondition },
    hardToClose: { booleanValue: profile.hardToClose },
    pressure: { booleanValue: profile.pressure },
    version: { stringValue: CUSTOM_PROFILE_VERSION },
  };
}

function fromFirestoreFields(fields: Record<string, FsValue> | undefined): CustomComponentProfile | null {
  if (!fields) return null;
  const candidate = {
    purpose: fields.purpose?.stringValue,
    userChooses: fields.userChooses?.booleanValue,
    enabledByDefault: fields.enabledByDefault?.booleanValue,
    textSize: fields.textSize?.stringValue,
    tapTargetSize: fields.tapTargetSize?.stringValue,
    hasHiddenCondition: fields.hasHiddenCondition?.booleanValue,
    hardToClose: fields.hardToClose?.booleanValue,
    pressure: fields.pressure?.booleanValue,
  };
  return isCustomComponentProfile(candidate) ? pickCustomComponentProfile(candidate) : null;
}

async function readFromFirestore(key: string): Promise<CustomComponentProfile | null> {
  if (!isSafeFirestoreDocId(key)) return null;
  const res = await requestFirestoreJson<{ fields?: Record<string, FsValue> }>(
    `/${FIRESTORE_COLLECTION}/${encodeURIComponent(key)}`,
  );
  return res?.ok && res.data ? fromFirestoreFields(res.data.fields) : null;
}

/**
 * 読み取り結果を Firestore に書く。すでに同じ鍵があれば書かず、先に書かれた結果を返す。
 * 2台が同時に同じ部品を読み取っても、どちらも先に書かれたほうの結果を使うようにする
 */
async function writeOnceToFirestore(
  key: string,
  profile: CustomComponentProfile,
): Promise<CustomComponentProfile> {
  if (!isSafeFirestoreDocId(key)) return profile;
  const res = await requestFirestoreJson(
    `/${FIRESTORE_COLLECTION}?documentId=${encodeURIComponent(key)}`,
    {
      method: 'POST',
      body: { fields: toFirestoreFields(profile) },
    },
  );
  if (res?.status === 409) {
    return (await readFromFirestore(key)) ?? profile;
  }
  return profile;
}

// ============================================================================
// 読み取り結果をそろえる
// ============================================================================

async function resolveOne(
  key: string,
  comp: AppComponent,
  convert: CustomComponentConverter,
): Promise<CustomComponentProfile | null> {
  const cached = memoryCache.get(key);
  if (cached) return cached;
  const running = inFlight.get(key);
  if (running) return running;

  const task = (async () => {
    const stored = await readFromFirestore(key);
    if (stored) {
      rememberInMemory(key, stored);
      return stored;
    }
    let converted: CustomComponentProfile;
    try {
      const raw = await convert(comp);
      if (!isCustomComponentProfile(raw)) return null;
      converted = pickCustomComponentProfile(raw);
    } catch {
      // 失敗や時間切れの結果は保存しない。次に評価したときに読み取り直す
      return null;
    }
    const settled = await writeOnceToFirestore(key, converted);
    rememberInMemory(key, settled);
    return settled;
  })();
  inFlight.set(key, task);
  try {
    return await task;
  } finally {
    inFlight.delete(key);
  }
}

/**
 * 画面にある独自の部品をすべて読み取る。読み取り済みの部品は保存した結果を使い、Gemini を呼ばない
 */
export async function resolveCustomComponentProfiles(
  spec: AppSpec,
  convert: CustomComponentConverter,
  model: string,
): Promise<ResolvedCustomProfiles> {
  const targets = spec.components.filter((comp) => isCustomComponent(comp) && isSafeComponentId(comp.id));
  const profiles = Object.create(null) as Record<string, CustomComponentProfile>;
  const fallbackIds: string[] = [];
  await Promise.all(
    targets.map(async (comp) => {
      const profile = await resolveOne(customProfileCacheKey(comp, model), comp, convert);
      if (profile) profiles[comp.id] = profile;
      else fallbackIds.push(comp.id);
    }),
  );
  fallbackIds.sort();
  return { profiles: { ...profiles }, fallbackIds };
}

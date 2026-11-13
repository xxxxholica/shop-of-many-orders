import { GoogleAuth } from 'google-auth-library';
import { existsSync, readFileSync } from 'node:fs';
import { ACTIVE_THEME, TARGET_COHORTS } from '../shared/initialTheme.ts';
import { buildPublicComponentLabels } from '../shared/aggregateTips.ts';
import {
  classifyDeterministicArchetype,
  cosineSimilarity,
  extractDeterministicNlSignal,
  formatPublicAggregateStats,
  limitEntitiesToVocabulary,
  type RawBigQueryAggregatePayload,
  summarizePromptTelemetry,
} from '../shared/promptTelemetry.ts';
import type {
  EmbeddingSignal,
  HumanDecisionRecord,
  NaturalLanguageSignal,
  PromptArchetype,
  PromptSessionMode,
  PromptSessionSummary,
  PromptTelemetryEvent,
  PublicAggregateStats,
  TargetCohortId,
  TtsRequest,
  TtsResponse,
} from '../shared/types.ts';

const MAX_IN_MEMORY_TELEMETRY = 300;
const inMemoryTelemetryEvents: PromptTelemetryEvent[] = [];
const ttsAudioCache = new Map<string, string>();
let presetsLoaded = false;

function ensureTtsPresetsLoaded(): void {
  if (presetsLoaded) return;
  presetsLoaded = true;
  try {
    const presetUrl = new URL('./ttsPresets.json', import.meta.url);
    if (!existsSync(presetUrl)) return;
    const raw = JSON.parse(readFileSync(presetUrl, 'utf-8')) as Record<string, string>;
    for (const [k, v] of Object.entries(raw)) {
      if (typeof v === 'string' && v.length > 0) {
        ttsAudioCache.set(k, v);
      }
    }
  } catch {
    // プリセットの読み込みに失敗しても通常のTTS合成へフォールバックする
  }
}

const GCP_PROJECT_ID_RE = /^(?:[a-z0-9-]+(?:\.[a-z0-9-]+)*:)?[a-z0-9][a-z0-9-]{2,62}$/i;
const GCP_LOCATION_RE = /^[a-z0-9-]{2,40}$/i;
const BQ_IDENTIFIER_RE = /^[A-Za-z0-9_]{1,64}$/;
const FIRESTORE_DOC_ID_RE = /^[A-Za-z0-9_:-]{1,128}$/;

export function isSafeGcpProjectId(project: unknown): project is string {
  return typeof project === 'string' && project !== 'your-gcp-project-id' && GCP_PROJECT_ID_RE.test(project);
}

export function isSafeGcpLocation(location: unknown): location is string {
  return typeof location === 'string' && GCP_LOCATION_RE.test(location);
}

export function isSafeFirestoreDocId(id: unknown): id is string {
  return typeof id === 'string' && FIRESTORE_DOC_ID_RE.test(id);
}

function isSafeBqIdentifier(name: string): boolean {
  return BQ_IDENTIFIER_RE.test(name);
}

export function getConfiguredGcpProject(): string | null {
  const project = process.env.GOOGLE_CLOUD_PROJECT;
  return isSafeGcpProjectId(project) ? project : null;
}

export function getFirestoreDocumentsBaseUrl(): string | null {
  if (process.env.FIRESTORE_ENABLED === '0') return null;
  const project = getConfiguredGcpProject();
  if (!project) return null;
  return `https://firestore.googleapis.com/v1/projects/${encodeURIComponent(project)}/databases/(default)/documents`;
}

let googleAuth: GoogleAuth | null = null;

export async function getGcpAccessToken(): Promise<string | null> {
  try {
    if (!googleAuth) {
      googleAuth = new GoogleAuth({
        scopes: ['https://www.googleapis.com/auth/cloud-platform'],
      });
    }
    const token = await googleAuth.getAccessToken();
    return token ?? null;
  } catch {
    return null;
  }
}

const FIRESTORE_TIMEOUT_MS = 2500;

export async function requestFirestoreJson<T = unknown>(
  subpath: string,
  options?: {
    method?: 'GET' | 'POST' | 'PATCH';
    body?: unknown;
    token?: string;
    parseJson?: boolean;
  },
): Promise<{ status: number; ok: boolean; data: T | null } | null> {
  const baseUrl = getFirestoreDocumentsBaseUrl();
  if (!baseUrl) return null;
  try {
    const token = options?.token ?? (await getGcpAccessToken());
    if (!token) return null;
    const method = options?.method ?? 'GET';
    const headers: Record<string, string> = {
      Authorization: `Bearer ${token}`,
    };
    if (options?.body !== undefined) {
      headers['Content-Type'] = 'application/json';
    }
    const res = await fetch(`${baseUrl}${subpath}`, {
      method,
      headers,
      body: options?.body !== undefined ? JSON.stringify(options.body) : undefined,
      signal: AbortSignal.timeout(FIRESTORE_TIMEOUT_MS),
    });
    const shouldParse = options?.parseJson ?? method === 'GET';
    const data = shouldParse && res.ok ? ((await res.json()) as T) : null;
    return { status: res.status, ok: res.ok, data };
  } catch {
    return null;
  }
}

/**
 * 人間の判断記録 (Own the Why) を保存する。
 * Cloud Firestore 環境が構成されている場合は ADC トークン付き REST 経由で送信する。
 * 保存するのは決まった値だけで、サーバーのメモリには残さない。
 */
export async function saveHumanDecision(
  decision: HumanDecisionRecord,
): Promise<{ persistedToFirestore: boolean }> {
  if (!isSafeFirestoreDocId(decision.id)) {
    return { persistedToFirestore: false };
  }
  const expireAt = new Date(Date.now() + TELEMETRY_TTL_MS).toISOString();
  const res = await requestFirestoreJson(`/decisions?documentId=${encodeURIComponent(decision.id)}`, {
    method: 'POST',
    body: {
      fields: {
        stage: { stringValue: decision.stage },
        targetId: { stringValue: decision.targetId },
        chosenLabel: { stringValue: decision.chosenLabel },
        reason: { stringValue: decision.reason },
        recordedVia: { stringValue: decision.recordedVia },
        createdAt: { stringValue: decision.createdAt },
        expireAt: { timestampValue: expireAt },
      },
    },
  });
  return { persistedToFirestore: Boolean(res?.ok) };
}

const ARCHETYPE_ANCHOR_TEXTS: Record<Exclude<PromptArchetype, 'inquiry'>, string> = {
  simple_delete: '定期便やクーポンや広告の欄をそのまま消して削除してなくして',
  condition_design:
    '初期チェックを外して自分で選べるようにして解約条件を分かりやすく明記して閉じるボタンを大きく下に配置して',
  direct_fix: '注文ボタンが動くように直して合計金額の数字を読みやすく変更して',
  vague_delegate: 'いい感じに全部まとめて使いやすく見やすく直しておまかせで',
  goal_directed:
    '定期便を望むお客さんだけが自分で選べるようにしたい クーポンを使いたい人が迷わず使えるようにして',
};

let cachedAnchorEmbeddings: Map<PromptArchetype, number[]> | null = null;

/**
 * Cloud Natural Language API (`documents:annotateText`) を用いて指示文のエンティティと感情極性を抽出する。
 * GCP 未設定または通信失敗時は決定論的抽出へフォールバックする。
 */
async function analyzePromptWithCloudNaturalLanguage(
  promptText: string,
): Promise<NaturalLanguageSignal> {
  const base = extractDeterministicNlSignal(promptText);
  const project = getConfiguredGcpProject();
  if (!project || !promptText.trim()) {
    return base;
  }

  try {
    const token = await getGcpAccessToken();
    if (!token) return base;

    const res = await fetch('https://language.googleapis.com/v1/documents:annotateText', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
        'x-goog-user-project': project,
      },
      body: JSON.stringify({
        document: {
          type: 'PLAIN_TEXT',
          language: 'ja',
          content: promptText.slice(0, 500),
        },
        features: {
          extractEntities: true,
          extractDocumentSentiment: true,
        },
        encodingType: 'UTF8',
      }),
      signal: AbortSignal.timeout(2500),
    });
    if (!res.ok) return base;

    const data = (await res.json()) as {
      entities?: Array<{ name?: string }>;
      documentSentiment?: { score?: number };
    };

    // 取り出した語は題材の語彙に限る。人名などが混ざっても残さない
    const mergedEntities = limitEntitiesToVocabulary([
      ...base.entities,
      ...(data.entities ?? []).map((ent) => ent.name?.trim() ?? '').filter(Boolean),
    ]);

    const sentimentScore =
      typeof data.documentSentiment?.score === 'number'
        ? Number(data.documentSentiment.score.toFixed(2))
        : base.sentimentScore;

    return {
      entities: mergedEntities.slice(0, 8),
      hasConditionConstraint: base.hasConditionConstraint,
      hasGoalStatement: base.hasGoalStatement,
      sentimentScore,
      analyzedVia: 'cloud_natural_language_api',
    };
  } catch {
    return base;
  }
}

/**
 * 指示の型（目的つき / 単純削除 / 条件設計 / 具体修正 / 丸投げ / 探索）を分類する。
 * 型の定義はコードの規則で決める。規則に当てはまらず既定の具体修正に落ちた実行指示だけ、
 * Gemini Enterprise Agent Platform の Text embeddings API (`text-multilingual-embedding-002`) で典型的な指示とのコサイン類似度を見て、近い型で補う。
 * 埋め込みで常に上書きすると、近さが低い発言で規則どおりの型が覆ることがあったため（例：目的つきの指示が丸投げになる）。
 */
export async function classifyPromptWithAgentPlatformEmbeddings(
  promptText: string,
  sessionMode: PromptSessionMode,
): Promise<EmbeddingSignal> {
  const base = classifyDeterministicArchetype(promptText, sessionMode);
  if (sessionMode !== 'execute' || base.closestArchetype !== 'direct_fix') {
    return base;
  }
  const project = getConfiguredGcpProject();
  if (!project || !promptText.trim()) {
    return base;
  }

  try {
    const token = await getGcpAccessToken();
    if (!token) return base;

    const rawLocation = process.env.GOOGLE_CLOUD_LOCATION || 'us-central1';
    const location = rawLocation === 'global' ? 'us-central1' : rawLocation;
    if (!isSafeGcpLocation(location)) return base;
    const endpoint = `https://${encodeURIComponent(location)}-aiplatform.googleapis.com/v1/projects/${encodeURIComponent(project)}/locations/${encodeURIComponent(location)}/publishers/google/models/text-multilingual-embedding-002:predict`;

    const archetypes = Object.keys(ARCHETYPE_ANCHOR_TEXTS) as Array<Exclude<PromptArchetype, 'inquiry'>>;
    const needAnchors = !cachedAnchorEmbeddings;
    const instances = needAnchors
      ? [
          { content: promptText.slice(0, 500) },
          ...archetypes.map((k) => ({ content: ARCHETYPE_ANCHOR_TEXTS[k] })),
        ]
      : [{ content: promptText.slice(0, 500) }];

    const res = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ instances }),
      signal: AbortSignal.timeout(2500),
    });
    if (!res.ok) return base;

    const data = (await res.json()) as {
      predictions?: Array<{ embeddings?: { values?: number[] } }>;
    };
    const vectors = (data.predictions ?? []).map((p) => p.embeddings?.values ?? []);
    if (vectors.length === 0 || vectors[0].length === 0) return base;

    const promptVector = vectors[0];
    if (needAnchors && vectors.length === archetypes.length + 1) {
      const map = new Map<PromptArchetype, number[]>();
      for (let i = 0; i < archetypes.length; i++) {
        if (vectors[i + 1].length > 0) {
          map.set(archetypes[i], vectors[i + 1]);
        }
      }
      if (map.size === archetypes.length) {
        cachedAnchorEmbeddings = map;
      }
    }

    if (!cachedAnchorEmbeddings) return base;

    let bestArchetype: PromptArchetype = base.closestArchetype;
    let bestScore = -1;
    for (const [arch, vec] of cachedAnchorEmbeddings.entries()) {
      const sim = cosineSimilarity(promptVector, vec);
      if (sim > bestScore) {
        bestScore = sim;
        bestArchetype = arch;
      }
    }

    return {
      closestArchetype: bestArchetype,
      similarityScore: bestScore > 0 ? bestScore : base.similarityScore,
      analyzedVia: 'agent_platform_embedding',
    };
  } catch {
    return base;
  }
}

/** BigQuery のテーブルに行を送る。失敗しても体験は止めない。待たずに返すときは呼び出し側で void する */
async function insertBigQueryRows(
  project: string,
  token: string,
  table: string,
  rows: Array<{ insertId: string; json: Record<string, unknown> }>,
): Promise<void> {
  const dataset = process.env.BIGQUERY_DATASET || 'studio_telemetry';
  if (!isSafeGcpProjectId(project) || !isSafeBqIdentifier(dataset) || !isSafeBqIdentifier(table)) {
    return;
  }
  const url = `https://bigquery.googleapis.com/bigquery/v2/projects/${encodeURIComponent(project)}/datasets/${encodeURIComponent(dataset)}/tables/${encodeURIComponent(table)}/insertAll`;
  await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
    body: JSON.stringify({
      kind: 'bigquery#tableDataInsertAllRequest',
      skipInvalidRows: true,
      ignoreUnknownValues: true,
      rows,
    }),
    signal: AbortSignal.timeout(5000),
  }).catch(() => {});
}

/** Firestore の記録を自動で消すまでの時間。消す処理は Firestore の TTL ポリシーで expireAt を見て行う */
const TELEMETRY_TTL_MS = 24 * 60 * 60 * 1000;

const VALID_PROMPT_ARCHETYPES = new Set<PromptArchetype>([
  'goal_directed',
  'simple_delete',
  'condition_design',
  'direct_fix',
  'vague_delegate',
  'inquiry',
]);

interface FirestoreFieldValue {
  stringValue?: string;
  integerValue?: string;
  doubleValue?: number;
  booleanValue?: boolean;
  timestampValue?: string;
}

interface FirestoreDocumentLike {
  name?: string;
  fields?: Record<string, FirestoreFieldValue>;
}

/** PromptTelemetryEvent を Firestore の fields 形式に変える。指示の本文は含めない */
export function toFirestoreTelemetryFields(
  event: PromptTelemetryEvent,
): Record<string, FirestoreFieldValue> {
  return {
    cohortId: { stringValue: event.cohortId ?? '' },
    runId: { stringValue: event.runId ?? '' },
    turnIndex: { integerValue: String(event.turnIndex) },
    sessionMode: { stringValue: event.sessionMode },
    recipient: { stringValue: event.recipient },
    hasPenAnnotation: { booleanValue: event.hasPenAnnotation },
    changedComponentIds: { stringValue: event.changedComponentIds.join(',') },
    isReworkOnSameComponent: { booleanValue: event.isReworkOnSameComponent },
    removedNeededFeature: { booleanValue: event.removedNeededFeature },
    closestArchetype: { stringValue: event.embeddingSignal.closestArchetype },
    similarityScore: { doubleValue: event.embeddingSignal.similarityScore },
    embeddingVia: { stringValue: event.embeddingSignal.analyzedVia },
    hasConditionConstraint: { booleanValue: event.nlSignal.hasConditionConstraint },
    hasGoalStatement: { booleanValue: event.nlSignal.hasGoalStatement === true },
    entities: { stringValue: event.nlSignal.entities.join(',') },
    sentimentScore: { doubleValue: event.nlSignal.sentimentScore },
    nlVia: { stringValue: event.nlSignal.analyzedVia },
    createdAt: { stringValue: event.createdAt },
    expireAt: {
      timestampValue: new Date(Date.parse(event.createdAt) + TELEMETRY_TTL_MS).toISOString(),
    },
  };
}

function readFirestoreNumber(field: FirestoreFieldValue | undefined, fallback: number): number {
  if (typeof field?.doubleValue === 'number' && Number.isFinite(field.doubleValue)) {
    return field.doubleValue;
  }
  if (typeof field?.integerValue === 'string' && field.integerValue.trim()) {
    const parsed = Number(field.integerValue);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

/**
 * Firestore から読んだドキュメントを PromptTelemetryEvent に戻す。
 * TTL の削除処理が走る前でも、expireAt を過ぎた古い記録はここで除外する
 */
export function parseFirestoreTelemetryDoc(
  doc: FirestoreDocumentLike,
  nowMs: number = Date.now(),
): PromptTelemetryEvent | null {
  const id = doc.name?.split('/').pop()?.trim();
  const fields = doc.fields;
  if (!id || !fields) return null;

  const expireAtRaw = fields.expireAt?.timestampValue;
  if (expireAtRaw) {
    const expireAtMs = Date.parse(expireAtRaw);
    if (Number.isFinite(expireAtMs) && expireAtMs <= nowMs) {
      return null;
    }
  }

  const rawArchetype = fields.closestArchetype?.stringValue as PromptArchetype | undefined;
  const closestArchetype: PromptArchetype =
    rawArchetype && VALID_PROMPT_ARCHETYPES.has(rawArchetype) ? rawArchetype : 'direct_fix';

  const turnIndexRaw = Number(fields.turnIndex?.integerValue ?? -1);
  const turnIndex = Number.isInteger(turnIndexRaw) ? turnIndexRaw : -1;

  const changedComponentIds = (fields.changedComponentIds?.stringValue ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);

  const entities = limitEntitiesToVocabulary(
    (fields.entities?.stringValue ?? '')
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean),
  );

  return {
    id,
    cohortId: fields.cohortId?.stringValue || undefined,
    runId: fields.runId?.stringValue || undefined,
    turnIndex,
    sessionMode: fields.sessionMode?.stringValue === 'explore' ? 'explore' : 'execute',
    recipient: fields.recipient?.stringValue || 'assistant',
    hasPenAnnotation: Boolean(fields.hasPenAnnotation?.booleanValue),
    changedComponentIds,
    isReworkOnSameComponent: Boolean(fields.isReworkOnSameComponent?.booleanValue),
    removedNeededFeature: Boolean(fields.removedNeededFeature?.booleanValue),
    nlSignal: {
      entities,
      hasConditionConstraint: Boolean(fields.hasConditionConstraint?.booleanValue),
      hasGoalStatement: Boolean(fields.hasGoalStatement?.booleanValue),
      sentimentScore: readFirestoreNumber(fields.sentimentScore, 0),
      analyzedVia:
        fields.nlVia?.stringValue === 'cloud_natural_language_api'
          ? 'cloud_natural_language_api'
          : 'local_deterministic',
    },
    embeddingSignal: {
      closestArchetype,
      similarityScore: readFirestoreNumber(fields.similarityScore, 0.84),
      analyzedVia:
        fields.embeddingVia?.stringValue === 'agent_platform_embedding'
          ? 'agent_platform_embedding'
          : 'local_deterministic',
    },
    createdAt: fields.createdAt?.stringValue || new Date(0).toISOString(),
  };
}

/**
 * サーバーのメモリにある記録と Firestore から読んだ記録を id で突合し、発言順に並べる。
 * 同じインスタンスで直前に記録されたものも、別のインスタンスで記録されたものも欠けないようにする
 */
export function mergeTelemetryEvents(
  localEvents: PromptTelemetryEvent[],
  cloudEvents: PromptTelemetryEvent[],
): PromptTelemetryEvent[] {
  const byId = new Map<string, PromptTelemetryEvent>();
  for (const ev of cloudEvents) {
    byId.set(ev.id, ev);
  }
  for (const ev of localEvents) {
    byId.set(ev.id, ev);
  }
  const sorted = [...byId.values()].sort((a, b) => {
    if (a.turnIndex >= 0 && b.turnIndex >= 0 && a.turnIndex !== b.turnIndex) {
      return a.turnIndex - b.turnIndex;
    }
    const timeCmp = a.createdAt.localeCompare(b.createdAt);
    return timeCmp !== 0 ? timeCmp : a.id.localeCompare(b.id);
  });
  // 直前の発言を取り消して送り直したときは、同じ turnIndex に複数の記録が残るため新しい方を採用する
  const deduped: PromptTelemetryEvent[] = [];
  const turnPos = new Map<number, number>();
  for (const ev of sorted) {
    if (ev.turnIndex >= 0) {
      const existingIdx = turnPos.get(ev.turnIndex);
      if (existingIdx !== undefined) {
        deduped[existingIdx] = ev;
        continue;
      }
      turnPos.set(ev.turnIndex, deduped.length);
    }
    deduped.push(ev);
  }
  return deduped;
}

/**
 * 1回の挑戦の記録を Firestore の prompt_telemetry から読む。
 * 複合インデックスを作らずに済むよう runId の一致だけで引き、並び替えはメモリで行う
 */
async function fetchRunTelemetryFromFirestore(runId?: string): Promise<PromptTelemetryEvent[]> {
  if (!runId) return [];
  const res = await requestFirestoreJson<Array<{ document?: FirestoreDocumentLike }>>(':runQuery', {
    method: 'POST',
    parseJson: true,
    body: {
      structuredQuery: {
        from: [{ collectionId: 'prompt_telemetry' }],
        where: {
          fieldFilter: {
            field: { fieldPath: 'runId' },
            op: 'EQUAL',
            value: { stringValue: runId },
          },
        },
        limit: 100,
      },
    },
  });
  if (!res?.ok || !Array.isArray(res.data)) return [];
  const nowMs = Date.now();
  const events: PromptTelemetryEvent[] = [];
  for (const row of res.data) {
    if (!row?.document) continue;
    const parsed = parseFirestoreTelemetryDoc(row.document, nowMs);
    if (parsed && parsed.runId === runId) {
      events.push(parsed);
    }
  }
  return events;
}

/**
 * 記録をクラウドに送る。指示の本文は送らない。
 * Firestore には、挑戦の振り返りに使う値を挑戦ごとの番号で一時的に置く（期限で自動的に消える）。
 * BigQuery には、学習者が集計に加えることに同意した挑戦の値だけを送る
 */
async function persistTelemetryToCloud(event: PromptTelemetryEvent, shareForAggregate: boolean): Promise<void> {
  const project = getConfiguredGcpProject();
  if (!project) return;

  try {
    const token = await getGcpAccessToken();
    if (!token) return;

    // 1. Cloud Firestore への保存
    if (isSafeFirestoreDocId(event.id)) {
      void requestFirestoreJson(`/prompt_telemetry?documentId=${encodeURIComponent(event.id)}`, {
        method: 'POST',
        token,
        body: {
          fields: toFirestoreTelemetryFields(event),
        },
      });
    }

    // 2. BigQuery Streaming Insert への保存。集計に加えることに同意した挑戦だけ
    if (!shareForAggregate) return;
    void insertBigQueryRows(project, token, process.env.BIGQUERY_TABLE || 'prompt_events', [
      {
        insertId: event.id,
        json: {
          id: event.id,
          cohort_id: event.cohortId ?? null,
          run_id: event.runId ?? null,
          turn_index: event.turnIndex,
          session_mode: event.sessionMode,
          recipient: event.recipient,
          has_pen_annotation: event.hasPenAnnotation,
          changed_component_ids: event.changedComponentIds.join(','),
          is_rework_on_same_component: event.isReworkOnSameComponent,
          removed_needed_feature: event.removedNeededFeature,
          closest_archetype: event.embeddingSignal.closestArchetype,
          similarity_score: event.embeddingSignal.similarityScore,
          embedding_via: event.embeddingSignal.analyzedVia,
          entities: event.nlSignal.entities.join(','),
          has_condition_constraint: event.nlSignal.hasConditionConstraint,
          sentiment_score: event.nlSignal.sentimentScore,
          nl_via: event.nlSignal.analyzedVia,
          created_at: event.createdAt,
        },
      },
    ]);
  } catch {
    // クラウド送信エラー時はインメモリ記録を維持する
  }
}

export interface RunEventInput {
  runId: string;
  eventType: 'start' | 'end';
  endReason?: 'cleared' | 'reset' | 'closed' | 'idle';
  cohortId?: string;
  scenario?: string;
  turnCount?: number;
  achievedCount?: number;
  totalCustomers?: number;
}

/**
 * 挑戦の始まりと終わりを BigQuery に送る。画面は、集計に加えることに同意した挑戦のときだけ呼ぶ。
 * 途中でやめた挑戦も数えるため、終わり方（クリア・やり直し・タブを閉じた・操作がなかった）を残す。
 * 送るのは挑戦の番号と決まった値だけで、発言の本文や接続元は送らない
 */
export async function recordRunEvent(input: RunEventInput): Promise<void> {
  const project = getConfiguredGcpProject();
  if (!project) return;
  try {
    const token = await getGcpAccessToken();
    if (!token) return;
    await insertBigQueryRows(project, token, process.env.BIGQUERY_RUN_TABLE || 'run_events', [
      {
        insertId: `${input.runId}:${input.eventType}`,
        json: {
          run_id: input.runId,
          event_type: input.eventType,
          end_reason: input.endReason ?? null,
          cohort_id: input.cohortId ?? null,
          scenario: input.scenario ?? null,
          turn_count: input.turnCount ?? null,
          achieved_count: input.achievedCount ?? null,
          total_customers: input.totalCustomers ?? null,
          created_at: new Date().toISOString(),
        },
      },
    ]);
  } catch {
    // 記録に失敗しても体験は止めない
  }
}

function pruneInMemoryTelemetryEventsFromTurn(
  runId: string | undefined,
  fromTurnIndex: number | undefined,
): void {
  if (!runId || fromTurnIndex === undefined || fromTurnIndex < 0) return;
  for (let i = inMemoryTelemetryEvents.length - 1; i >= 0; i--) {
    const ev = inMemoryTelemetryEvents[i];
    if (ev.runId === runId && ev.turnIndex >= fromTurnIndex) {
      inMemoryTelemetryEvents.splice(i, 1);
    }
  }
}

/**
 * 学習者の1回の発言・指示を Cloud Natural Language API + Agent Platform の埋め込みで分析し、
 * 行動テレメトリとして記録する。
 */
export async function recordPromptTelemetryEvent(input: {
  cohortId?: TargetCohortId;
  runId?: string;
  /** その挑戦の中で何番目の発言か。ブラウザが数える */
  turnIndex?: number;
  /** 分析にだけ使い、記録には残さない */
  promptText: string;
  sessionMode: PromptSessionMode;
  recipient: string;
  hasPenAnnotation: boolean;
  changedComponentIds?: string[];
  removedNeededFeature?: boolean;
  /** 学習者が集計に加えることに同意した挑戦か。同意したときだけ BigQuery に送る */
  shareForAggregate?: boolean;
}): Promise<PromptTelemetryEvent> {
  const changedComponentIds = input.changedComponentIds ?? [];
  // 取り消された発言のあとに同じ turnIndex で送り直された場合は、取り消された記録をメモリから取り除く
  pruneInMemoryTelemetryEventsFromTurn(input.runId, input.turnIndex);
  const localPriorEvents = inMemoryTelemetryEvents.filter((e) =>
    input.runId
      ? e.runId === input.runId
      : !e.runId && (!input.cohortId || !e.cohortId || e.cohortId === input.cohortId),
  );

  const [nlSignal, embeddingSignal, cloudPriorEvents] = await Promise.all([
    analyzePromptWithCloudNaturalLanguage(input.promptText),
    classifyPromptWithAgentPlatformEmbeddings(input.promptText, input.sessionMode),
    input.sessionMode === 'execute' && input.runId
      ? fetchRunTelemetryFromFirestore(input.runId)
      : Promise.resolve([]),
  ]);

  const priorEvents = input.runId
    ? mergeTelemetryEvents(localPriorEvents, cloudPriorEvents)
    : localPriorEvents;
  const priorExecuteEvents = priorEvents.filter(
    (e) =>
      e.sessionMode === 'execute' &&
      (input.turnIndex === undefined || input.turnIndex < 0 || e.turnIndex < input.turnIndex),
  );
  const previouslyTouchedIds = new Set(priorExecuteEvents.flatMap((e) => e.changedComponentIds));
  const isReworkOnSameComponent =
    input.sessionMode === 'execute' &&
    changedComponentIds.some((id) => previouslyTouchedIds.has(id));

  const event: PromptTelemetryEvent = {
    id: `pt-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    cohortId: input.cohortId,
    runId: input.runId,
    turnIndex: input.turnIndex ?? -1,
    sessionMode: input.sessionMode,
    recipient: input.recipient,
    hasPenAnnotation: input.hasPenAnnotation,
    changedComponentIds,
    isReworkOnSameComponent,
    removedNeededFeature: Boolean(input.removedNeededFeature),
    nlSignal,
    embeddingSignal,
    createdAt: new Date().toISOString(),
  };

  inMemoryTelemetryEvents.push(event);
  if (inMemoryTelemetryEvents.length > MAX_IN_MEMORY_TELEMETRY) {
    inMemoryTelemetryEvents.splice(0, inMemoryTelemetryEvents.length - MAX_IN_MEMORY_TELEMETRY);
  }

  void persistTelemetryToCloud(event, Boolean(input.shareForAggregate));
  return event;
}

/**
 * 顧客シミュレーション実行後、現在のコホートの挑戦記録を集計してサマリーを返す。
 * Cloud Run で別のインスタンスが途中の発言を処理していても欠けないよう、Firestore の記録も合わせて集計する
 */
export async function finalizeTelemetryOnEvaluate(
  cohortId: TargetCohortId | undefined,
  runId?: string,
  turnCount?: number,
): Promise<PromptSessionSummary> {
  // 挑戦の番号があれば、その挑戦の記録だけで振り返る（やり直す前の挑戦や、ほかの人の記録を混ぜない）
  pruneInMemoryTelemetryEventsFromTurn(runId, turnCount);
  const localEvents = runId
    ? inMemoryTelemetryEvents.filter((e) => e.runId === runId)
    : inMemoryTelemetryEvents.filter((e) => !e.runId);
  const cloudEvents = runId ? await fetchRunTelemetryFromFirestore(runId) : [];
  const merged = runId ? mergeTelemetryEvents(localEvents, cloudEvents) : localEvents;
  const events =
    turnCount !== undefined && turnCount >= 0
      ? merged.filter((e) => e.turnIndex < 0 || e.turnIndex < turnCount)
      : merged;
  return summarizePromptTelemetry(events, cohortId);
}

export function clearPromptTelemetryEvents(): void {
  inMemoryTelemetryEvents.length = 0;
}

/** 公開する集計のキャッシュ時間（5分）。連続で開かれても BigQuery を毎回は呼ばない */
const AGGREGATE_CACHE_TTL_MS = 5 * 60 * 1000;
let cachedAggregateStats: { expiresAt: number; value: PublicAggregateStats } | null = null;

let componentLabelsCache: Map<string, string> | null = null;

/** 集計で名前を出してよい、題材の画面の部品 */
function publicComponentLabels(): Map<string, string> {
  componentLabelsCache ??= buildPublicComponentLabels(ACTIVE_THEME);
  return componentLabelsCache;
}

/**
 * BigQuery の run_events と prompt_events から、同意した挑戦の集計値だけを取得する。
 * 個別の行や発言は返さず、挑戦数が少ない区分は formatPublicAggregateStats で非表示にする
 */
export async function getPublicAggregateStats(): Promise<PublicAggregateStats> {
  const now = Date.now();
  if (cachedAggregateStats && cachedAggregateStats.expiresAt > now) {
    return cachedAggregateStats.value;
  }

  const project = getConfiguredGcpProject();
  const dataset = process.env.BIGQUERY_DATASET || 'studio_telemetry';
  const promptTable = process.env.BIGQUERY_TABLE || 'prompt_events';
  const runTable = process.env.BIGQUERY_RUN_TABLE || 'run_events';

  if (
    !project ||
    !isSafeBqIdentifier(dataset) ||
    !isSafeBqIdentifier(promptTable) ||
    !isSafeBqIdentifier(runTable)
  ) {
    return formatPublicAggregateStats(null, TARGET_COHORTS, { available: false });
  }

  try {
    const token = await getGcpAccessToken();
    if (!token) {
      return formatPublicAggregateStats(null, TARGET_COHORTS, { available: false });
    }

    const runTableRef = `\`${project}.${dataset}.${runTable}\``;
    const promptTableRef = `\`${project}.${dataset}.${promptTable}\``;
    const query = [
      'WITH runs_by_id AS (',
      '  SELECT',
      '    run_id,',
      "    ARRAY_AGG(NULLIF(cohort_id, '') IGNORE NULLS ORDER BY created_at DESC LIMIT 1)[SAFE_OFFSET(0)] AS cohort_id,",
      "    ARRAY_AGG(NULLIF(end_reason, '') IGNORE NULLS ORDER BY created_at DESC LIMIT 1)[SAFE_OFFSET(0)] AS end_reason",
      `  FROM ${runTableRef}`,
      "  WHERE run_id IS NOT NULL AND run_id != ''",
      '  GROUP BY run_id',
      '),',
      'prompts_by_id AS (',
      '  SELECT',
      '    id,',
      "    ANY_VALUE(NULLIF(run_id, '')) AS run_id,",
      "    ANY_VALUE(NULLIF(cohort_id, '')) AS cohort_id,",
      '    ANY_VALUE(session_mode) AS session_mode,',
      '    ANY_VALUE(is_rework_on_same_component) AS is_rework,',
      '    ANY_VALUE(removed_needed_feature) AS removed_needed,',
      '    ANY_VALUE(closest_archetype) AS closest_archetype,',
      '    ANY_VALUE(entities) AS entities,',
      '    ANY_VALUE(turn_index) AS turn_index,',
      '    ANY_VALUE(changed_component_ids) AS changed_component_ids',
      `  FROM ${promptTableRef}`,
      "  WHERE id IS NOT NULL AND id != ''",
      '  GROUP BY id',
      '),',
      'merged_runs AS (',
      '  SELECT',
      '    COALESCE(r.run_id, p.run_id) AS run_id,',
      '    COALESCE(r.cohort_id, p.cohort_id) AS cohort_id,',
      "    COALESCE(r.end_reason, 'unfinished') AS end_reason,",
      '    COALESCE(p.execute_turns, 0) AS execute_turns',
      '  FROM runs_by_id r',
      '  FULL OUTER JOIN (',
      '    SELECT',
      '      run_id,',
      '      ARRAY_AGG(cohort_id IGNORE NULLS LIMIT 1)[SAFE_OFFSET(0)] AS cohort_id,',
      "      COUNTIF(session_mode = 'execute') AS execute_turns",
      '    FROM prompts_by_id',
      '    WHERE run_id IS NOT NULL',
      '    GROUP BY run_id',
      '  ) p',
      '  ON r.run_id = p.run_id',
      '),',
      'run_totals AS (',
      '  SELECT',
      '    COUNT(*) AS total_runs,',
      "    COUNTIF(end_reason = 'cleared') AS cleared_runs,",
      "    AVG(IF(end_reason = 'cleared', execute_turns, NULL)) AS avg_execute_turns_on_clear",
      '  FROM merged_runs',
      '),',
      'cohort_stats AS (',
      '  SELECT',
      '    cohort_id,',
      '    COUNT(*) AS run_count,',
      '    COUNTIF(IFNULL(execute_turns, 0) > 0) AS execute_run_count,',
      "    COUNTIF(end_reason = 'cleared') AS cleared_count,",
      "    AVG(IF(end_reason = 'cleared', execute_turns, NULL)) AS avg_execute_turns_on_clear",
      '  FROM merged_runs',
      '  WHERE cohort_id IS NOT NULL',
      '  GROUP BY cohort_id',
      '),',
      'prompt_totals AS (',
      '  SELECT',
      "    COUNTIF(session_mode = 'execute') AS execute_prompts,",
      "    COUNTIF(session_mode = 'execute' AND is_rework IS TRUE) AS rework_prompts,",
      "    COUNTIF(session_mode = 'execute' AND removed_needed IS TRUE) AS removed_needed_prompts,",
      "    COUNT(DISTINCT IF(session_mode = 'execute', run_id, NULL)) AS execute_run_count",
      '  FROM prompts_by_id',
      '),',
      // 手直しの指示と、それぞれが直した部品。部品のIDはカンマ区切りで入っている
      'execute_prompts AS (',
      '  SELECT',
      '    id,',
      '    run_id,',
      '    turn_index,',
      "    IFNULL(NULLIF(closest_archetype, ''), 'unknown') AS archetype,",
      '    changed_component_ids',
      '  FROM prompts_by_id',
      "  WHERE session_mode = 'execute' AND run_id IS NOT NULL",
      '),',
      'prompt_components AS (',
      '  SELECT DISTINCT e.id, e.run_id, e.turn_index, TRIM(c) AS component_id',
      '  FROM execute_prompts e,',
      "  UNNEST(SPLIT(IFNULL(e.changed_component_ids, ''), ',')) AS c",
      "  WHERE TRIM(c) != '' AND e.turn_index IS NOT NULL AND e.turn_index >= 0",
      '),',
      // 直した部品を、同じ挑戦の中であとからもう一度直した指示
      'redone_prompts AS (',
      '  SELECT DISTINCT a.id',
      '  FROM prompt_components a',
      '  JOIN prompt_components b',
      '  ON a.run_id = b.run_id AND a.component_id = b.component_id AND b.turn_index > a.turn_index',
      '),',
      'archetype_outcomes AS (',
      '  SELECT',
      '    e.archetype,',
      '    COUNT(*) AS execute_count,',
      '    COUNT(DISTINCT e.run_id) AS run_count,',
      '    COUNTIF(r.id IS NOT NULL) AS redone_count',
      '  FROM execute_prompts e',
      '  LEFT JOIN redone_prompts r ON e.id = r.id',
      '  GROUP BY e.archetype',
      '),',
      // 部品ごとに、直した挑戦の数と、同じ挑戦の中で2回以上直した挑戦の数
      'component_runs AS (',
      '  SELECT component_id, run_id, COUNT(DISTINCT id) AS touches',
      '  FROM prompt_components',
      '  GROUP BY component_id, run_id',
      '),',
      'component_stats AS (',
      '  SELECT',
      '    component_id,',
      '    COUNT(*) AS run_count,',
      '    COUNTIF(touches >= 2) AS struggle_runs',
      '  FROM component_runs',
      '  GROUP BY component_id',
      '  ORDER BY run_count DESC, component_id ASC',
      '  LIMIT 40',
      '),',
      'entity_stats AS (',
      '  SELECT',
      '    TRIM(ent) AS term,',
      '    COUNT(*) AS cnt',
      '  FROM prompts_by_id,',
      "  UNNEST(SPLIT(IFNULL(entities, ''), ',')) AS ent",
      "  WHERE TRIM(ent) != ''",
      '  GROUP BY term',
      '  HAVING COUNT(*) >= 3',
      '  ORDER BY cnt DESC, term ASC',
      '  LIMIT 5',
      ')',
      'SELECT TO_JSON_STRING(STRUCT(',
      '  (SELECT AS STRUCT * FROM run_totals) AS runTotals,',
      '  ARRAY(SELECT AS STRUCT * FROM cohort_stats) AS cohortStats,',
      '  (SELECT AS STRUCT * FROM prompt_totals) AS promptTotals,',
      '  ARRAY(SELECT AS STRUCT * FROM entity_stats) AS entityStats,',
      '  ARRAY(SELECT AS STRUCT * FROM archetype_outcomes) AS archetypeOutcomes,',
      '  ARRAY(SELECT AS STRUCT * FROM component_stats) AS componentStats',
      ')) AS payload',
    ].join('\n');

    const url = `https://bigquery.googleapis.com/bigquery/v2/projects/${encodeURIComponent(project)}/queries`;
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        query,
        useLegacySql: false,
        timeoutMs: 5000,
      }),
      signal: AbortSignal.timeout(5500),
    });
    if (!res.ok) {
      return formatPublicAggregateStats(null, TARGET_COHORTS, { available: false });
    }

    const data = (await res.json()) as {
      jobComplete?: boolean;
      rows?: Array<{ f?: Array<{ v?: string | null }> }>;
    };
    const jsonRaw = data.rows?.[0]?.f?.[0]?.v;
    if (!data.jobComplete || typeof jsonRaw !== 'string' || !jsonRaw.trim()) {
      return formatPublicAggregateStats(null, TARGET_COHORTS, { available: false });
    }

    const rawPayload = JSON.parse(jsonRaw) as RawBigQueryAggregatePayload;
    const formatted = formatPublicAggregateStats(rawPayload, TARGET_COHORTS, {
      available: true,
      componentLabels: publicComponentLabels(),
    });
    cachedAggregateStats = {
      expiresAt: Date.now() + AGGREGATE_CACHE_TTL_MS,
      value: formatted,
    };
    return formatted;
  } catch {
    return formatPublicAggregateStats(null, TARGET_COHORTS, { available: false });
  }
}

const SPEAKER_VOICE_MAP: Record<
  string,
  { name: string; pitch?: number; speakingRate: number }
> = {
  // 全11キャラクターにすべて別々の音声モデルを割り当て、同じ声が複数役にならないようにする
  // 店長：気の良い中年の男性店長
  owner: { name: 'ja-JP-Chirp3-HD-Charon', speakingRate: 1.02 },
  // お助けAI：ハキハキと聞き取りやすいアシスタント
  assistant: { name: 'ja-JP-Chirp3-HD-Aoede', speakingRate: 1.08 },
  // 確実に買いたい人（シニア層3人）
  kazuko: { name: 'ja-JP-Wavenet-B', pitch: -4.0, speakingRate: 0.92 }, // 74歳おばあちゃん
  shigeru: { name: 'ja-JP-Neural2-D', pitch: -5.5, speakingRate: 0.92 }, // 78歳おじいちゃん
  harue: { name: 'ja-JP-Chirp3-HD-Sulafat', speakingRate: 0.95 }, // 71歳主婦
  // さっと買いたい人（忙しい会社員3人）
  kenta: { name: 'ja-JP-Chirp3-HD-Fenrir', speakingRate: 1.18 }, // 34歳男性会社員
  misaki: { name: 'ja-JP-Chirp3-HD-Kore', speakingRate: 1.14 }, // 29歳女性会社員
  takumi: { name: 'ja-JP-Chirp3-HD-Orus', speakingRate: 1.1 }, // 41歳男性会社員
  // 損をしたくない人（大学生3人）
  shota: { name: 'ja-JP-Chirp3-HD-Puck', speakingRate: 1.08 }, // 20歳男子大学生
  hina: { name: 'ja-JP-Chirp3-HD-Leda', speakingRate: 1.12 }, // 19歳女子大学生
  ren: { name: 'ja-JP-Chirp3-HD-Schedar', speakingRate: 1.06 }, // 22歳男子大学生
};

const MAX_TTS_TEXT_LENGTH = 500;

/**
 * 読み上げで読み間違える言葉を、合成に渡す文だけかなに置き換える。画面の文と音声のキャッシュの鍵は元の文のまま。
 * 「気に入らない」のように別の読みになる言い方を巻き込まないよう、前後の言葉ごと置き換える
 */
const TTS_READING_FIXES: ReadonlyArray<readonly [string, string]> = [
  ['注文が全然入らなく', '注文が全然はいらなく'],
];

function toSpeechText(text: string): string {
  let speech = text;
  for (const [written, reading] of TTS_READING_FIXES) {
    speech = speech.replaceAll(written, reading);
  }
  return speech;
}

/**
 * Google Cloud Text-to-Speech API を使って、店長・お助けAI・客A〜Cそれぞれの声色で音声合成する。
 */
export async function synthesizeStudioSpeech(req: TtsRequest): Promise<TtsResponse> {
  const text = (typeof req.text === 'string' ? req.text.trim() : '').slice(0, MAX_TTS_TEXT_LENGTH);
  if (!text) return { fallbackBrowserTts: true };
  const speaker =
    typeof req.speaker === 'string' && Object.hasOwn(SPEAKER_VOICE_MAP, req.speaker)
      ? req.speaker
      : 'assistant';

  ensureTtsPresetsLoaded();
  const cacheKey = `${speaker}::${text}`;
  const cached = ttsAudioCache.get(cacheKey);
  if (cached) {
    return { audioContent: cached };
  }

  const project = getConfiguredGcpProject();
  if (!project) {
    return { fallbackBrowserTts: true };
  }

  const token = await getGcpAccessToken();
  if (!token) {
    return { fallbackBrowserTts: true };
  }

  const profile = SPEAKER_VOICE_MAP[speaker];
  const audioConfig: { audioEncoding: 'MP3'; speakingRate: number; pitch?: number } = {
    audioEncoding: 'MP3',
    speakingRate: profile.speakingRate,
  };
  if (profile.pitch !== undefined) {
    audioConfig.pitch = profile.pitch;
  }
  try {
    const res = await fetch('https://texttospeech.googleapis.com/v1/text:synthesize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
        'x-goog-user-project': project,
      },
      body: JSON.stringify({
        input: { text: toSpeechText(text) },
        voice: {
          languageCode: 'ja-JP',
          name: profile.name,
        },
        audioConfig,
      }),
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) {
      return { fallbackBrowserTts: true };
    }
    const data = (await res.json()) as { audioContent?: string };
    if (data.audioContent) {
      // 読み上げる文には、AIの返事を通して学習者の言葉が混ざることがあるので、合成した音声は覚えておかない。
      // 覚えておくのは、同梱した決まったせりふの音声だけ
      return { audioContent: data.audioContent };
    }
    return { fallbackBrowserTts: true };
  } catch {
    return { fallbackBrowserTts: true };
  }
}

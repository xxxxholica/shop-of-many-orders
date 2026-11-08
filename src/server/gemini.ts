import {
  GoogleGenAI,
  Type,
  type FunctionDeclaration,
  type GenerateContentParameters,
  type GenerateContentResponse,
  type Schema,
  type ThinkingLevel,
} from '@google/genai';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  ACTIVE_THEME,
  ALL_REVIEWER_PERSONAS,
  getReviewersByCohort,
  TARGET_COHORTS,
} from '../shared/initialTheme.ts';
import { buildScenarioSpec } from '../shared/theme.ts';
import { hasCompletionAction } from '../shared/appRuntime.ts';
import { describeOtherCohortImpact } from '../shared/otherCohorts.ts';
import {
  buildOwnerChangeReaction,
  buildOwnerClosingLines,
  buildOwnerReport,
  describeOwnerChange,
} from '../shared/owner.ts';
import {
  COUPON_FRICTIONS,
  simulatePersonaComputerUse,
  validateSandboxSpec,
} from '../shared/sandboxValidator.ts';
import { dailyGeminiCounter } from './rateLimit.ts';
import { describeErrorForLog } from './logSafe.ts';
import {
  classifyUserMessageIntent,
  detectExplicitRecipientMention,
} from '../shared/intent.ts';
import { applySpecPatches } from '../shared/specPatch.ts';
import {
  buildClearFeedbackFacts,
  inspectAppliedChangeForTelemetry,
  type ClearFeedbackFacts,
  type ClearFeedbackInput,
} from '../shared/promptTelemetry.ts';
import { describeWorkSteps } from '../shared/workSteps.ts';

export { classifyUserMessageIntent, detectExplicitRecipientMention };
import type {
  AppComponent,
  ApplyRequest,
  ApplyResponse,
  AppSpec,
  ChatHistoryEntry,
  ChatRequest,
  ChatResponse,
  EvaluateRequest,
  EvaluationResponse,
  HumanDecisionRecord,
  LatestReviewRecord,
  OwnerLineRequest,
  OwnerLineResponse,
  RecordDecisionRequest,
  RecordDecisionResponse,
  ReviewerEvaluation,
  ReviewOutcome,
  SpecPatch,
  TranscribeRequest,
  TranscribeResponse,
} from '../shared/types.ts';
import {
  finalizeTelemetryOnEvaluate,
  recordPromptTelemetryEvent,
  saveHumanDecision,
} from './store.ts';

let envLoaded = false;

// ローカルの .env / .env.local を外部パッケージなしで初回のみ読み込む
function loadLocalEnv(): void {
  if (envLoaded) return;
  envLoaded = true;
  for (const file of ['.env.local', '.env']) {
    const fullPath = resolve(process.cwd(), file);
    if (!existsSync(fullPath)) continue;
    try {
      const content = readFileSync(fullPath, 'utf-8');
      for (const rawLine of content.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (!line || line.startsWith('#')) continue;
        const eqIdx = line.indexOf('=');
        if (eqIdx <= 0) continue;
        const key = line.slice(0, eqIdx).trim();
        const val = line.slice(eqIdx + 1).trim().replace(/^['"]|['"]$/g, '');
        if (!(key in process.env) && val) {
          process.env[key] = val;
        }
      }
    } catch {
      // 読み込み失敗時は無視
    }
  }
}

loadLocalEnv();

let cachedClientInfo: {
  cacheKey: string;
  client: GoogleGenAI;
  model: string;
  mode: string;
} | null = null;

type GenAIClientInfo = { client: GoogleGenAI; model: string; mode: string };

/** Gemini の接続先（APIキー または Google Cloud プロジェクト）が未設定のときに投げるエラー */
export class AiNotConfiguredError extends Error {
  readonly statusCode = 503;
  constructor() {
    super(
      'Gemini の接続設定がありません。.env.local に GEMINI_API_KEY または GOOGLE_CLOUD_PROJECT を設定してください。',
    );
    this.name = 'AiNotConfiguredError';
  }
}

function createGenAIClient(): GenAIClientInfo | null {
  loadLocalEnv();
  const model = process.env.GEMINI_MODEL || 'gemini-3.8-flash';
  const apiKey = process.env.GEMINI_API_KEY;
  const project = process.env.GOOGLE_CLOUD_PROJECT;
  const location = process.env.GOOGLE_CLOUD_LOCATION || 'global';

  if (apiKey) {
    const cacheKey = `key:${apiKey}:${model}`;
    if (cachedClientInfo?.cacheKey !== cacheKey) {
      cachedClientInfo = {
        cacheKey,
        client: new GoogleGenAI({ apiKey }),
        model,
        mode: 'gemini-api-key',
      };
    }
    return cachedClientInfo;
  }
  if (project && project !== 'your-gcp-project-id') {
    const cacheKey = `agent-platform:${project}:${location}:${model}`;
    if (cachedClientInfo?.cacheKey !== cacheKey) {
      cachedClientInfo = {
        cacheKey,
        client: new GoogleGenAI({
          // Gemini Enterprise Agent Platform（旧称 Vertex AI）を使う指定。引数名は SDK が決めた旧称のまま
          vertexai: true,
          project,
          location,
        }),
        model,
        mode: `Gemini Enterprise Agent Platform (${project}/${location})`,
      };
    }
    return cachedClientInfo;
  }
  return null;
}

function requireGenAIClient(): GenAIClientInfo {
  const info = createGenAIClient();
  if (!info) {
    throw new AiNotConfiguredError();
  }
  return info;
}

/**
 * Gemini が答えを考える量。どの呼び出しも短い構造化データを返すだけなので、既定は LOW にする。
 * 計測では既定の設定に比べて、改修の正確さは変わらずに待ち時間が数分の1に縮み、ばらつきも小さくなった。
 * GEMINI_THINKING_LEVEL で変えられ、空文字にするとモデルの既定に任せる（考える量を指定できないモデル向け）。
 */
function thinkingConfig(): { thinkingConfig?: { thinkingLevel: ThinkingLevel } } {
  const level = process.env.GEMINI_THINKING_LEVEL ?? 'LOW';
  return level ? { thinkingConfig: { thinkingLevel: level as ThinkingLevel } } : {};
}

/**
 * Gemini の応答はまれに数分固まることがあるため、呼び出しごとに待つ時間の上限を決める。
 * 短い返事はすぐ切ってやり直す。画面の改造は確認画面のように大きくなると時間がかかるので、長めに待ってやり直さない。
 * それでも返らなければ、学習者にもう一度送ってもらうよう伝えるエラーにする
 */
const TIMEOUTS = {
  /** 質問への回答・囲みへの反応 */
  reply: { timeoutMs: 15_000, retries: 0 },
  /** 画面の改造 */
  apply: { timeoutMs: 90_000, retries: 0 },
  /** お客さんの感想（間に合わなければシミュレーターの言葉を使う） */
  review: { timeoutMs: 8_000, retries: 0 },
  /** 店長のせりふ・判断の記録（間に合わなければ決まった形で済ませる） */
  side: { timeoutMs: 8_000, retries: 0 },
} as const;

export class AiTimeoutError extends Error {
  readonly statusCode = 504;
  constructor() {
    super('お助けAIの返事が遅れています。少し待ってから、もう一度送ってください。');
    this.name = 'AiTimeoutError';
  }
}

async function generateContentWithTimeout(
  aiInfo: GenAIClientInfo,
  params: GenerateContentParameters,
  { timeoutMs, retries }: { timeoutMs: number; retries: number },
): Promise<GenerateContentResponse> {
  let rateLimitRetried = false;
  for (let attempt = 0; attempt <= retries; attempt++) {
    // 1日あたりの上限を数える。やり直しも1回として数え、上限に達したらそれ以上呼ばない
    dailyGeminiCounter.consume();
    try {
      return await aiInfo.client.models.generateContent({
        ...params,
        config: { ...params.config, abortSignal: AbortSignal.timeout(timeoutMs) },
      });
    } catch (err) {
      const status = (err as { status?: number })?.status;
      const msg = String((err as Error)?.message ?? err);
      const isRateLimited =
        status === 429 || msg.includes('429') || msg.includes('RESOURCE_EXHAUSTED');
      if (isRateLimited && !rateLimitRetried) {
        rateLimitRetried = true;
        attempt--;
        console.warn('[gemini] 利用枠（429）に当たったため2秒待って1回やり直します');
        await new Promise((resolve) => setTimeout(resolve, 2_000));
        continue;
      }
      const name = (err as Error)?.name;
      const timedOut = name === 'TimeoutError' || name === 'AbortError';
      if (!timedOut) throw err;
      console.warn(`[gemini] ${timeoutMs}ms 以内に応答がなかったため${attempt < retries ? 'やり直します' : '諦めます'}`);
    }
  }
  throw new AiTimeoutError();
}

const MAX_HISTORY_ENTRIES = 12;

/** 直近の会話をプロンプト用の文字列にする */
function formatHistory(history: ChatHistoryEntry[] | undefined): string {
  const recent = (history ?? []).slice(-MAX_HISTORY_ENTRIES);
  if (recent.length === 0) return 'なし';
  return recent.map((h) => `${h.speaker}: ${h.text}`).join('\n');
}

/**
 * Gemini がまれに混ぜる中国語の簡体字。日本語では使わない字だけを並べる
 * （点・来・会・国・学など日本語と同じ形の字は入れない）
 */
const SIMPLIFIED_ONLY_CHARS =
  '们这个说时间为门见长车东问么还过对开关发经现进动实头样书钱让请认识边应该乐习电话语买卖气马鱼鸟爱难观听记论读谁给红绿蓝颜页题顾风饭华图园厅广张归总战报换无显杀杂权极检欢毕汇测灭热爷环确种积稳穷笔简类紧约级纪纸线组细终结绝统继续维编网罗职联脑脸节苏药获营虽补视觉计认讨训讲许设访证评词试诚询详误课调谈谢负财责败货质购贵费资赏赛赞赶轮软轻载较辆输达运远违连迟适选递邮释针钟钢铁银锁错键镜闭闲闹闻队阳阴阶际陆隐顶项顺须预领频额饮馆驾验骂鲜鸡齐龙';
const SIMPLIFIED_PATTERN = new RegExp(`[${SIMPLIFIED_ONLY_CHARS}]`);

/** 簡体字が混ざった文だけを取り除く。全部取り除かれたら代わりの文を使う */
function cleanJapaneseText(text: string, fallback: string): string {
  if (!SIMPLIFIED_PATTERN.test(text)) return text;
  const kept = text
    .split(/(?<=[。！？!?])/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence && !SIMPLIFIED_PATTERN.test(sentence));
  return kept.length > 0 ? kept.join('') : fallback;
}

const OWNER_LINE_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    lines: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: '店長のせりふ。1つが1つの吹き出しになる。1つ35文字以内、括弧なし',
    },
  },
  required: ['lines'],
};

/**
 * 店長のせりふをAIに作らせる。何が起きたか（facts）はコードで決めたものだけを渡し、言い方だけを任せる。
 * 同じ言い回しが続かないよう、これまでの店長のせりふも渡す。AIが使えないときは決まったせりふを返す
 */
export async function voiceOwnerLine(req: OwnerLineRequest): Promise<OwnerLineResponse> {
  if (req.facts.length === 0) return { text: '' };
  const aiInfo = createGenAIClient();
  if (!aiInfo) return { text: req.fallback };
  const owner = ACTIVE_THEME.owner;
  const pastLines = (req.history ?? [])
    .filter((h) => h.speaker === owner.name)
    .slice(-8)
    .map((h) => h.text);

  const situation =
    req.situation === 'change'
      ? 'お手伝いに来てくれた人が、あなたのお店の注文画面を次のように変えました。それを見た直後の一言を話してください。'
      : req.situation === 'closing'
        ? '選んだお客さん全員が満足して買ってくれました。お手伝いの人への締めのお礼を、2つの吹き出しで話してください。'
        : [
            `お手伝いに来てくれた人が、あなたにこう話しかけました：${req.userMessage ?? ''}`,
            'まず相手の言葉に素直に応えてください。お礼にはお礼を、質問には答えを返します。お店の状況の話は、話の流れに必要なときだけにしてください。',
            '意見や説得には、店長として本音で返事をしてください。お客さんを大事にしたい気持ちと、売上が減る不安の両方を正直に話してください。',
            '相手の言うことに一理あれば認めてかまいませんが、言いなりにはならないでください。画面をどうするかは、お手伝いの人が決めることです。',
            '画面を変えると約束したり、こう直してと頼んだりはしないでください。',
          ].join('\n');

  const prompt = [
    `あなたは次のような人物です。${owner.voice}`,
    situation,
    req.situation === 'reply' ? '【今のお店の状況（起きたこととして扱う）】' : '【起きたこと】',
    ...req.facts.map((f) => `- ${f}`),
    '【絶対ルール】',
    '1. 起きたことにないことは言わないでください。件数などの数字は、起きたことに書かれたとおりにしてください。',
    '2. どこを直せばよいか、次に何をすべきかといった指示や助言は言わないでください。店長としての気持ちだけを話してください。',
    '3. お客さんの気持ちを代わりに言わないでください。',
    '4. これまでの店長のせりふと同じ言い回しや、同じ書き出しは使わないでください。',
    '5. 吹き出しは change と reply なら1〜2つ、closing なら2つ。1つ35文字以内、括弧なし、日本語だけで書いてください。',
    `【話し方の見本（言い回しはまねしない）】\n${[owner.lines.completionFixed, owner.lines.popupRemoved, owner.lines.metricUnchecked].join('\n')}`,
    `【これまでの店長のせりふ】\n${pastLines.length > 0 ? pastLines.join('\n') : 'なし'}`,
  ].join('\n');

  try {
    const response = await generateContentWithTimeout(aiInfo, {
      model: aiInfo.model,
      contents: prompt,
      config: {
        responseMimeType: 'application/json',
        responseSchema: OWNER_LINE_SCHEMA,
        temperature: 0.9,
        ...thinkingConfig(),
      },
    }, TIMEOUTS.side);
    const parsed = parseJsonResponse<{ lines: string[] }>(response.text, 'owner');
    const lines = (parsed.lines ?? [])
      .map((l) => cleanJapaneseText(l.trim(), ''))
      .filter(Boolean)
      .slice(0, 2);
    return { text: lines.length > 0 ? lines.join('\n') : req.fallback };
  } catch (err) {
    console.warn('[gemini:owner] 店長のせりふを作れなかったため決まったせりふを使います:', describeErrorForLog(err));
    return { text: req.fallback };
  }
}

const OWNER_REPLY_FALLBACK = 'うーん、言いたいことは分かるんだけどなあ…売上のことを考えると悩ましいよ';

/** 赤ペンで部品を囲んだときに出す、調べるための質問候補（手直しの答えは出さない） */
export const CIRCLE_INQUIRY_FOLLOW_UPS = [
  'この部品の仕組みを聞く',
  'お客さんの反応を聞く',
  '店長の考えを聞く',
] as const;

function formatCircledTargetLabel(targets: AppComponent[]): string {
  const names = targets
    .map(
      (t) =>
        t.label
          .replace(/\{countdown\}/g, '2:59')
          .replace(/\{[^}]+\}/g, '')
          .replace(/\s+/g, ' ')
          .trim() ||
        t.caption?.trim() ||
        '',
    )
    .filter(Boolean)
    .map((n) => (n.length > 18 ? `${n.slice(0, 18)}…` : n));
  return names.length > 0 ? names.slice(0, 2).join('と') : 'この部品';
}

/**
 * 店長が話しかけられたときに使う、今のお店の状況の事実。数字や事実をでっちあげないよう、
 * 画面とお客さんの結果から分かることだけをコードでまとめる
 */
export function describeOwnerContext(
  spec: AppSpec,
  latestReviews: LatestReviewRecord | undefined,
  targetComponentIds?: string[],
): string[] {
  const owner = ACTIVE_THEME.owner;
  const facts: string[] = [];
  if (targetComponentIds && targetComponentIds.length > 0) {
    const targetComponents = spec.components.filter((c) => targetComponentIds.includes(c.id));
    if (targetComponents.length > 0) {
      facts.push(
        `いま相手が赤ペンで指して聞いている部品は「${formatCircledTargetLabel(targetComponents)}」である`,
      );
    }
  }
  const state = Object.fromEntries(spec.state.map((s) => [s.key, s.initialValue]));
  for (const m of owner.metrics) {
    // 店にもともとない項目（状態の変数がない）は伝えない。消された欄を嘆くのは、店にあったものだけ
    if (!(m.stateKey in state)) continue;
    const comp = spec.components.find((c) => c.stateKey === m.stateKey);
    if (!comp) {
      facts.push(`今の画面には、店長が売上のために付けた${m.label}の欄がない`);
    } else if (m.countWhen === true && state[m.stateKey] === true) {
      facts.push(`今の画面では、${m.label}に最初からチェックが入っている`);
    } else {
      facts.push(`今の画面にも${m.label}の欄はあるが、最初からはチェックが入っていない`);
    }
  }
  const labelsByRole = (role: string) =>
    spec.components.filter((c) => c.role === role).map((c) => `「${c.label.replace(/\{countdown\}/g, '2:59')}」`);
  const popups = labelsByRole('popup');
  if (popups.length > 0) facts.push(`今の画面には、店長が付けた広告${popups.join('と')}がある`);
  const pushes = [...labelsByRole('decoy'), ...labelsByRole('distraction')];
  if (pushes.length > 0) facts.push(`今の画面には、店長が売上のために目立たせた表示${pushes.join('と')}がある`);
  if (latestReviews && latestReviews.reviews.length > 0) {
    const bought = latestReviews.reviews.filter((r) => r.outcome !== 'abandoned').length;
    const happy = latestReviews.reviews.filter((r) => r.outcome === 'achieved').length;
    const where = latestReviews.isCurrentScreen ? '今の画面' : 'ひとつ前に試した画面';
    facts.push(
      `${where}では、来たお客さん${latestReviews.reviews.length}人のうち${bought}人が買い、${happy}人が満足していた`,
    );
    for (const m of latestReviews.ownerMetrics ?? []) {
      if (m.missing) continue;
      facts.push(`${where}では、${m.label}の申し込みが${m.count}件あった`);
    }
  } else {
    facts.push('まだお客さんに試してもらっていない');
  }
  return facts;
}

/** お助けエージェントが使い方を聞かれたときに答える、スタジオの使い方 */
const STUDIO_GUIDE = [
  '左のスマホは本物のように触れます。下の操作するを選ぶと、ボタンを押したり選んだりできます。',
  '赤ペンで囲むを選んでスマホの部品を囲み、チャットで指示を送ると、その部品を話題にできます。囲まずに文章だけで頼んでも大丈夫です。',
  'チャットで手直しを頼むと、お助けAIが画面を書き換えます。「1つ前の画面に戻して」「最初の画面に戻して」とチャットで伝えると、前の画面に戻れます。',
  'お客さんの反応を見るを押すと、選んだ3人がその画面で実際に買ってみて、感想を言います。',
  '入力欄の上のボタンで、お助けAI・店長・お客さんの誰に話しかけるかを選べます。初期設定はお助けAIで、文頭や@で別の人を明確に呼びかけると宛先も切り替わります。',
  '目標は、選んだ3人全員が満足して買える画面にすることです。',
  '誰のために直すかは途中では変えられません。選んだ3人全員が満足して買えると、直した画面のまま別のお客さんを呼ぶか、最初の画面からやり直すかを選べます。',
];

const CUSTOMER_COHORT_REPLY_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    replies: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          personaId: { type: Type.STRING },
          text: {
            type: Type.STRING,
            description: 'そのお客さん本人になりきった返答1文（35文字以内、括弧なし、直し方の指示は言わない）',
          },
        },
        required: ['personaId', 'text'],
      },
    },
  },
  required: ['replies'],
};

/**
 * お客さんに話しかけられたときに、選んだ客層の3人それぞれが自分の立場・体験から一言ずつ返事をする。
 * 実際に画面を操作した記録（actionTrace）だけを根拠にし、改善指示は言わせない。
 */
async function voiceCohortCustomers(
  req: ChatRequest,
): Promise<Array<{ personaId: string; text: string }>> {
  const reviewers = getReviewersByCohort(req.cohortId);
  const sims = reviewers.map((persona) => {
    const sim = simulatePersonaComputerUse(req.currentSpec, persona);
    const latestForPersona = req.latestReviews?.reviews.find((r) => r.name === persona.name);
    const trace = latestForPersona?.actionTrace ?? sim.actionTrace;
    const outcome = latestForPersona?.outcome ?? sim.outcome;
    const fallback = sim.simulatedComment || persona.greeting;
    return { persona, sim, trace, outcome, fallback };
  });

  const fallbackReplies = sims.map(({ persona, fallback }) => ({
    personaId: persona.id,
    text: fallback,
  }));

  const aiInfo = createGenAIClient();
  if (!aiInfo) return fallbackReplies;

  const customerProfiles = sims.map(({ persona, trace, outcome, fallback }) => ({
    personaId: persona.id,
    name: persona.name,
    role: persona.role,
    trait: persona.traitLabel,
    goal: persona.goal,
    voiceSample: `${persona.greeting} / ${fallback}`,
    outcome: OUTCOME_LABELS[outcome],
    actionTrace: trace,
  }));

  const targetComponents = req.currentSpec.components.filter((c) =>
    req.targetComponentIds.includes(c.id),
  );
  const targetNote =
    targetComponents.length > 0
      ? `相手は画面の部品である ${formatCircledTargetLabel(targetComponents)} を指して聞いています。`
      : '';

  const prompt = [
    `あなたは注文画面を使いに来た3人のお客さんです。お店の画面を手直ししている人から、お客さんに向けてこう質問・声かけされました：${req.userMessage}`,
    ...(targetNote ? [targetNote] : []),
    '【3人のお客さんそれぞれのプロフィールと実際に画面を操作した記録】',
    JSON.stringify(customerProfiles),
    '【絶対ルール】',
    '1. 3人それぞれ（personaId ごと）が、自分の実際の操作記録と希望だけを根拠に、本人らしい口調で1文ずつ素直に答えてください。',
    '2. こう直して・ボタンを大きくして・色を揃えて等の改善指示やデザイン用語は絶対に言わず、どこで迷ったか・何が困ったか・何が欲しかったか・どう感じたかだけを話してください。',
    '3. 各人の text は35文字以内、括弧なし、日本語だけで書いてください。',
    `【これまでの会話】\n${formatHistory(req.history)}`,
  ].join('\n');

  try {
    const response = await generateContentWithTimeout(
      aiInfo,
      {
        model: aiInfo.model,
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          responseSchema: CUSTOMER_COHORT_REPLY_SCHEMA,
          temperature: 0.7,
          ...thinkingConfig(),
        },
      },
      TIMEOUTS.side,
    );
    const parsed = parseJsonResponse<{
      replies?: Array<{ personaId?: string; text?: string }>;
    }>(response.text, 'customer-cohort');
    return sims.map(({ persona, fallback }) => {
      const found = (parsed.replies ?? []).find((r) => r.personaId === persona.id);
      const cleaned = cleanJapaneseText((found?.text ?? '').trim(), '');
      return {
        personaId: persona.id,
        text: cleaned || fallback,
      };
    });
  } catch (err) {
    console.warn('[gemini:customer] お客さんの返事を作れなかったためシミュレーターの言葉を使います:', describeErrorForLog(err));
    return fallbackReplies;
  }
}

const OUTCOME_LABELS: Record<ReviewOutcome, string> = {
  achieved: '満足して買えた',
  abandoned: '途中でやめた',
  regretted: '買えたが後悔した',
  bothered: '買えたが途中で嫌な思いをした',
};

/** お客さんが実際に操作した記録をプロンプト用の文字列にする */
function formatLatestReviews(record: LatestReviewRecord | undefined): string {
  if (!record || record.reviews.length === 0) return 'まだお客さんの反応を見ていません';
  const header = record.isCurrentScreen
    ? `今の画面での記録です`
    : `${record.screenLabel}の画面での記録です。今の画面はそのあと変わっています`;
  const lines = record.reviews.map(
    (r) =>
      `${r.name}: ${OUTCOME_LABELS[r.outcome]}\n${r.actionTrace.map((t) => `  ${t}`).join('\n')}`,
  );
  return [header, ...lines].join('\n');
}

function parseJsonResponse<T>(text: string | undefined, label: string): T {
  if (!text) {
    throw new Error(`AIの応答が空でした (${label})`);
  }
  return JSON.parse(text) as T;
}

export function getAiConnectionStatus(): { configured: boolean; mode: string; model: string } {
  const info = createGenAIClient();
  if (!info) {
    return {
      configured: false,
      mode: '未設定',
      model: process.env.GEMINI_MODEL || 'gemini-3.8-flash',
    };
  }
  return {
    configured: true,
    mode: info.mode,
    model: info.model,
  };
}

function extractBase64Data(dataUrl?: string): string | null {
  if (!dataUrl) return null;
  const prefix = 'base64,';
  const idx = dataUrl.indexOf(prefix);
  if (idx !== -1) {
    return dataUrl.slice(idx + prefix.length);
  }
  return dataUrl;
}

// 企画書 2章「関数呼び出しによる判断の記録 (Own the Why)」用 Function Declaration
const RECORD_HUMAN_DECISION_DECLARATION: FunctionDeclaration = {
  name: 'record_human_decision',
  description:
    '学習者が決めた顧客層・画面の改造指示・レビュー指摘の採否と、その理由を構造化して記録する。',
  parameters: {
    type: Type.OBJECT,
    properties: {
      stage: {
        type: Type.STRING,
        enum: ['select_cohort', 'apply_change', 'review_disposition'],
        description: '判断を行ったフェーズ',
      },
      targetId: {
        type: Type.STRING,
        description: '選択対象のID（顧客層ID・改造した部品ID・レビュアーID）',
      },
      chosenLabel: {
        type: Type.STRING,
        description: '人間が選んだ方針の表示名',
      },
      reason: {
        type: Type.STRING,
        description: 'なぜその選択をしたかの人間自身の理由',
      },
    },
    required: ['stage', 'targetId', 'chosenLabel', 'reason'],
  },
};

/**
 * 学習者の判断（顧客層選択・トレードオフ選択・レビュー採否）を Gemini Function Calling で構造化記録する。
 */
export async function recordHumanDecisionWithFunctionCalling(
  req: RecordDecisionRequest,
): Promise<RecordDecisionResponse> {
  const aiInfo = createGenAIClient();
  let recordedVia: 'gemini_function_call' | 'local_function_call' = 'local_function_call';
  // 理由は任意。空のまま記録し、AIが勝手に理由を補わないようにする
  const originalReason = (req.reason ?? '').trim();

  // 手直し（applyOptionToSpec）のたびに理由なしで並列呼び出しすると、利用枠（429）の消費が2倍になり、
  // しかも返り値の引数は使われないため、客層選択（/api/decision）や理由があるときだけ Gemini を呼ぶ
  const shouldCallGeminiFunction = req.stage !== 'apply_change' || Boolean(originalReason);
  if (aiInfo && shouldCallGeminiFunction) {
    try {
      const response = await generateContentWithTimeout(aiInfo, {
        model: aiInfo.model,
        contents: [
          '学習者がアプリ改造スタジオで以下の判断を行いました。関数 record_human_decision を呼び出して判断と理由をそのまま記録してください。人間の判断を上書きしないでください。',
          `フェーズ: ${req.stage}`,
          `対象ID: ${req.targetId}`,
          `選んだ内容: ${req.chosenLabel}`,
          `学習者の理由: ${originalReason || 'なし'}`,
        ].join('\n'),
        config: {
          tools: [{ functionDeclarations: [RECORD_HUMAN_DECISION_DECLARATION] }],
          temperature: 0,
          ...thinkingConfig(),
        },
      }, TIMEOUTS.side);

      const fnCall = response.functionCalls?.[0];
      if (fnCall && fnCall.name === 'record_human_decision') {
        recordedVia = 'gemini_function_call';
      }
    } catch (err) {
      // 判断の記録はAIの成否に左右されないよう、AI呼び出しに失敗しても人間の入力どおりに保存する
      console.error('[gemini:functionCall] AI経由の記録に失敗したため入力どおりに保存します:', describeErrorForLog(err));
    }
  }

  const decision: HumanDecisionRecord = {
    id: `dec-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    stage: req.stage,
    targetId: req.targetId,
    chosenLabel: req.chosenLabel,
    // 自由に書かれた理由は、学習者を特定できる言葉が混ざりうるので保存しない。保存するのは決まった値だけ
    reason: '',
    recordedVia,
    createdAt: new Date().toISOString(),
  };

  const { persistedToFirestore } = await saveHumanDecision(decision);
  return {
    decision,
    persistedToFirestore,
  };
}

const EVALUATE_COMMENTS_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    comments: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          reviewerId: { type: Type.STRING },
          outcome: {
            type: Type.STRING,
            enum: ['achieved', 'bothered', 'abandoned', 'regretted'],
            description:
              '画面画像と仕様・操作ログを踏まえたこの顧客の体験結果（achieved:満足して買えた / bothered:買えたが途中で嫌な思いや手間があった / abandoned:途中でやめた / regretted:買えたが余計な契約等で後悔した）',
          },
          frictionPoint: {
            type: Type.STRING,
            description: 'どこで迷ったか・何が良かったかの短い要点（18文字以内）',
          },
          comment: {
            type: Type.STRING,
            description: 'ペルソナ本人になりきった一言感想（30文字以内、直し方や改善指示は絶対に言わない）',
          },
        },
        required: ['reviewerId', 'frictionPoint', 'comment'],
      },
    },
  },
  required: ['comments'],
};

const CONDITION_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    stateKey: { type: Type.STRING },
    operator: { type: Type.STRING, enum: ['eq', 'neq', 'gt', 'lt'] },
    value: { type: Type.STRING },
  },
  required: ['stateKey', 'operator', 'value'],
};

const COMPONENT_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    id: { type: Type.STRING },
    type: {
      type: Type.STRING,
      enum: [
        'heading',
        'text',
        'product_hero',
        'product_card',
        'progress_steps',
        'choice_cards',
        'summary',
        'input',
        'select',
        'checkbox',
        'toggle',
        'slider',
        'button',
        'status_banner',
        'popup',
        'custom',
      ],
    },
    darkPattern: {
      type: Type.STRING,
      enum: ['false_urgency'],
      description: 'お客さんを急かす仕掛け（偽の残り時間など）。急かさない表示に変えたら外す',
    },
    role: {
      type: Type.STRING,
      enum: [
        'primary_action',
        'decoy',
        'danger',
        'required_input',
        'distraction',
        'trap',
        'total',
        'popup',
        'info',
      ],
    },
    label: { type: Type.STRING },
    icon: { type: Type.STRING },
    badge: { type: Type.STRING },
    caption: { type: Type.STRING },
    priceLabel: { type: Type.STRING },
    optionDetails: { type: Type.ARRAY, items: { type: Type.STRING } },
    rows: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          label: { type: Type.STRING },
          value: { type: Type.STRING },
        },
        required: ['label', 'value'],
      },
    },
    placeholder: { type: Type.STRING },
    options: { type: Type.ARRAY, items: { type: Type.STRING } },
    stateKey: { type: Type.STRING },
    style: {
      type: Type.OBJECT,
      properties: {
        size: { type: Type.STRING, enum: ['xs', 'sm', 'md', 'lg', 'xl'] },
        variant: {
          type: Type.STRING,
          enum: ['default', 'primary', 'danger', 'warning', 'success', 'muted'],
        },
        align: { type: Type.STRING, enum: ['left', 'center', 'right'] },
        fullWidth: { type: Type.BOOLEAN },
        noisyAnimation: { type: Type.BOOLEAN },
        rotateDeg: { type: Type.NUMBER },
        numberFormat: { type: Type.STRING, enum: ['arabic', 'kanji'] },
        sticky: { type: Type.BOOLEAN },
        skin: { type: Type.STRING, enum: ['native', 'intrusive'] },
      },
      required: ['size', 'variant'],
    },
    behavior: {
      type: Type.OBJECT,
      properties: {
        effect: {
          type: Type.STRING,
          enum: ['none', 'escape_on_hover', 'require_3_clicks', 'confirm_dialog'],
        },
        confirmText: { type: Type.STRING },
      },
      required: ['effect'],
    },
    visibleWhen: CONDITION_SCHEMA,
    trapWhen: CONDITION_SCHEMA,
    trapSafeValue: { type: Type.STRING },
    popupDelayMs: {
      type: Type.OBJECT,
      properties: {
        min: { type: Type.NUMBER },
        max: { type: Type.NUMBER },
      },
      required: ['min', 'max'],
    },
    closePosition: {
      type: Type.STRING,
      enum: ['top-right', 'top-left', 'bottom-right', 'bottom-left', 'bottom-bar'],
      description: 'popup の閉じるボタンの位置。押しやすくする指示などで変更可能',
    },
    closeSize: {
      type: Type.STRING,
      enum: ['xs', 'sm', 'md', 'lg', 'xl'],
      description: 'popup の閉じるボタンの大きさ',
    },
    closeLabel: {
      type: Type.STRING,
      description: 'popup の閉じるボタンの文字（例: 閉じる、あとで見る、×）',
    },
    customHtml: {
      type: Type.STRING,
      description:
        'サンドボックス内で自由に描画するHTML。{状態キー}で値表示、data-set-state="key:value" / data-toggle-state="key" / data-increment-state="key" / data-complete-task="message" / data-close-popup でクリック動作と連動',
    },
    customCss: {
      type: Type.STRING,
      description: 'この部品に適用する追加のCSSプロパティ（例: padding: 12px; border-radius: 12px;）',
    },
    actions: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          type: {
            type: Type.STRING,
            enum: [
              'set_state',
              'toggle_state',
              'increment_state',
              'reset_all',
              'show_toast',
              'open_page',
              'complete_task',
            ],
          },
          targetKey: { type: Type.STRING },
          value: { type: Type.STRING },
          message: { type: Type.STRING },
        },
        required: ['type'],
      },
    },
  },
  required: ['id', 'type', 'role', 'label', 'style', 'actions'],
};

const PATCH_ITEM_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    op: {
      type: Type.STRING,
      enum: [
        'remove_component',
        'replace_component',
        'insert_component',
        'move_component',
        'upsert_state',
        'upsert_derived',
      ],
    },
    id: {
      type: Type.STRING,
      description: 'remove_component または move_component の対象部品ID',
    },
    afterId: {
      type: Type.STRING,
      description: 'insert_component または move_component で直前に置く部品ID。先頭に置くときは空文字',
    },
    component: COMPONENT_SCHEMA,
    variable: {
      type: Type.OBJECT,
      properties: {
        key: { type: Type.STRING },
        label: { type: Type.STRING },
        type: { type: Type.STRING, enum: ['string', 'number', 'boolean'] },
        initialValue: { type: Type.STRING },
      },
      required: ['key', 'label', 'type', 'initialValue'],
    },
    key: {
      type: Type.STRING,
      description: 'upsert_derived の計算値キー',
    },
    expression: {
      type: Type.STRING,
      description: 'upsert_derived の計算式',
    },
  },
  required: ['op'],
};

export const APPLY_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    commitMessage: {
      type: Type.STRING,
      description: '何を変えたかを「〜しました」で終わる1文。30文字以内、括弧なし',
    },
    patches: {
      type: Type.ARRAY,
      description: '変更する部品や状態に対する最小限の差分操作リスト。触らない部品は一切含めない',
      items: PATCH_ITEM_SCHEMA,
    },
  },
  required: ['commitMessage', 'patches'],
};

export function normalizeUpdatedSpec(rawSpec: AppSpec, previousSpec: AppSpec): AppSpec {
  const normalizedState = (rawSpec.state || previousSpec.state).map((s) => {
    let initialValue: string | number | boolean = s.initialValue;
    if (s.type === 'boolean') {
      initialValue = s.initialValue === true || s.initialValue === 'true';
    } else if (s.type === 'number') {
      const n = Number(s.initialValue);
      initialValue = Number.isFinite(n) ? n : 0;
    } else {
      initialValue = String(s.initialValue ?? '');
    }
    return { ...s, initialValue };
  });

  // 計算値の型は分からないため、数値として読めれば数値、true/false なら真偽値として扱う
  const stateTypeMap = new Map(normalizedState.map((s) => [s.key, s.type]));

  const coerceByStateKey = (key: string | undefined, val: unknown): string | number | boolean => {
    const t = key ? stateTypeMap.get(key) : undefined;
    if (t === 'boolean' || val === 'true' || val === 'false' || typeof val === 'boolean') {
      return val === true || val === 'true';
    }
    if (t === 'number' || (t === undefined && val !== '' && Number.isFinite(Number(val)))) {
      const n = Number(val);
      return Number.isFinite(n) ? n : 0;
    }
    return String(val ?? '');
  };

  const coerceCondition = (cond: AppComponent['visibleWhen']) =>
    cond ? { ...cond, value: coerceByStateKey(cond.stateKey, cond.value) } : undefined;

  const normalizedComponents: AppComponent[] = (rawSpec.components || []).map((c) => {
    const visibleWhen = coerceCondition(c.visibleWhen);
    const trapWhen = coerceCondition(c.trapWhen);
    const actions = (c.actions || []).map((a) => ({
      ...a,
      ...(a.value !== undefined ? { value: coerceByStateKey(a.targetKey, a.value) } : {}),
    }));

    // おとりを押せるように直したのに role が decoy のままだと、お客さんの判定が狂うので本物のボタンとして扱う
    const worksNow = actions.some(
      (a) => a.type === 'complete_task' || (a.type === 'set_state' && a.targetKey === 'step'),
    );
    // ポップアップをバナーやボタン等の通常部品に変えた場合は role も info として扱う
    const convertedFromPopup = c.role === 'popup' && c.type !== 'popup';
    const role =
      c.role === 'decoy' && worksNow
        ? 'primary_action'
        : convertedFromPopup
          ? 'info'
          : (c.role ?? 'info');
    // AIが部品を書き換えるときに label や style を省くことがある。画面の描画が落ちないよう、同じ id の元の部品から補う
    const prev = previousSpec.components.find((p) => p.id === c.id);
    const normalized: AppComponent = {
      ...c,
      label: typeof c.label === 'string' ? c.label : (prev?.label ?? ''),
      style: c.style ?? prev?.style ?? { size: 'md', variant: 'default' },
      role,
      actions,
    };
    if (visibleWhen) normalized.visibleWhen = visibleWhen;
    else delete normalized.visibleWhen;
    if (trapWhen) normalized.trapWhen = trapWhen;
    else delete normalized.trapWhen;
    if (c.trapSafeValue !== undefined) {
      normalized.trapSafeValue = coerceByStateKey(c.stateKey, c.trapSafeValue);
    }
    return normalized;
  });

  return {
    ...rawSpec,
    state: normalizedState,
    derived: rawSpec.derived ?? previousSpec.derived,
    components: normalizedComponents,
  };
}

/**
 * Gemini が返した差分パッチを現在の仕様に適用し、正規化・整合性検証・店長のリアクション生成を行う。
 */
export function applyPatchesToSpecWithValidation(
  currentSpec: AppSpec,
  commitMessage: string,
  rawPatches: SpecPatch[],
): Omit<ApplyResponse, 'decisionRecord'> {
  if (!Array.isArray(rawPatches) || rawPatches.length === 0) {
    return {
      commitMessage: 'どこをどう変えるか、もう少し詳しく教えてください',
      updatedSpec: structuredClone(currentSpec),
      patches: [],
      rolledBack: true,
      validation: {
        valid: false,
        canRenderHtml: true,
        hasTaskCompletionEvent: false,
        errors: ['変更差分が空でした'],
      },
    };
  }

  const { spec: patchedSpec, touchedComponentIds } = applySpecPatches(currentSpec, rawPatches);
  const updatedSpec = normalizeUpdatedSpec(patchedSpec, currentSpec);

  const prevHadCompletion = currentSpec.components.some(
    (c) => c.type === 'button' && hasCompletionAction(c),
  );
  const touchedPrimaryButton = currentSpec.components.some(
    (c) => c.role === 'primary_action' && touchedComponentIds.has(c.id),
  );

  const validation = validateSandboxSpec(updatedSpec, {
    requireTaskCompletionEvent: prevHadCompletion || touchedPrimaryButton,
    previousSpec: currentSpec,
    touchedComponentIds,
  });

  if (!validation.valid) {
    console.warn('[gemini:apply] 検証失敗のためロールバックします:', `${validation.errors.length}件`);
    return {
      commitMessage: 'うまく動かなかったので元に戻しました',
      updatedSpec: structuredClone(currentSpec),
      patches: rawPatches,
      validation,
      rolledBack: true,
    };
  }

  const ownerReaction = buildOwnerChangeReaction(ACTIVE_THEME.owner, currentSpec, updatedSpec);
  const ownerFacts = describeOwnerChange(ACTIVE_THEME.owner, currentSpec, updatedSpec).map(
    (f) => f.fact,
  );

  return {
    commitMessage,
    updatedSpec,
    patches: rawPatches,
    workSteps: describeWorkSteps(currentSpec, updatedSpec, rawPatches),
    ownerReaction,
    ownerFacts,
    validation,
    rolledBack: false,
  };
}

/**
 * 実装AI:
 * 人間が指示した変更内容だけで、最小限の差分パッチ（patches）を生成してアプリを更新する。
 * 触っていない部品は一切変更せず、検証失敗時は直前の版へ自動ロールバックする。
 */
export async function applyOptionToSpec(req: ApplyRequest): Promise<ApplyResponse> {
  const aiInfo = requireGenAIClient();
  const reason = req.selectionReason?.trim() ?? '';
  const targetComponents = req.currentSpec.components.filter((c) =>
    req.targetComponentIds.includes(c.id),
  );
  const chosenLabel = req.userInstruction;
  const targetId =
    req.targetComponentIds.length > 0 ? req.targetComponentIds.join(',') : 'chat_instruction';

  const decisionPromise = recordHumanDecisionWithFunctionCalling({
    stage: 'apply_change',
    targetId,
    chosenLabel,
    reason,
    cohortId: req.cohortId,
  }).then((r) => r.decision);

  const defaultScenarioSpec = buildScenarioSpec(
    ACTIVE_THEME,
    ACTIVE_THEME.defaultScenarioId,
  ).spec;
  const currentIdSet = new Set(req.currentSpec.components.map((c) => c.id));
  const removedInitialComponents = defaultScenarioSpec.components.filter(
    (c) => !currentIdSet.has(c.id),
  );

  const prompt = [
    'あなたは開発スタジオの実装AI（お助けAI）です。人間が指示した変更に従って、サンドボックス内のアプリを更新する最小限の差分操作（patches）を返してください。',
    '【差分操作（patches）の絶対ルール】',
    '0. 人間が頼んだ部品と、その変更のために必要な部品・状態・計算式を変えてください。頼まれていない部品を勝手に消したり書き換えたりしないでください。',
    '0-2. 指示が対象と目的だけで直し方を言っていないとき（例：定期便を望む人だけが自分で選べるようにしたい）は、その目的をかなえる具体的な変更をあなたが選んでください。変えるのは指示された対象と、目的のために必要な部品だけです。対象の機能は、目的に反しない限り消さずに残してください（定期便やクーポンを望むお客さんもいるため）。',
    '1. 変更しない部品は patches に一切含めないでください。触らない既存の product_card, choice_cards, summary 等の型は維持しつつ、人間から改造を頼まれた部品（例：広告の閉じるボタンの位置・大きさ・文字を closePosition / closeSize / closeLabel で変える、広告を status_banner や button に変える、新しい＋α機能を type: "custom" や customHtml / customCss で追加するなど）は拒絶せず指示どおり自由に実装してください。',
    '2. 使える操作 op は次の6つです：',
    '   - remove_component: 部品を消す（id を指定）',
    '   - replace_component: 既存の部品1つを同じ id のまま書き換える（component に更新後の部品全体を指定。badge, caption, priceLabel, optionDetails, rows, style.sticky 等の元の設定を勝手に消さないこと）',
    '   - insert_component: 新しい部品を追加する（afterId に直前の部品ID、component に新しい部品を指定）',
    '   - move_component: 部品の位置を動かす（id と afterId を指定）',
    '   - upsert_state: 状態変数を追加または初期値を変更する（variable を指定。チェックボックスなどの最初の状態を変えるときは、その部品の stateKey の initialValue を変える）',
    '   - upsert_derived: 計算式を追加または変更する（key と expression を指定）',
    '3. 画面の見た目だけでなく、裏側の処理（actions, behavior, state, visibleWhen）も整合するように更新してください。',
    '4. 押しても何も起きない本物のボタン（role が primary_action で actions が空）を動くように直すときは、replace_component でその部品の actions に complete_task を設定してください。style.sticky など元の見た目の設定はそのまま残してください。',
    ...ACTIVE_THEME.implementerNotes.map((note, i) => `${i + 5}. ${note}`),
    `${ACTIVE_THEME.implementerNotes.length + 5}. commitMessage は何を変えたかを「〜しました」で終わる1文、30文字以内で書いてください。括弧・記号・英語は使わないでください。`,
    '',
    `【現在のアプリ仕様】\n${JSON.stringify(req.currentSpec)}`,
    removedInitialComponents.length > 0
      ? `【初期画面から削除された部品（人間から特定の部品を戻す・復活させるよう指示された場合は、この id・stateKey・仕様を使って insert_component で復元してください）】\n${JSON.stringify(removedInitialComponents)}`
      : '',
    `【赤ペンで囲まれた話題の部品】\n${
      targetComponents.length > 0 ? JSON.stringify(targetComponents) : '特定の部品は指していません'
    }`,
    `【直近の会話】\n${formatHistory(req.history)}`,
    `【人間の手直し指示】\n${req.userInstruction}`,
    reason ? `【人間がこの手直しを選んだ理由】\n${reason}` : '',
  ]
    .filter(Boolean)
    .join('\n');

  const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [
    { text: prompt },
  ];
  const b64 = extractBase64Data(req.annotatedImageBase64);
  if (b64) {
    parts.push({
      inlineData: {
        mimeType: 'image/png',
        data: b64,
      },
    });
  }

  const specUpdatePromise = generateContentWithTimeout(aiInfo, {
      model: aiInfo.model,
      contents: parts,
      config: {
        responseMimeType: 'application/json',
        responseSchema: APPLY_SCHEMA,
        temperature: 0.2,
        ...thinkingConfig(),
      },
    }, TIMEOUTS.apply)
    .then((response) => {
      const parsed = parseJsonResponse<{ commitMessage: string; patches: SpecPatch[] }>(
        response.text,
        'apply',
      );
      return applyPatchesToSpecWithValidation(
        req.currentSpec,
        cleanJapaneseText(parsed.commitMessage, '画面を変えました'),
        parsed.patches,
      );
    });

  // 判断の記録は裏で続けてよい。画面の改造が終わったら、記録を少しだけ待って返事をする
  const safeDecision = decisionPromise.catch(() => undefined);
  const candidate = await specUpdatePromise;
  const changeStats = candidate.rolledBack
    ? { changedComponentIds: [], removedNeededFeature: false }
    : inspectAppliedChangeForTelemetry(req.currentSpec, candidate.updatedSpec, candidate.patches);
  const [decisionRecord, telemetryEvent] = await Promise.all([
    Promise.race([
      safeDecision,
      new Promise<undefined>((resolve) => setTimeout(() => resolve(undefined), 1500)),
    ]),
    recordPromptTelemetryEvent({
      cohortId: req.cohortId,
      runId: req.runId,
      turnIndex: req.turnIndex,
      shareForAggregate: req.shareForAggregate,
      promptText: req.userInstruction,
      sessionMode: 'execute',
      recipient: 'assistant',
      hasPenAnnotation: req.targetComponentIds.length > 0 || Boolean(req.annotatedImageBase64),
      changedComponentIds: changeStats.changedComponentIds,
      removedNeededFeature: changeStats.removedNeededFeature,
    }),
  ]);

  return {
    ...candidate,
    decisionRecord,
    telemetryEvent,
  };
}

const CLEAR_FEEDBACK_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    overview: {
      type: Type.STRING,
      description:
        '1つ目の吹き出し：今回の挑戦の概要。選んだ客層が全員買えたこと、質問や手直しの回数、お店の定期便やほかの客層の結果をねぎらいとともにまとめる。100文字以内、括弧なし',
    },
    goodPoints: {
      type: Type.STRING,
      description:
        '2つ目の吹き出し：今回の挑戦でどこがよかったか。よかった点の事実にもとづき、画面の両立や指示の出し方で良かったところを具体的に褒める。事実にある `{{turn:数字}}` の目印は1文字も変えずにそのまま文中に残す。110文字以内、括弧なし',
    },
    nextImprovements: {
      type: Type.STRING,
      description:
        '3つ目の吹き出し：もっと何をしたら良くなるか。改善点の事実にもとづき、次に試すとさらに良くなる画面の工夫や指示の出し方を前向きに提案する。110文字以内、括弧なし',
    },
  },
  required: ['overview', 'goodPoints', 'nextImprovements'],
};

/**
 * 吹き出し文から丸括弧やかぎ括弧を取り除き、裸の {{turn:N}} 目印があればバッククォートで囲む
 */
function sanitizeClearFeedbackLine(raw: string, fallback: string): string {
  const cleaned = cleanJapaneseText(raw.trim(), '')
    .replace(/[()（）「」『』]/g, '')
    .replace(/(?<!`)\{\{turn:(\d+)\}\}(?!`)/g, '`{{turn:$1}}`')
    .replace(/\s+/g, ' ')
    .trim();
  return cleaned || fallback;
}

/**
 * クリア時のお助けAIの3つの振り返り吹き出し（概要・どこがよかったか・もっと何をしたら良くなるか）をAIに作らせる。
 * 事実判定はコードで行った ClearFeedbackFacts だけを渡し、通信失敗や目印欠落時は fallbackLines を使う
 */
export async function voiceAssistantClearReport(
  facts: ClearFeedbackFacts,
): Promise<[string, string, string]> {
  const aiInfo = createGenAIClient();
  if (!aiInfo) return facts.fallbackLines;

  const prompt = [
    'あなたは注文画面の手直しを一緒に進めてきた進行役のお助けAIです。丁寧で親しみやすいです・ます調で話します。',
    '学習者が選んだ客層の3人全員を満足させてクリアしました。コードで判定した事実だけをもとに、お助けAIからの振り返りを3つの吹き出しで作ってください。',
    '',
    '【1つ目の吹き出し overview に含める事実】',
    ...facts.overviewFacts.map((f) => `- ${f}`),
    '',
    '【2つ目の吹き出し goodPoints に含める事実】',
    ...facts.goodPointFacts.map((f) => `- ${f}`),
    '',
    '【3つ目の吹き出し nextImprovements に含める事実】',
    ...facts.improvementFacts.map((f) => `- ${f}`),
    '',
    '【絶対ルール】',
    '1. 上の事実に書かれていないことは絶対に作らないでください。回数や人数の数字も事実のとおりにしてください。',
    '2. 事実の中に `{{turn:数字}}` という目印がある場合は、ブラウザ側で学習者の実際の発言に置き換える記号なので、1文字も変えずにバッククォート付きの `{{turn:数字}}` のまま goodPoints の文中に必ず残してください。',
    '3. 丸括弧やかぎ括弧は一切使わないでください。強調したい語や数字はバッククォート `...` で囲んでよいです。',
    '4. それぞれの吹き出しは80〜110文字程度の自然な話し言葉1〜2文にまとめてください。',
  ].join('\n');

  try {
    const response = await generateContentWithTimeout(
      aiInfo,
      {
        model: aiInfo.model,
        contents: prompt,
        config: {
          responseMimeType: 'application/json',
          responseSchema: CLEAR_FEEDBACK_SCHEMA,
          temperature: 0.4,
          ...thinkingConfig(),
        },
      },
      TIMEOUTS.side,
    );
    const parsed = parseJsonResponse<{
      overview: string;
      goodPoints: string;
      nextImprovements: string;
    }>(response.text, 'clear_feedback');

    const overview = sanitizeClearFeedbackLine(parsed.overview ?? '', facts.fallbackLines[0]);
    let goodPoints = sanitizeClearFeedbackLine(parsed.goodPoints ?? '', facts.fallbackLines[1]);
    const nextImprovements = sanitizeClearFeedbackLine(
      parsed.nextImprovements ?? '',
      facts.fallbackLines[2],
    );

    // 元のフォールバック文に含まれていた {{turn:N}} 目印が1つでも欠けていたらフォールバック文を使う
    const expectedRefs = [...facts.fallbackLines[1].matchAll(/\{\{turn:\d+\}\}/g)].map((m) => m[0]);
    if (expectedRefs.some((ref) => !goodPoints.includes(ref))) {
      goodPoints = facts.fallbackLines[1];
    }

    return [overview, goodPoints, nextImprovements];
  } catch (err) {
    console.warn(
      '[gemini:clear_feedback] クリア振り返りの生成に失敗したため決まった文面を使います:',
      describeErrorForLog(err),
    );
    return facts.fallbackLines;
  }
}

function detectCouponStatusForClearFeedback(
  spec: AppSpec,
  bounds?: EvaluateRequest['componentBounds'],
  existingSimMap?: ReadonlyMap<string, ReturnType<typeof simulatePersonaComputerUse>>,
): NonNullable<ClearFeedbackInput['couponStatus']> {
  const hadCouponInShop = spec.state.some(
    (s) => s.key === 'couponPopup' || s.key.toLowerCase().includes('coupon'),
  );
  if (!hadCouponInShop) return 'none';
  const couponSeeker = ALL_REVIEWER_PERSONAS.find((p) => p.constraints.wantsCoupon);
  if (!couponSeeker) return 'none';
  const sim =
    existingSimMap?.get(couponSeeker.id) ??
    simulatePersonaComputerUse(spec, couponSeeker, bounds);
  if (sim.actionTrace.some((t) => t.includes('クーポンを使いたかったが見当たらなかった'))) {
    return 'removed';
  }
  if (
    sim.actionTrace.some((t) => t.includes('の特典を適用した') || t.includes('の特典を適用して閉じた')) &&
    !sim.actionTrace.some((t) => t.includes('合計金額が変わっていなかった'))
  ) {
    return 'usable_discount';
  }
  return 'kept_no_discount';
}

/**
 * レビューAI:
 * 改修意図も議論も一切知らず、画面と Computer Use 実操作シミュレーション結果（制約変数適用済み）のみを受け取る。
 * 達成/離脱はアプリのタスク完了イベントでコード判定し、Geminiは各ペルソナの感想の言葉だけを担う（直し方は言わない）。
 */
export async function evaluateAppWithPersonas(req: EvaluateRequest): Promise<EvaluationResponse> {
  const aiInfo = requireGenAIClient();
  const reviewers = getReviewersByCohort(req.cohortId);

  // 1. まずコード側で、各ペルソナの制約変数（視力ぼかし・指のズレ・手数上限・待てる時間）をかけた
  //    Computer Use 実操作シミュレーションを実行し、タスク完了イベント発火の有無で達成/離脱を確定する。
  const simMap = new Map(
    reviewers.map((persona) => [
      persona.id,
      simulatePersonaComputerUse(req.currentSpec, persona, req.componentBounds),
    ]),
  );

  // お店の側の結果（注文数や、題材で決めた数字）はコードで数える
  const owner = buildOwnerReport(
    ACTIVE_THEME.owner,
    req.currentSpec,
    reviewers.map((p) => ({
      outcome: simMap.get(p.id)!.outcome,
      finalState: simMap.get(p.id)!.finalState,
    })),
  );
  const buildFeedbackForSummary = async (summary: Awaited<ReturnType<typeof finalizeTelemetryOnEvaluate>>) => {
    const cohortLabel =
      TARGET_COHORTS.find((c) => c.id === req.cohortId)?.label ?? '選んだ客層';
    const otherPersonas = ALL_REVIEWER_PERSONAS.filter(
      (p) => !reviewers.some((r) => r.id === p.id),
    );
    const otherImpact = describeOtherCohortImpact(
      req.currentSpec,
      otherPersonas,
      req.componentBounds,
    );
    const couponStatus = detectCouponStatusForClearFeedback(
      req.currentSpec,
      req.componentBounds,
      simMap,
    );
    const facts = buildClearFeedbackFacts({
      cohortLabel,
      summary,
      owner,
      otherCohorts: {
        achieved: otherImpact.achieved,
        total: otherImpact.total,
        unhappy: otherImpact.unhappy.map((u) => ({ name: u.name, reason: u.reason })),
      },
      couponStatus,
    });
    return voiceAssistantClearReport(facts);
  };

  const finalizeWithClearFeedback = () =>
    finalizeTelemetryOnEvaluate(req.cohortId, reviewers.length, reviewers.length, req.runId).then(
      async (summary) => ({
        ...summary,
        assistantFeedbackLines: await buildFeedbackForSummary(summary),
      }),
    );

  const simulationSummary = reviewers.map((p) => {
    const sim = simMap.get(p.id)!;
    return {
      reviewerId: p.id,
      name: p.name,
      role: p.role,
      trait: p.traitLabel,
      goal: p.goal,
      appliedConstraints: sim.appliedConstraintNote,
      actionTrace: sim.actionTrace,
      codeDeterminedOutcome: sim.outcome,
      codeDeterminedFriction: sim.frictionPoint,
    };
  });

  const hasCustomComponents = req.currentSpec.components.some(
    (c) => c.type === 'custom' || Boolean(c.customHtml),
  );

  const prompt = [
    `あなたは${ACTIVE_THEME.appName}で${ACTIVE_THEME.taskLabel}ことを試した3人の顧客です。改修の意図やこれまでの議論は一切知りません。`,
    '添付された画面のスクリーンショット（視覚情報）・現在のUI部品仕様・各顧客が実際に画面を操作したログ（actionTrace）と基準結果（codeDeterminedOutcome / codeDeterminedFriction）に基づき、本人になりきった感想（comment）、困った点・良かった点（frictionPoint）、および体験結果（outcome）を生成してください。',
    '結果は achieved が達成、abandoned が途中でやめた、regretted が完了はできたが余計な契約などにあとで気づいて後悔した、bothered が買えたが途中で嫌な思いをした、です。',
    'outcome は原則として codeDeterminedOutcome をそのまま使い、comment と frictionPoint も必ずその人の actionTrace と codeDeterminedFriction にある事実だけを本人の口調で言い換えてください（独自の追加部品 customHtml が画面にある場合のみ、その部品による満足・不満を反映してよいです）。',
    'actionTrace に書かれていないこと（例：定期便を申し込んでいない人が定期便を申し込んだと言う、記録にない文字の重なりや画面崩れを指摘する、電話解約でやめたのに満足したことにする等）は絶対に作らないでください。',
    'regretted の人は、あとから気づいたがっかりした気持ちを話してください。bothered の人は、買えたことと途中の嫌だったことの両方が伝わるように話してください。achieved の人は、素直に満足や工夫への好感を話してください。',
    '',
    '【絶対ルール：短く・直し方は絶対に言わない・括弧は使わない】',
    '1. comment は30文字以内の、独り言のような自然な話し言葉にしてください。',
    '2. frictionPoint は18文字以内で要点だけ書いてください。',
    '3. こう直してほしい・ボタンを大きくして等の改善指示は絶対に言わず、自分が体験した感想だけを述べてください。',
    '',
    `【現在の画面の部品仕様】\n${JSON.stringify(req.currentSpec.components)}`,
    `【操作ログと基準判定結果】\n${JSON.stringify(simulationSummary)}`,
  ].join('\n');

  const evalParts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [
    { text: prompt },
  ];
  const screenshotB64 = extractBase64Data(req.screenshotBase64);
  if (screenshotB64) {
    evalParts.push({
      inlineData: {
        mimeType: 'image/png',
        data: screenshotB64,
      },
    });
  }

  let parsed: {
    comments: Array<{
      reviewerId: string;
      outcome?: ReviewOutcome;
      frictionPoint: string;
      comment: string;
    }>;
  } = {
    comments: [],
  };
  try {
    const response = await generateContentWithTimeout(
      aiInfo,
      {
        model: aiInfo.model,
        contents: evalParts,
        config: {
          responseMimeType: 'application/json',
          responseSchema: EVALUATE_COMMENTS_SCHEMA,
          temperature: 0.4,
          ...thinkingConfig(),
        },
      },
      TIMEOUTS.review,
    );
    parsed = parseJsonResponse<typeof parsed>(response.text, 'evaluate');
  } catch (err) {
    if (err instanceof AiNotConfiguredError) throw err;
    console.warn('[gemini:evaluate] 感想の生成に失敗したため、シミュレーターの言葉を使います:', describeErrorForLog(err));
  }

  const results: ReviewerEvaluation[] = reviewers.map((p) => {
    const sim = simMap.get(p.id)!;
    const aiComment = parsed.comments.find((c) => c.reviewerId === p.id);
    // 勝敗はコードのシミュレーター判定（sim.outcome）を厳守する。
    // 独自の＋α部品（customHtml）が追加されている場合のみ、注文不能・後悔・希望機能欠落でない限りAI判定も反映する
    const missingDesiredFeature =
      sim.frictionPoint.includes('定期便の設定がなかった') ||
      Object.values(COUPON_FRICTIONS).some((f) => sim.frictionPoint.includes(f));
    const resolvedOutcome: ReviewOutcome =
      hasCustomComponents &&
      sim.outcome !== 'abandoned' &&
      sim.outcome !== 'regretted' &&
      !missingDesiredFeature &&
      (aiComment?.outcome === 'achieved' || aiComment?.outcome === 'bothered')
        ? aiComment.outcome
        : sim.outcome;
    return {
      reviewerId: p.id,
      outcome: resolvedOutcome,
      frictionPoint:
        cleanJapaneseText(aiComment?.frictionPoint || '', sim.frictionPoint) || sim.frictionPoint,
      comment:
        cleanJapaneseText(aiComment?.comment || '', sim.simulatedComment) || sim.simulatedComment,
      actionTrace: sim.actionTrace,
      appliedConstraintNote: sim.appliedConstraintNote,
    };
  });

  const achievedCount = results.filter((r) => r.outcome === 'achieved').length;
  // 最終評価で全員が達成した場合だけ、店長の締めのせりふとクリア振り返りを作る
  const allAchieved = achievedCount === results.length;
  const closingPromise = allAchieved
    ? voiceOwnerLine({
        situation: 'closing',
        facts: [
          `選んだお客さん${owner.customers}人全員が、満足して買ってくれた`,
          ...owner.metrics.map((m) =>
            m.missing
              ? `${m.label}の申し込み欄は画面からなくなった`
              : `${m.label}の申し込みは${m.count}件`,
          ),
        ],
        fallback: buildOwnerClosingLines(ACTIVE_THEME.owner, owner).join('\n'),
        history: req.history,
      })
    : Promise.resolve(undefined);

  const telemetryPromise =
    achievedCount === results.length
      ? finalizeWithClearFeedback()
      : finalizeTelemetryOnEvaluate(req.cohortId, achievedCount, results.length, req.runId);
  const [closing, telemetrySummary] = await Promise.all([closingPromise, telemetryPromise]);
  if (closing) owner.closingLines = closing.text.split('\n').filter(Boolean);

  return {
    evaluatedAt: new Date().toISOString(),
    cohortId: req.cohortId,
    results,
    owner,
    telemetrySummary,
  };
}

const TRIAGE_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    intent: {
      type: Type.STRING,
      enum: ['question', 'change', 'unclear'],
      description:
        'question は画面や部品・お客さんの反応について聞いている、change は何をどう変えるかが決まる改造の指示か、対象の部品と目的（誰がどうできるようにしたいか）がそろった指示、unclear は感想・不満・あいさつや、対象も目的もない丸投げ、対象の部品がない目的だけの発言',
    },
    reply: {
      type: Type.STRING,
      description:
        'question なら仕様や記録にある事実だけの説明とどう変えたいかの問いかけ、unclear なら受け止めとどう変えるかの聞き返し。change なら空文字。どう直せばよいかの答えや手直し案は言わない。70文字以内、お客さんの操作の記録を伝えるときだけ100文字以内。括弧なし',
    },
    pointedComponentIds: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description:
        '学習者が自分で指した部品のID。赤ペンで囲んだ部品、発言で名前や見た目を挙げた部品、直前に自分で押して画面通知が出た部品だけ。お客さんの発言にだけ出てくる部品は含めない',
    },
  },
  required: ['intent', 'reply', 'pointedComponentIds'],
};

async function applyFromChat(req: ChatRequest): Promise<ChatResponse> {
  const applied = await applyOptionToSpec({
    currentSpec: req.currentSpec,
    targetComponentIds: req.targetComponentIds,
    userInstruction: req.userMessage,
    cohortId: req.cohortId,
    runId: req.runId,
    turnIndex: req.turnIndex,
    shareForAggregate: req.shareForAggregate,
    annotatedImageBase64: req.annotatedImageBase64,
    history: req.history,
  });
  return {
    intent: 'apply',
    reply: applied.commitMessage,
    suggestedFollowUps: [],
    applied,
    telemetryEvent: applied.telemetryEvent,
  };
}

const TRANSCRIBE_SCHEMA: Schema = {
  type: Type.OBJECT,
  properties: {
    hasMeaningfulSpeech: {
      type: Type.BOOLEAN,
      description:
        'アプリの手直し・質問・相談に関係する意味のある発話が含まれていれば true。無音・雑音・咳払い・マイクテスト（あー聞こえる？等）・全く無関係な雑談のみなら false',
    },
    cleanedText: {
      type: Type.STRING,
      description:
        'フィラー（あー、えっと、なんか等）や言い淀み・無関係な独り言を取り除き、赤ペンで囲んだ部品を指す指示語（これ・ここ）があればその部品名を補った、入力欄にそのまま入れる自然な日本語1文。人間が言っていない解決策や判断は絶対に勝手に足さない。hasMeaningfulSpeech が false のときは空文字',
    },
  },
  required: ['hasMeaningfulSpeech', 'cleanedText'],
};

const ALLOWED_AUDIO_MIME_TYPES = new Set([
  'audio/webm',
  'audio/webm;codecs=opus',
  'audio/mp4',
  'audio/ogg',
  'audio/wav',
  'audio/mpeg',
]);

/**
 * マイク音声入力（Base64）と赤ペン囲み画像・画面仕様から、
 * 1) フィラーや無関係な雑音・独り言を取り除き、
 * 2) 赤ペンで囲んだ部品に対する「これ」「ここ」等の指示語を補い、
 * 3) 人間の意図（判断）は一切変えずに、チャット入力欄へ入れるスッキリした1文に整える。
 */
export async function transcribeAndRefineUserAudio(
  req: TranscribeRequest,
): Promise<TranscribeResponse> {
  const aiInfo = requireGenAIClient();
  const rawAudio = extractBase64Data(req.audioBase64) ?? req.audioBase64;
  if (!rawAudio.trim()) {
    return { hasMeaningfulSpeech: false, cleanedText: '' };
  }
  const rawMime = (req.audioMimeType ?? '').trim().toLowerCase();
  const safeAudioMimeType = ALLOWED_AUDIO_MIME_TYPES.has(rawMime) ? rawMime : 'audio/webm';

  const targetIds = req.targetComponentIds ?? [];
  const circledComponents = req.currentSpec.components
    .filter((c) => targetIds.includes(c.id))
    .map((c) => ({
      id: c.id,
      type: c.type,
      role: c.role,
      label: c.label,
      caption: c.caption,
    }));
  const screenComponentSummary = req.currentSpec.components.map((c) => ({
    id: c.id,
    type: c.type,
    role: c.role,
    label: c.label,
  }));

  const prompt = [
    `あなたはアプリ手直しスタジオの音声入力アシスタントです。学習者がマイクで話した音声を聞き取り、チャット入力欄に入れる1文に整えてください。`,
    '【絶対ルール】',
    '1. 人間が言っていない判断や解決策は絶対に勝手に足さないでください（例：「見づらい」としか言っていないのに「文字を大きくして」と書き換えるのは厳禁。その場合は「見づらい」のままにする）。',
    '2. 「あー」「えーと」「なんか」「聞こえてるかな」などのフィラー・言い淀み・言い直し・咳払い・独り言ノイズはすべて取り除いてください。',
    '3. 赤ペンで囲まれた部品があるとき、学習者が「これ」「ここ」「このボタン」など指示語で呼んでいれば、どの部品のことか伝わるように部品の名前や役割を自然に補ってください（例：赤ペンでクーポン広告を囲んで「これ邪魔だから閉じるボタン下に大きくして」と言った場合 → 「クーポン広告の閉じるボタンを下に大きくして」）。',
    '4. 無音・雑音・マイクテスト・アプリと全く無関係な雑談しか含まれていない場合は、hasMeaningfulSpeech を false、cleanedText を空文字にしてください。',
    '5. 括弧や句点は付けず、チャット入力欄にそのまま入る自然な口語1文（60文字以内）にしてください。',
    '',
    `【赤ペンで囲まれている部品】\n${
      circledComponents.length > 0 ? JSON.stringify(circledComponents) : 'なし'
    }`,
    `【現在の画面にある部品一覧】\n${JSON.stringify(screenComponentSummary)}`,
  ].join('\n');

  const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [
    { text: prompt },
    {
      inlineData: {
        mimeType: safeAudioMimeType,
        data: rawAudio,
      },
    },
  ];
  const imgB64 = extractBase64Data(req.annotatedImageBase64);
  if (imgB64) {
    parts.push({
      inlineData: {
        mimeType: 'image/png',
        data: imgB64,
      },
    });
  }

  const res = await generateContentWithTimeout(
    aiInfo,
    {
      model: aiInfo.model,
      contents: parts,
      config: {
        responseMimeType: 'application/json',
        responseSchema: TRANSCRIBE_SCHEMA,
        temperature: 0.1,
        ...thinkingConfig(),
      },
    },
    TIMEOUTS.reply,
  );
  const parsed = parseJsonResponse<TranscribeResponse>(res.text, 'transcribe');
  const cleanedText = cleanJapaneseText((parsed.cleanedText ?? '').replace(/[\r\n]+/g, ' ').trim(), '');
  if (!parsed.hasMeaningfulSpeech || !cleanedText) {
    return { hasMeaningfulSpeech: false, cleanedText: '' };
  }
  return {
    hasMeaningfulSpeech: true,
    cleanedText,
  };
}

/**
 * 自由対話の入口。
 * - 宛先（recipient）はお助けAI（'assistant'）を既定とし、発言内で別の人を明確にメンションしているときはその人に切り替える。
 * - 店長やお客さんが宛先のときは、その本人が返事をする。
 * - お助けAIが宛先のときは、はっきりした改造の指示なら差分パッチ（applyOptionToSpec）を適用し、それ以外は質問・改造・曖昧な発言のどれかを判断する。
 */
export async function chatWithStudio(req: ChatRequest): Promise<ChatResponse> {
  const aiInfo = requireGenAIClient();
  const effectiveReq: ChatRequest = {
    ...req,
    userMessage: req.userMessage.trim(),
  };

  const circledOnly = effectiveReq.mode === 'circled';
  const explicitMention = circledOnly
    ? null
    : detectExplicitRecipientMention(effectiveReq.userMessage);
  const recipient =
    explicitMention ??
    (effectiveReq.recipient && effectiveReq.recipient !== 'auto'
      ? effectiveReq.recipient
      : 'assistant');

  // 宛先が「店長」または「客A〜C」のときは、その本人に繋ぐ
  if (!circledOnly && recipient === 'owner') {
    const [owner, telemetryEvent] = await Promise.all([
      voiceOwnerLine({
        situation: 'reply',
        userMessage: effectiveReq.userMessage,
        facts: describeOwnerContext(
          effectiveReq.currentSpec,
          effectiveReq.latestReviews,
          effectiveReq.targetComponentIds,
        ),
        fallback: OWNER_REPLY_FALLBACK,
        history: effectiveReq.history,
      }),
      recordPromptTelemetryEvent({
        cohortId: effectiveReq.cohortId,
        runId: effectiveReq.runId,
        turnIndex: effectiveReq.turnIndex,
        shareForAggregate: effectiveReq.shareForAggregate,
        promptText: effectiveReq.userMessage,
        sessionMode: 'explore',
        recipient: 'owner',
        hasPenAnnotation:
          effectiveReq.targetComponentIds.length > 0 || Boolean(effectiveReq.annotatedImageBase64),
      }),
    ]);
    return {
      intent: 'owner',
      reply: owner.text || OWNER_REPLY_FALLBACK,
      suggestedFollowUps: [],
      telemetryEvent,
    };
  }
  if (!circledOnly && recipient !== 'assistant') {
    const [customerReplies, telemetryEvent] = await Promise.all([
      voiceCohortCustomers(effectiveReq),
      recordPromptTelemetryEvent({
        cohortId: effectiveReq.cohortId,
        runId: effectiveReq.runId,
        turnIndex: effectiveReq.turnIndex,
        shareForAggregate: effectiveReq.shareForAggregate,
        promptText: effectiveReq.userMessage,
        sessionMode: 'explore',
        recipient: 'customer',
        hasPenAnnotation:
          effectiveReq.targetComponentIds.length > 0 || Boolean(effectiveReq.annotatedImageBase64),
      }),
    ]);
    return {
      intent: 'customer',
      customerPersonaId: customerReplies[0]?.personaId,
      customerReplies,
      reply: customerReplies.map((r) => r.text).join('\n'),
      suggestedFollowUps: [],
      telemetryEvent,
    };
  }

  if (!circledOnly && classifyUserMessageIntent(effectiveReq.userMessage) === 'apply') {
    return applyFromChat(effectiveReq);
  }

  const targets = effectiveReq.currentSpec.components.filter((c) =>
    effectiveReq.targetComponentIds.includes(c.id),
  );
  // 赤ペンで部品を囲んだだけの段階では、手直しの答えは出さず、調べるための質問ボタンだけを出す
  if (circledOnly && targets.length > 0) {
    const targetName = formatCircledTargetLabel(targets);
    return {
      intent: 'explain',
      reply: `${targetName}ですね。気になることを聞くか、どう手直ししたいか教えてください`,
      suggestedFollowUps: [...CIRCLE_INQUIRY_FOLLOW_UPS],
    };
  }

  const componentFacts = effectiveReq.currentSpec.components.map((c) => ({
    id: c.id,
    type: c.type,
    role: c.role,
    label: c.label,
    stateKey: c.stateKey,
    actions: c.actions,
    behavior: c.behavior,
    trapWhen: c.trapWhen,
    closePosition: c.closePosition,
    closeSize: c.closeSize,
    closeLabel: c.closeLabel,
    customHtml: c.customHtml,
    size: c.style.size,
    skin: c.style.skin,
  }));

  const prompt = [
    `あなたは開発スタジオの進行役・実装担当であるお助けAIです。ユーザーが${ACTIVE_THEME.appName}についてチャットで発言しました。`,
    '【まず発言を次の3つのどれかに分けてください】',
    '- question: 画面や部品の仕組み・お客さんの反応・このスタジオの使い方について聞いている。「嘘じゃない？」「これ何？」のような、部品に対する問いかけも question です。',
    '- change: 何をどう変えるかが決まる手直しや機能追加の指示。例：広告を消して、閉じるボタンを左下に移動して、広告を画面内のクーポン枠に変えて、注文ボタンを動くように直す、豆の診断機能を追加して。',
    '- change には、対象の部品と目的（誰がどうできるようにしたいか）がそろっていれば、具体的な直し方を言っていない指示も含めます。直し方を決めるのはあなたの役目です。例：定期便を望む人だけが自分で選べるようにしたい、広告はクーポンを使いたい人にだけ役立つようにして、この注意書きを目が遠い人でも読めるようにして（赤ペンで囲んでいる）。',
    '- unclear: 感想・不満・あいさつ・相づち、または変え方がまだ決まっていない発言。例：赤いボタンが邪魔、うーん、使いにくい。',
    '- 対象も目的もない丸投げ（いい感じにして、全部直して）と、対象の部品がない目的だけの発言（高齢の人でも買いやすくして）も unclear です。どこが悪いかを見つけるのは学習者自身なので、あなたが部品を選んではいけません。',
    '- unclear で目的は言えているときは、目的を短く受け止めてから、どの部品でそう感じたかを尋ねてください。',
    '【重要：消すだけに誘導せず、自由な工夫や＋α機能の実装を受け入れる】',
    '- ポップアップ広告や定期便、カウントダウン、ボタンなどは、ただ消すだけでなく、閉じるボタンの位置（左下・右下・下部バー等）や大きさを変える、画面内のバナーや割引ボタンに変える、独自の＋α機能（診断や特典など）を足すといったあらゆる手直しが可能です。「できません」と断ったり消すことだけに誘導したりせず、学習者の工夫をそのまま change として実装へ回してください。',
    '【question のときの絶対ルール】',
    '1. 画面の現在の仕様（componentFacts と state）にある事実だけを根拠に答えてください。',
    '   - ボタンの actions が空配列 [] の場合は、押しても何も起きない壊れた設定になっていると事実を伝えてください。お支払い方法が未選択だから等の推測や嘘の説明は絶対にしないでください。',
    '   - role が decoy の部品は押しても何も起きないおとり、role が trap の部品は最初からチェック等が入っている仕掛け、role が popup の部品は時間差で出る広告です。',
    '   - role が info や total の部品、actions が設定された primary_action の部品について聞かれたら、正常に動いている部品だと事実を伝えてください。',
    '   - お客さんの反応についてお助けAIに聞かれたら、お客さんが実際に操作した記録を根拠に、その人が何手目に何をしたかを、何も起きなかった操作も省かずに起きた順に伝え、最後にやめた理由・気づかなかったことを記録の言葉どおりに伝えてください。この場合に限り3文・100文字まで使ってよいです。記録が今の画面より前のものなら、どの画面での記録かも添えてください。',
    '   - 記録に書かれていないことは推測で足さないでください。どう直せばよいか・どの部品が悪いかの判断や具体的な直し方の提案は言わず、起きたことだけを伝えてください。',
    '   - まだお客さんの反応を見ていないときは、左下のお客さんの反応を見るで試してもらえると伝えてください。',
    '   - 部品の仕様だけでは分からない動き（時間で変わる表示など）は、画面の仕組みの補足にある事実で答えてください。',
    '   - 「嘘じゃない？」のような感想まじりの問いかけには、まず気持ちを短く受け止めてから、事実を答えてください。',
    '   - 「消した方がいい？」「残すべき？」のように判断を求められたら、自分の意見は言わず、決めるのはあなたで、お客さんの反応や店長の考えを聞けば確かめられると伝えてください。',
    '2. 画面や部品についての質問なら、最後にどう変えたいかを短く尋ねてください。使い方の質問なら尋ねなくてかまいません。',
    '【unclear のときのルール】',
    '1. 発言を短く受け止め、どこをどう変えたいかを聞き返してください。手直しはまだしません。',
    '【どこが悪いか・どう直すかを決めるのは学習者自身です。絶対ルール】',
    '1. 説明は、学習者が自分で指した部品についてだけ出してください。自分で指した部品とは、赤ペンで囲んだ部品、発言で名前や見た目を挙げた部品、直前に自分で押して画面通知が出た部品です。',
    '2. 学習者が何も指していないときは、どの部品が悪いか・どこを直すべきかを一切言わないでください。左のスマホで実際に買ってみるか、気になるところを赤ペンで囲むよう短く促すだけにしてください。',
    '3. 画面全体の問題点を挙げたり、まだ話題に出ていない部品を勧めたり、具体的な手直しの答えを先回りして教えたりしないでください。',
    '【共通のルール】',
    '1. 誰が喜ぶか・誰が困るかといったお客さんや店長の気持ちの先回り代弁や、それっぽい一般論は一切言わないでください（反応はお客さんや店長本人が言います）。',
    '2. 若年層のチャットのように読みやすくするため、1文は35文字以内で短く区切り、合計2文以内（70文字以内）・括弧なしにしてください。',
    '3. change のときは reply を空文字にしてください。',
    '',
    `【話題の部品】\n${targets.length > 0 ? JSON.stringify(targets) : '特定の部品は指していません'}`,
    `【画面にある全部品の実際の仕様】\n${JSON.stringify(componentFacts)}`,
    `【画面の仕組みの補足】\n${(ACTIVE_THEME.explainerNotes ?? []).join('\n') || 'なし'}`,
    `【スタジオの使い方】\n${STUDIO_GUIDE.join('\n')}`,
    `【現在の状態初期値】\n${JSON.stringify(effectiveReq.currentSpec.state)}`,
    `【お客さんが実際に操作した記録】\n${formatLatestReviews(effectiveReq.latestReviews)}`,
    `【これまでの会話と画面通知】\n${formatHistory(effectiveReq.history)}`,
    circledOnly
      ? '【ユーザーの発言】\nまだ何も言っていません。赤ペンで話題の部品を囲んだところです。question として扱い、囲んだ部品が何かを1文で受け止めてから、どうしたいかを短く尋ねてください。部品の仕組みの説明や直し方の提案は、聞かれるまでしないでください。'
      : `【ユーザーの発言】\n${effectiveReq.userMessage}`,
  ].join('\n');

  const triageParts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> =
    [{ text: prompt }];
  const b64 = extractBase64Data(effectiveReq.annotatedImageBase64);
  if (b64) {
    triageParts.push({
      inlineData: {
        mimeType: 'image/png',
        data: b64,
      },
    });
  }

  let parsed: {
    intent: 'question' | 'change' | 'unclear';
    reply: string;
    pointedComponentIds?: string[];
  };
  try {
    const response = await generateContentWithTimeout(
      aiInfo,
      {
        model: aiInfo.model,
        contents: triageParts,
        config: {
          responseMimeType: 'application/json',
          responseSchema: TRIAGE_SCHEMA,
          temperature: 0.3,
          ...thinkingConfig(),
        },
      },
      TIMEOUTS.reply,
    );
    parsed = parseJsonResponse<typeof parsed>(response.text, 'triage');
  } catch (err) {
    if (err instanceof AiNotConfiguredError) throw err;
    console.warn('[gemini:triage] 振り分け応答に失敗したためフォールバックします:', describeErrorForLog(err));
    if (circledOnly && targets.length > 0) {
      const targetName = formatCircledTargetLabel(targets);
      return {
        intent: 'explain',
        reply: `${targetName}ですね。気になることを聞くか、どう手直ししたいか教えてください`,
        suggestedFollowUps: [...CIRCLE_INQUIRY_FOLLOW_UPS],
      };
    }
    return {
      intent: 'clarify',
      reply: 'どこをどう変えたいか、もう少し詳しく教えてください',
      suggestedFollowUps: [],
    };
  }

  // 囲んだだけのときは、AIが改造と判断しても画面は変えない
  if (parsed.intent === 'change' && !circledOnly) {
    return applyFromChat(effectiveReq);
  }

  const existingIds = new Set(effectiveReq.currentSpec.components.map((c) => c.id));
  const pointedIds = new Set<string>([
    ...effectiveReq.targetComponentIds,
    ...(parsed.pointedComponentIds ?? []).filter((id) => existingIds.has(id)),
  ]);
  // 調査用の質問ボタンは、赤ペンで囲んだ直後だけに出し、通常の質問・相談では手直し案のボタンを出さない
  const filteredFollowUps =
    circledOnly && pointedIds.size > 0 ? [...CIRCLE_INQUIRY_FOLLOW_UPS] : [];

  const telemetryEvent =
    !circledOnly && effectiveReq.userMessage
      ? await recordPromptTelemetryEvent({
          cohortId: effectiveReq.cohortId,
          runId: effectiveReq.runId,
          turnIndex: effectiveReq.turnIndex,
          shareForAggregate: effectiveReq.shareForAggregate,
          promptText: effectiveReq.userMessage,
          sessionMode: 'explore',
          recipient: 'assistant',
          hasPenAnnotation:
            effectiveReq.targetComponentIds.length > 0 ||
            Boolean(effectiveReq.annotatedImageBase64),
        })
      : undefined;

  return {
    intent: parsed.intent === 'unclear' ? 'clarify' : 'explain',
    reply: cleanJapaneseText(parsed.reply, 'どこをどう変えたいか教えてください'),
    suggestedFollowUps: filteredFollowUps,
    telemetryEvent,
  };
}

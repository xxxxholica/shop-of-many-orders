import {
  clipToVisibleRects,
  detectHitComponents,
  type LayeredRect2D,
  type Point2D,
  type Rect2D,
} from '../shared/hitTest.ts';
import {
  createInitialState,
  hasCompletionAction,
  isSafeStateKey,
  withDerivedValues,
} from '../shared/appRuntime.ts';
import {
  ACTIVE_THEME,
  ALL_REVIEWER_PERSONAS,
  getReviewersByCohort,
  TARGET_COHORTS,
} from '../shared/initialTheme.ts';
import {
  classifyUserMessageIntent,
  detectCheckpointRevertCommand,
  detectExplicitRecipientMention,
} from '../shared/intent.ts';
import { buildOwnerClosingLines } from '../shared/owner.ts';
import {
  shouldShowNextFrictionNotice,
  simulatePersonaComputerUse,
} from '../shared/sandboxValidator.ts';
import { describeOtherCohortImpact, describeOtherCohortReport } from '../shared/otherCohorts.ts';
import { splitIntoChatChunks } from '../shared/chatText.ts';
import {
  formatPredictionBadgeText,
  formatPredictionReportLine,
  outcomePredictionLabel,
  PREDICTION_OPTIONS,
  scorePredictionRound,
  type PredictionRound,
} from '../shared/prediction.ts';
import {
  AGGREGATE_TIP_BOX_CAPTIONS,
  AGGREGATE_TIP_BOX_IDS,
  type AggregateTipBox,
  type AggregateTipBoxId,
  type PublicAggregateTips,
} from '../shared/aggregateTips.ts';
import { fillTurnRefs } from '../shared/promptTelemetry.ts';
import { buildInitialSpec, buildScenarioSpec, type DifficultyLevel } from '../shared/theme.ts';
import { describeComponentFunctionName } from '../shared/workSteps.ts';
import type {
  AppSpec,
  ChatHistoryEntry,
  ChatResponse,
  Checkpoint,
  EvaluationResponse,
  LatestReviewRecord,
  OwnerLineRequest,
  OwnerLineResponse,
  RecordDecisionRequest,
  ReviewerPersona,
  ReviewOutcome,
  StateValue,
  TargetCohortId,
  TranscribeResponse,
  TtsResponse,
} from '../shared/types.ts';
import { ICONS } from './icons.ts';
import {
  findSandboxComponentElement,
  getStageRelativeRect,
  querySandboxComponentElements,
  renderSandboxApp,
  scrollSandboxBy,
} from './sandboxRenderer.ts';
import { captureSandboxSnapshot } from './snapshot.ts';
import { measureFixedLayoutBounds } from './fixedLayoutMeasure.ts';
import { prepareSimulationInput } from '../shared/customComponentProfile.ts';

/**
 * チャットの発言者
 * - system: スマホ画面の操作通知・改造の作業ログ・エラー表示
 * - assistant: お助けエージェント（進行案内・事実の説明・改造の実行報告）
 * - customer: お客さん本人の発言（来店時の希望・反応確認時の感想・質問への返答）
 * - owner: 依頼主の店長（変更直後に売上・定期便の立場からリアクション・質問への返答）
 * - user: 学習者（改善の意思決定者）
 */
type MessageRole = 'system' | 'assistant' | 'customer' | 'owner' | 'user';

interface TimelineMessage {
  id: string;
  role: MessageRole;
  text: string;
  tone?: 'default' | 'warning';
  /** role が customer のときの発言者 */
  personaId?: string;
  showCohortSelector?: boolean;
  /** この選択肢で選ばれた顧客層 */
  chosenCohortId?: TargetCohortId;
  /** 次のお客さんチャレンジ等で選ばれた選択肢キー */
  chosenOption?: string;
  /** 改造完了後にチャット内からも「お客さんの反応を見る」を押せるようにする */
  showEvaluateAction?: boolean;
  /** お客さんの反応を見る前に、3人がそれぞれどう思うかを1人ずつ予想する選択肢 */
  predictionPrompt?: {
    draft: Record<string, ReviewOutcome>;
  };
  /** 1人ずつ予想していく途中で確定した予想の1行（反応結果が出るまで表示し、押せば選び直せる） */
  predictionStepDone?: {
    personaIndex: number;
    draftBefore: Record<string, ReviewOutcome>;
  };
  /** この発言が対象にしている、赤ペンで囲んだ部品 */
  targetComponentIds?: string[];
  /** お客さん・お助けAI・店長が画面上で指し示している部品のID一覧 */
  pointedComponentIds?: string[];
  /** 改造の作業の過程。システムのメッセージとして1行ずつ進み、完了後もそのまま残す */
  workLog?: {
    steps: Array<{ text: string; done: boolean; componentId?: string }>;
    finished: boolean;
  };
  /** 赤ペンで囲んだことへのお助けエージェントの反応。囲み直したら置き換える */
  circleReaction?: boolean;
  /** 赤ペンで囲んだ部品を、画面の文字ではなく役割の名前でシステム行に出す */
  circledNoticeIds?: string[];
  /** お助けエージェントの回答に続けて押せる返信候補 */
  followUps?: string[];
  /** クリア後の「ほかのお客さんにもチャレンジしますか」の発言 */
  showNextChallenge?: boolean;
  /** お客さんの反応 */
  review?: {
    outcome: ReviewOutcome;
    predictedOutcome?: ReviewOutcome;
  };
}

interface StudioState {
  /** 3人全員が満足して買えた顧客層 */
  clearedCohortIds: TargetCohortId[];
  checkpoints: Checkpoint[];
  activeCheckpointIndex: number;
  liveAppState: Record<string, StateValue>;
  selectedCohortId?: TargetCohortId;
  /** 今の挑戦で、お客さんの反応を見る前に立てた予想の記録 */
  predictionRounds: PredictionRound[];
  interactionMode: 'play' | 'pen';
  penPoints: Point2D[];
  circledComponentIds: string[];
  /** 会話の話題になっている部品。囲み直すか改造するまで次の発言に引き継ぐ */
  topicTargetIds: string[];
  inputText: string;
  /** メッセージの宛先（assistant / owner / ペルソナID） */
  recipient: string;
  /** 書き込み中として表示している発言者（順番に出す演出用） */
  typingMessage?: Pick<TimelineMessage, 'role' | 'personaId'>;
  /** お助けエージェントが考え中に送ろうとした。返事が届いたら送る */
  pendingSend: boolean;
  /** 改造の作業ログを流している間や指し示し中に、スマホで光らせる部品 */
  justChangedIds: string[];
  /** お客さん・お助けAI・店長が指し示して色ペンで囲んでいる部品 */
  speakerPointed?: {
    componentIds: string[];
    role: 'customer' | 'assistant' | 'owner';
    startedAt: number;
  } | null;
  /** 選んだ3人が全員達成してレポートを出したあと。やり直すまで手直しは受け付けない */
  runFinished: boolean;
  /** 1回の挑戦ごとの番号。客層を選ぶたび・やり直すたびに新しくする */
  runId: string;
  /**
   * この挑戦で学習者が送った発言の本文。添字が発言の番号。本文はこのブラウザだけが持ち、サーバーには残さない。
   * 振り返りで指示を引用するときに、サーバーが返す目印をこの本文に置き換える
   */
  runTurnTexts: string[];
  /** 記録をみんなの集計に加えることに同意したか。まだ答えていなければ undefined（加えないとして扱う） */
  aggregateConsent?: boolean;
  /** 今の挑戦の記録を集計に加えるか。同意の切り替えは次の挑戦から効くので、挑戦の始まりに決める */
  runShareForAggregate: boolean;
  /** 今の挑戦の終わり（クリア・離脱）をもう記録したか */
  runEnded: boolean;
  chatLoading: boolean;
  evaluateLoading: boolean;
  /** Cloud TTS による音声読み上げのON/OFF */
  voiceEnabled: boolean;
  /** マイク音声入力の録音中フラグ */
  isRecordingAudio: boolean;
  /** 録音した音声をGeminiで整形・文字起こし中フラグ */
  isTranscribingAudio: boolean;
  messages: TimelineMessage[];
}

const OWNER = ACTIVE_THEME.owner;

function formatCheckpointLabel(cp: Pick<Checkpoint, 'stepNumber'>): string {
  return cp.stepNumber === 0 ? '最初' : `手直し${cp.stepNumber}`;
}

// ============================================================================
// 記録をみんなの集計に加えるかの同意と、挑戦の始まり・終わりの記録
// ============================================================================

const PUBLIC_NOTICE_ENABLED = import.meta.env.VITE_PUBLIC_NOTICE === '1';
const CONSENT_STORAGE_KEY = 'ponkotsu:aggregateConsent';
/** この時間なにも操作しなければ、挑戦をやめたとみなす */
const IDLE_END_MS = 30 * 60 * 1000;
let lastActivityAt = Date.now();

/** 同意はこのブラウザにだけ覚えておく。案内ウィンドウを出さない設定では常に未同意として扱う */
function loadAggregateConsent(): boolean | undefined {
  if (!PUBLIC_NOTICE_ENABLED) return undefined;
  try {
    const v = window.localStorage.getItem(CONSENT_STORAGE_KEY);
    return v === 'yes' ? true : v === 'no' ? false : undefined;
  } catch {
    return undefined;
  }
}

function saveAggregateConsent(consent: boolean): void {
  state.aggregateConsent = consent;
  try {
    window.localStorage.setItem(CONSENT_STORAGE_KEY, consent ? 'yes' : 'no');
  } catch {
    // 覚えておけなくても、この画面を開いている間は state の値で動く
  }
}

// ============================================================================
// 公開ホスト時のみ出すウィンドウ：利用上の注意と、集計への協力
// ============================================================================

const TERMS_SUMMARY = '学習用の架空のお店の注文画面を、AIと一緒に直すツールです。';

/** 利用上の注意。1項目を1行に収まる長さにする。リンクを含む項目は、文字とリンクを並べて書く */
type TermsPart = string | { text: string; href: string; external?: boolean };
const TERMS_ITEMS: TermsPart[][] = [
  ['学習用の試作品で、商用サービスではありません'],
  ['実際の注文や支払いは起きません'],
  ['指示と画面の画像は Google Cloud の Gemini に送られます。文章は保存しません'],
  ['名前や連絡先など、個人を特定できる情報は入力しないでください'],
  ['お客さんや店長の返事はAIが作ったもので、誤りを含むことがあります'],
  // 1分あたり・同時・1日あたりの上限があることを先に知らせる。数値は環境変数で変わるので書かない
  ['送れる回数と1日の利用量には上限があり、超えると一時的に使えなくなります'],
  ['不審なリクエストは、予告なく制限・拒否することがあります'],
  ['予告なく公開を止めたり内容を変えたりすることがあります'],
];

/** 運営者と問い合わせ先。環境変数が設定されている項目だけを注意のすぐ下に並べる */
function buildOperatorItems(): Array<{ label: string; value: TermsPart[] }> {
  const items: Array<{ label: string; value: TermsPart[] }> = [];
  const env = import.meta.env as Record<string, string | undefined>;
  const team = (env.VITE_OPERATOR_TEAM ?? '').trim();
  const representative = (env.VITE_OPERATOR_NAME ?? '').trim();
  const representativeUrl = (env.VITE_OPERATOR_URL ?? '').trim();
  const email = (env.VITE_OPERATOR_EMAIL ?? '').trim();

  if (team) {
    items.push({ label: '運営', value: [team] });
  }
  if (representative) {
    items.push({
      label: '代表者',
      value: /^https?:\/\//i.test(representativeUrl)
        ? [{ text: representative, href: representativeUrl, external: true }]
        : [representative],
    });
  }
  if (email && /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    items.push({ label: '問い合わせ', value: [{ text: email, href: `mailto:${email}` }] });
  }
  return items;
}

/** 文字とリンクを並べて、1つの要素に書き込む */
function appendTermsParts(target: HTMLElement, parts: TermsPart[]): void {
  for (const part of parts) {
    if (typeof part === 'string') {
      target.append(part);
      continue;
    }
    const a = document.createElement('a');
    a.href = part.href;
    a.textContent = part.text;
    if (part.external) {
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
    }
    target.appendChild(a);
  }
}

const AGGREGATE_NOTES = [
  '送るのは指示の型・回数・結果・途中でやめたかだけ。',
  '文章や個人を特定できる情報は送りません。',
  '画面下の「このツールについて」からいつでも変えられ、次の挑戦から反映されます。',
];

function mountModalDialog(options: {
  guardSelector: string;
  overlayClass?: string;
  dialogClass?: string;
  titleId: string;
  titleText: string;
  summaryText: string;
  bodyNodes: Node[];
  actionLabel: string;
  dismissible: boolean;
  onFinish?: (close: () => void) => void;
}): void {
  if (document.querySelector(options.guardSelector)) return;
  const overlay = document.createElement('div');
  overlay.className = options.overlayClass ? `terms-overlay ${options.overlayClass}` : 'terms-overlay';

  const dialog = document.createElement('div');
  dialog.className = options.dialogClass ? `terms-dialog ${options.dialogClass}` : 'terms-dialog';
  dialog.setAttribute('role', 'dialog');
  dialog.setAttribute('aria-modal', 'true');
  dialog.setAttribute('aria-labelledby', options.titleId);

  const title = document.createElement('h2');
  title.id = options.titleId;
  title.className = 'terms-title';
  title.textContent = options.titleText;

  const summary = document.createElement('p');
  summary.className = 'terms-summary';
  summary.textContent = options.summaryText;

  const actionBtn = document.createElement('button');
  actionBtn.type = 'button';
  actionBtn.className = 'terms-action';
  actionBtn.textContent = options.actionLabel;

  const close = () => {
    document.removeEventListener('keydown', onKeydown);
    overlay.remove();
  };
  const finish = () => {
    if (options.onFinish) {
      options.onFinish(close);
    } else {
      close();
    }
  };
  const onKeydown = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && options.dismissible) finish();
  };
  actionBtn.addEventListener('click', finish);
  document.addEventListener('keydown', onKeydown);
  if (options.dismissible) {
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) finish();
    });
  }

  dialog.append(title, summary, ...options.bodyNodes, actionBtn);
  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
  dialog.scrollTop = 0;
  actionBtn.focus({ preventScroll: true });
}

/**
 * 利用上の注意と、集計への協力のウィンドウを出す。
 * first：ページを読み込むたびに出す。同意して始めるを押すまで閉じられず、押すと店長の紹介が始まる
 * review：画面の下の「このツールについて」から開いたとき。閉じると、集計への協力の変更を反映する
 */
function openTermsDialog(mode: 'first' | 'review', onAccepted?: () => void): void {
  if (document.querySelector('.terms-overlay')) return;
  const noticeHeading = document.createElement('h3');
  noticeHeading.className = 'terms-heading';
  noticeHeading.textContent = '利用上の注意';
  const noticeList = document.createElement('ol');
  noticeList.className = 'terms-list';
  for (const parts of TERMS_ITEMS) {
    const li = document.createElement('li');
    appendTermsParts(li, parts);
    noticeList.appendChild(li);
  }
  const bodyNodes: Node[] = [noticeHeading, noticeList];
  const operatorItems = buildOperatorItems();
  if (operatorItems.length > 0) {
    const operatorLine = document.createElement('dl');
    operatorLine.className = 'terms-operator';
    for (const item of operatorItems) {
      const pair = document.createElement('div');
      pair.className = 'terms-operator-item';
      const dt = document.createElement('dt');
      dt.textContent = item.label;
      const dd = document.createElement('dd');
      appendTermsParts(dd, item.value);
      pair.append(dt, dd);
      operatorLine.appendChild(pair);
    }
    bodyNodes.push(operatorLine);
  }

  // 自分で選ぶ所。枠で囲んで目立たせ、チェックはまだ選んでいなければ外しておき、選んだことがあればその答えを出す
  const consentBox = document.createElement('div');
  consentBox.className = 'terms-consent';
  const checkLabel = document.createElement('label');
  checkLabel.className = 'terms-check';
  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.checked = state.aggregateConsent === true;
  const checkText = document.createElement('span');
  checkText.textContent = '私の挑戦の記録を、みんなの集計に加える';
  const optionalBadge = document.createElement('span');
  optionalBadge.className = 'terms-optional';
  optionalBadge.textContent = '任意';
  checkLabel.append(checkbox, checkText, optionalBadge);
  const notes = document.createElement('p');
  notes.className = 'terms-notes';
  notes.textContent = AGGREGATE_NOTES.join('');
  consentBox.append(checkLabel, notes);
  bodyNodes.push(consentBox);

  mountModalDialog({
    guardSelector: '.terms-overlay',
    titleId: 'terms-title',
    titleText: mode === 'first' ? 'はじめに' : 'このツールについて',
    summaryText: TERMS_SUMMARY,
    bodyNodes,
    actionLabel: mode === 'first' ? '同意して始める' : '閉じる',
    dismissible: mode === 'review',
    onFinish: (close) => {
      if (mode === 'first') {
        saveAggregateConsent(checkbox.checked);
        close();
        onAccepted?.();
        return;
      }
      const changed = checkbox.checked !== (state.aggregateConsent === true);
      close();
      if (changed) toggleAggregateConsent();
    },
  });
}

/** 「このツールについて」で集計への協力を切り替えたときに反映する。今の挑戦ではなく次の挑戦から効く */
function toggleAggregateConsent(): void {
  const next = state.aggregateConsent !== true;
  saveAggregateConsent(next);
  pushMessage({
    role: 'system',
    text: next
      ? '記録をみんなの集計に加えます。次の挑戦から反映されます'
      : '記録をみんなの集計に加えるのをやめます。次の挑戦から反映されます',
  });
  renderStudio();
  scrollToLatestMessage();
}

const TIP_STAGGER_MS = 70;
const TIP_BOX_MARKS: Record<AggregateTipBoxId, string> = {
  phrasing: '①',
  stumbling: '②',
  blindspot: '③',
};

/** 集計した時刻を、11月10日14時 のような短い言葉にする */
function formatTipsTime(iso: string): string | null {
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return null;
  return `${t.getMonth() + 1}月${t.getDate()}日${t.getHours()}時`;
}

function tipPendingText(box: AggregateTipBox, tips: PublicAggregateTips | null): string {
  if (!tips || box.status === 'unavailable') {
    return '今は集計を読み込めません。時間をおいて開き直してください';
  }
  return box.runsNeeded !== undefined
    ? `集計中です。あと${box.runsNeeded}件の挑戦で表示されます`
    : '集計中です。もう少し挑戦が集まると表示されます';
}

/**
 * みんなの集計の中身を描く。コツを3つ縦に並べ、それぞれに見出し・本文・根拠のタグを入れる。
 * tips が null なら読み込み中として、枠だけを出す
 */
function renderAggregateTips(
  container: HTMLElement,
  tips: PublicAggregateTips | null,
  options: { loading?: boolean } = {},
): void {
  container.innerHTML = '';
  const isLoading = options.loading === true;
  container.classList.toggle('is-loading', isLoading);

  const list = document.createElement('div');
  list.className = 'tips-list';
  AGGREGATE_TIP_BOX_IDS.forEach((id, index) => {
    const box: AggregateTipBox = tips?.boxes.find((b) => b.id === id) ?? {
      id,
      status: 'unavailable',
      tags: [],
    };
    const card = document.createElement('section');
    card.className = `tip-card is-${isLoading ? 'loading' : box.status}`;
    card.setAttribute('aria-labelledby', `tip-caption-${id}`);

    const caption = document.createElement('div');
    caption.className = 'tip-caption';
    caption.id = `tip-caption-${id}`;
    caption.textContent = `${TIP_BOX_MARKS[id]} ${AGGREGATE_TIP_BOX_CAPTIONS[id]}`;
    card.appendChild(caption);

    if (isLoading) {
      for (const cls of ['tip-skeleton is-heading', 'tip-skeleton', 'tip-skeleton is-short']) {
        const line = document.createElement('div');
        line.className = cls;
        card.appendChild(line);
      }
    } else if (box.status === 'ready' && box.heading && box.body) {
      const heading = document.createElement('h3');
      heading.className = 'tip-heading';
      heading.textContent = box.heading;
      const body = document.createElement('p');
      body.className = 'tip-body';
      body.textContent = box.body;
      card.append(heading, body);
      if (box.tags.length > 0) {
        const tagList = document.createElement('div');
        tagList.className = 'tip-tags';
        for (const tag of box.tags) {
          const tagEl = document.createElement('span');
          tagEl.className = 'tip-tag';
          tagEl.textContent = tag.text;
          tagEl.title = tag.title;
          tagList.appendChild(tagEl);
        }
        card.appendChild(tagList);
      }
    } else {
      const pending = document.createElement('p');
      pending.className = 'tip-pending';
      pending.textContent = tipPendingText(box, tips);
      card.appendChild(pending);
    }

    if (!isLoading) {
      card.classList.add('tip-enter');
      card.style.animationDelay = `${index * TIP_STAGGER_MS}ms`;
    }
    list.appendChild(card);
  });
  container.appendChild(list);

  const source = document.createElement('p');
  source.className = 'tips-source';
  const time = tips?.available ? formatTipsTime(tips.generatedAt) : null;
  source.textContent = isLoading
    ? 'BigQuery の集計を読み込んでいます'
    : time
      ? `BigQuery の集計から・${time}`
      : 'BigQuery の集計につながっていません';
  container.appendChild(source);
}

/**
 * みんなの集計のウィンドウを出す。
 * BigQuery の集計からサーバーが1時間ごとに作ったコツを表示する。開くたびに Gemini は呼ばない
 */
function openAggregateStatsDialog(): void {
  if (document.querySelector('.stats-overlay')) return;
  const bodyBox = document.createElement('div');
  bodyBox.className = 'tips-box';
  bodyBox.setAttribute('aria-live', 'polite');
  renderAggregateTips(bodyBox, null, { loading: true });

  mountModalDialog({
    guardSelector: '.stats-overlay',
    overlayClass: 'stats-overlay',
    dialogClass: 'stats-dialog',
    titleId: 'stats-title',
    titleText: 'みんなの集計',
    summaryText:
      '集計への協力にチェックを入れた挑戦から、みんなの進め方の傾向をコツにまとめました。指示の文章や個人を特定できる情報は使わず、挑戦が3件未満の区分は表示しません。',
    bodyNodes: [bodyBox],
    actionLabel: '閉じる',
    dismissible: true,
  });

  const minWait = sleep(160);
  void Promise.all([
    fetch('/api/aggregate-tips').then(async (res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return (await res.json()) as PublicAggregateTips;
    }),
    minWait,
  ])
    .then(([tips]) => {
      renderAggregateTips(bodyBox, Array.isArray(tips?.boxes) ? tips : null);
    })
    .catch(() => {
      renderAggregateTips(bodyBox, null);
    });
}

/** 集計で、どの問題の画面で遊んだかを分けるための名前 */
function sessionScenarioLabel(): string {
  const params = new URLSearchParams(window.location.search);
  if (params.has('clean')) return 'clean';
  if (params.get('mode') === 'random') return `random:${params.get('level') ?? 'normal'}`;
  const scenarioParam = params.get('scenario');
  return scenarioParam && ACTIVE_THEME.scenarios[scenarioParam] ? scenarioParam : ACTIVE_THEME.defaultScenarioId;
}

type RunEndReason = 'cleared' | 'reset' | 'closed' | 'idle';

/**
 * 挑戦の始まりと終わりを記録する。集計に加えることに同意した挑戦だけ送る。
 * 送るのは、挑戦の番号・客層・問題・終わり方・人数だけで、発言の本文は送らない
 */
function sendRunEvent(
  eventType: 'start' | 'end',
  endReason?: RunEndReason,
  extra: { achievedCount?: number; totalCustomers?: number } = {},
): void {
  if (!state.runShareForAggregate || !state.runId) return;
  if (eventType === 'end') {
    if (state.runEnded) return;
    state.runEnded = true;
  }
  const body = JSON.stringify({
    runId: state.runId,
    eventType,
    endReason,
    cohortId: state.selectedCohortId,
    scenario: sessionScenarioLabel(),
    turnCount: state.runTurnTexts.length,
    ...extra,
  });
  // タブを閉じるときは、ページが消えても届く sendBeacon を使う
  if (endReason === 'closed' && navigator.sendBeacon) {
    navigator.sendBeacon('/api/run-event', new Blob([body], { type: 'application/json' }));
    return;
  }
  void fetch('/api/run-event', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
    keepalive: true,
  }).catch(() => {});
}

/** 挑戦の途中かどうか。客層を選んでから、クリアするか離脱するまで */
function isRunInProgress(): boolean {
  return Boolean(state.selectedCohortId && state.runId && !state.runEnded);
}

window.addEventListener('pagehide', () => {
  if (isRunInProgress()) sendRunEvent('end', 'closed');
});
for (const type of ['pointerdown', 'keydown'] as const) {
  window.addEventListener(type, () => {
    lastActivityAt = Date.now();
  }, { passive: true });
}
window.setInterval(() => {
  if (isRunInProgress() && Date.now() - lastActivityAt > IDLE_END_MS) sendRunEvent('end', 'idle');
}, 60_000);

/**
 * 体験の画面を決める。
 * - 既定：題材の固定の問題（?scenario=名前 で別の問題も選べる）
 * - ?mode=random：妨害要素の組み合わせを毎回ランダムにする練習用。
 *   ?level=easy|normal|hard で壊れ方の多さを、?seed=数字 で組み合わせの固定を選べる
 * - ?clean：妨害要素を入れる前の画面（題材のデザイン調整用）
 */
function createSessionSpec(): AppSpec {
  const params = new URLSearchParams(window.location.search);
  if (params.has('clean')) {
    return structuredClone(ACTIVE_THEME.baseSpec);
  }
  if (params.get('mode') !== 'random') {
    const scenarioParam = params.get('scenario');
    const scenarioId =
      scenarioParam && ACTIVE_THEME.scenarios[scenarioParam]
        ? scenarioParam
        : ACTIVE_THEME.defaultScenarioId;
    return buildScenarioSpec(ACTIVE_THEME, scenarioId).spec;
  }
  const seedParam = Number(params.get('seed'));
  const seed = Number.isFinite(seedParam) && params.has('seed') ? seedParam : Date.now();
  const levelParam = params.get('level');
  const level: DifficultyLevel =
    levelParam === 'easy' || levelParam === 'hard' ? levelParam : 'normal';
  return buildInitialSpec(ACTIVE_THEME, seed, level).spec;
}

const INITIAL_SPEC = createSessionSpec();

const state: StudioState = {
  checkpoints: [
    {
      id: 'cp-0',
      stepNumber: 0,
      commitMessage: 'かいぞう前',
      createdAt: new Date().toISOString(),
      spec: INITIAL_SPEC,
    },
  ],
  activeCheckpointIndex: 0,
  liveAppState: createInitialState(INITIAL_SPEC),
  selectedCohortId: undefined,
  clearedCohortIds: [],
  predictionRounds: [],
  interactionMode: 'play',
  penPoints: [],
  circledComponentIds: [],
  topicTargetIds: [],
  inputText: '',
  recipient: 'assistant',
  typingMessage: undefined,
  pendingSend: false,
  justChangedIds: [],
  speakerPointed: null,
  runFinished: false,
  runId: '',
  runTurnTexts: [],
  aggregateConsent: loadAggregateConsent(),
  runShareForAggregate: false,
  runEnded: true,
  chatLoading: false,
  evaluateLoading: false,
  voiceEnabled: true,
  isRecordingAudio: false,
  isTranscribingAudio: false,
  messages: [],
};

function getCurrentCheckpoint(): Checkpoint {
  return state.checkpoints[state.activeCheckpointIndex] ?? state.checkpoints[0];
}

/**
 * 反応を見た結果のうち、満足の欄にもう出したお客さん。お客さんが言い終えたところで1人ずつ足す。
 * 記録がない結果は、全員出したものとして扱う
 */
const revealedReviewerIds = new WeakMap<EvaluationResponse, Set<string>>();

function isReviewRevealed(evaluation: EvaluationResponse, reviewerId: string): boolean {
  return revealedReviewerIds.get(evaluation)?.has(reviewerId) ?? true;
}

/** その画面で、いま選んでいる顧客層のお客さんに反応を見てもらった結果。別の顧客層の結果は使わない */
function evaluationFor(cp: Checkpoint | undefined): EvaluationResponse | undefined {
  const evaluation = cp?.evaluation;
  return evaluation && evaluation.cohortId === state.selectedCohortId ? evaluation : undefined;
}

function getCurrentSpec(): AppSpec {
  return getCurrentCheckpoint().spec;
}

/** 指定した番号以前で、お客さんの反応を見た一番新しい記録を探す（1回の挑戦は1つの客層なので、客層では絞らない） */
function findLatestCohortEvaluation(
  fromIndex: number = state.activeCheckpointIndex,
): { checkpoint: Checkpoint; index: number; evaluation: EvaluationResponse } | undefined {
  for (let i = fromIndex; i >= 0; i--) {
    const cp = state.checkpoints[i];
    const evaluation = evaluationFor(cp);
    if (evaluation) {
      return { checkpoint: cp, index: i, evaluation };
    }
  }
  return undefined;
}

/** 今の画面か、それより前で一番新しくお客さんの反応を見た画面の、操作の記録 */
function buildLatestReviewRecord(): LatestReviewRecord | undefined {
  const found = findLatestCohortEvaluation();
  if (found) {
    const { checkpoint: cp, index: i, evaluation } = found;
    return {
      screenLabel: formatCheckpointLabel(cp),
      isCurrentScreen: i === state.activeCheckpointIndex,
      reviews: evaluation.results.map((r) => ({
        name: findPersona(r.reviewerId)?.name ?? r.reviewerId,
        outcome: r.outcome,
        actionTrace: r.actionTrace ?? [],
        frictionComponentIds: r.frictionComponentIds ?? [],
      })),
      ownerMetrics: evaluation.owner?.metrics,
    };
  }
  if (!state.selectedCohortId) return undefined;
  const initialCp = state.checkpoints[0];
  if (!initialCp) return undefined;
  const reviewers = getReviewersByCohort(state.selectedCohortId);
  return {
    screenLabel: formatCheckpointLabel(initialCp),
    isCurrentScreen: state.activeCheckpointIndex === 0,
    reviews: reviewers.map((p) => {
      const sim = simulatePersonaComputerUse(initialCp.spec, p);
      return {
        name: p.name,
        outcome: sim.outcome,
        actionTrace: sim.actionTrace,
        frictionComponentIds: sim.frictionComponentIds,
      };
    }),
  };
}

function findPersona(personaId: string | undefined): ReviewerPersona | undefined {
  if (!personaId) return undefined;
  return ALL_REVIEWER_PERSONAS.find((p) => p.id === personaId);
}

let currentAudioEl: HTMLAudioElement | null = null;
const ttsPlaybackQueue: Array<{
  text: string;
  speaker: string;
  onDone?: () => void;
}> = [];
let isPlayingTts = false;
let currentTtsDoneResolver: (() => void) | null = null;
const ttsPrefetchCache = new Map<string, Promise<TtsResponse>>();

/** ブラウザ標準TTSフォールバック時も全11キャラクターで高さと速さを変える */
const BROWSER_VOICE_PROFILES: Record<string, { pitch: number; rate: number }> = {
  owner: { pitch: 0.75, rate: 1.0 },
  assistant: { pitch: 1.15, rate: 1.12 },
  kazuko: { pitch: 0.82, rate: 0.9 },
  shigeru: { pitch: 0.65, rate: 0.9 },
  harue: { pitch: 0.95, rate: 0.95 },
  kenta: { pitch: 0.92, rate: 1.2 },
  misaki: { pitch: 1.08, rate: 1.15 },
  takumi: { pitch: 0.82, rate: 1.1 },
  shota: { pitch: 1.02, rate: 1.08 },
  hina: { pitch: 1.28, rate: 1.14 },
  ren: { pitch: 0.86, rate: 1.05 },
};

function isSpokenMessage(msg: Pick<TimelineMessage, 'role' | 'text'>): boolean {
  return (
    Boolean(msg.text.trim()) &&
    (msg.role === 'assistant' || msg.role === 'owner' || msg.role === 'customer')
  );
}

function speakerOfMessage(msg: Pick<TimelineMessage, 'role' | 'personaId'>): string {
  return msg.role === 'customer' ? (msg.personaId ?? 'assistant') : msg.role;
}

function sanitizeTtsText(text: string): string {
  return text.replace(/`/g, '').trim();
}

function fetchTtsAudio(text: string, speaker: string): Promise<TtsResponse> {
  const trimmed = sanitizeTtsText(text);
  const key = `${speaker}::${trimmed}`;
  const cached = ttsPrefetchCache.get(key);
  if (cached) return cached;
  const promise = postJson<TtsResponse>('/api/tts', {
    text: trimmed,
    speaker,
  }).catch(() => {
    ttsPrefetchCache.delete(key);
    return { fallbackBrowserTts: true };
  });
  ttsPrefetchCache.set(key, promise);
  return promise;
}

function stopAllTtsPlayback(): void {
  while (ttsPlaybackQueue.length > 0) {
    const item = ttsPlaybackQueue.shift();
    item?.onDone?.();
  }
  isPlayingTts = false;
  if (currentAudioEl) {
    currentAudioEl.pause();
    currentAudioEl = null;
  }
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
  if (currentTtsDoneResolver) {
    const resolve = currentTtsDoneResolver;
    currentTtsDoneResolver = null;
    resolve();
  }
}

async function processNextTtsInQueue(): Promise<void> {
  if (isPlayingTts || !state.voiceEnabled) {
    if (!state.voiceEnabled) {
      while (ttsPlaybackQueue.length > 0) {
        ttsPlaybackQueue.shift()?.onDone?.();
      }
    }
    return;
  }
  const next = ttsPlaybackQueue.shift();
  if (!next) return;
  isPlayingTts = true;
  try {
    const res = await fetchTtsAudio(next.text, next.speaker);
    if (!state.voiceEnabled) {
      next.onDone?.();
      isPlayingTts = false;
      return;
    }
    if (res.audioContent) {
      await new Promise<void>((resolve) => {
        const audio = new Audio(`data:audio/mp3;base64,${res.audioContent}`);
        currentAudioEl = audio;
        let finished = false;
        const done = () => {
          if (finished) return;
          finished = true;
          if (currentAudioEl === audio) currentAudioEl = null;
          if (currentTtsDoneResolver === done) currentTtsDoneResolver = null;
          resolve();
        };
        currentTtsDoneResolver = done;
        audio.addEventListener('ended', done, { once: true });
        audio.addEventListener('error', done, { once: true });
        audio.play().catch(done);
      });
    } else if ('speechSynthesis' in window) {
      await new Promise<void>((resolve) => {
        const utter = new SpeechSynthesisUtterance(sanitizeTtsText(next.text));
        utter.lang = 'ja-JP';
        const profile = BROWSER_VOICE_PROFILES[next.speaker] ?? BROWSER_VOICE_PROFILES.assistant;
        utter.pitch = profile.pitch;
        utter.rate = profile.rate;
        let finished = false;
        const done = () => {
          if (finished) return;
          finished = true;
          if (currentTtsDoneResolver === done) currentTtsDoneResolver = null;
          resolve();
        };
        currentTtsDoneResolver = done;
        utter.addEventListener('end', done, { once: true });
        utter.addEventListener('error', done, { once: true });
        window.speechSynthesis.speak(utter);
      });
    }
  } catch {
    // 音声合成失敗時は静かにスキップ
  } finally {
    next.onDone?.();
    isPlayingTts = false;
    if (state.voiceEnabled && ttsPlaybackQueue.length > 0) {
      void processNextTtsInQueue();
    }
  }
}

function playCloudTtsForMessage(msg: TimelineMessage, onDone?: () => void): void {
  if (!state.voiceEnabled || !isSpokenMessage(msg)) {
    onDone?.();
    return;
  }
  const speaker = speakerOfMessage(msg);
  ttsPlaybackQueue.push({ text: msg.text, speaker, onDone });
  void processNextTtsInQueue();
}

function playTtsAndWait(msg: TimelineMessage): Promise<void> {
  if (!state.voiceEnabled || !isSpokenMessage(msg)) {
    return Promise.resolve();
  }
  return new Promise<void>((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      window.clearTimeout(safetyTimer);
      resolve();
    };
    // 万一ブラウザの音声再生イベントが返らない場合でも12秒で次へ進める安全弁
    const safetyTimer = window.setTimeout(finish, 12_000);
    playCloudTtsForMessage(msg, finish);
  });
}

function pushMessage(
  msg: Omit<TimelineMessage, 'id'>,
  options?: { skipAutoTts?: boolean },
): TimelineMessage {
  const full: TimelineMessage = {
    id: `${msg.role}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    ...msg,
  };
  state.messages.push(full);
  // プレイヤー以外が部品を囲むのは、ここだけにする。囲んだ発言の下に「〜は〜を気にしています」の行が出るので、囲みと行がいつもそろう
  if (
    full.role !== 'user' &&
    full.role !== 'system' &&
    full.pointedComponentIds &&
    full.pointedComponentIds.length > 0
  ) {
    highlightSandboxComponent(full.pointedComponentIds, full.role);
  }
  if (!options?.skipAutoTts) {
    playCloudTtsForMessage(full);
  }
  return full;
}

async function postJson<T>(url: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  const data = (await res.json().catch(() => ({}))) as T & { error?: string; code?: string };
  if (!res.ok) {
    const error = new Error(data.error ?? `通信エラーが発生しました (${res.status})`) as Error & { code?: string };
    error.code = data.code;
    throw error;
  }
  return data;
}

function describeError(err: unknown): string {
  return err instanceof Error ? err.message : '通信に失敗しました';
}

const appRoot = document.getElementById('app');
if (!appRoot) {
  throw new Error('#app element not found');
}

let shellEl: HTMLElement | null = null;
let headerSlotEl: HTMLElement | null = null;
let phoneFrameEl: HTMLElement | null = null;
let sandboxStageEl: HTMLElement | null = null;
let sandboxAppMountEl: HTMLElement | null = null;
let penCanvasEl: HTMLCanvasElement | null = null;
let bottomBarEl: HTMLElement | null = null;
let chatSlotEl: HTMLElement | null = null;
let chatScrollEl: HTMLElement | null = null;
let lastSandboxRenderKey = '';
const seenMessageIds = new Set<string>();
const recipientFirstSeenAt = new Map<string, number>();
const RECIPIENT_ENTER_DURATION_MS = 420;

/**
 * 順番に出す発言。typingMs が 0 のものは書き込み中の表示をせず、少し間を置いて出す。
 * onFinished は発言を言い終えたときに呼ぶ。音声OFFなら吹き出しを出したとき、飛ばしたときはその場で呼ぶ
 */
interface QueuedMessage {
  message: Omit<TimelineMessage, 'id'>;
  typingMs: number;
  onFinished?: () => void;
}

const QUEUE_PAUSE_MS = 650;
const messageQueue: QueuedMessage[] = [];
const queueTimers = new Set<number>();
let activeQueuedItem: QueuedMessage | null = null;
let isQueueRunning = false;
let queueRunId = 0;
/** 吹き出しを出して、まだ言い終えていない発言の onFinished */
let speakingFinishedCallback: (() => void) | null = null;

/** 言い終えていない発言を、言い終えたことにする。呼んだら true */
function finishSpeakingMessage(): boolean {
  const callback = speakingFinishedCallback;
  speakingFinishedCallback = null;
  callback?.();
  return Boolean(callback);
}

function clearQueueTimers(): void {
  for (const id of queueTimers) {
    window.clearTimeout(id);
  }
  queueTimers.clear();
}

function sleepWithQueueTimer(ms: number): Promise<void> {
  return new Promise<void>((resolve) => {
    const id = window.setTimeout(() => {
      queueTimers.delete(id);
      resolve();
    }, ms);
    queueTimers.add(id);
  });
}

/** 発言の長さに応じた書き込み時間（音声OFF時のテンポ） */
function typingDurationFor(text: string): number {
  return Math.min(2000, 850 + text.length * 38);
}

function enqueueMessages(items: QueuedMessage[]): void {
  // キューに入った時点で裏で音声データを先読みし、吹き出しが出た瞬間に遅延ゼロで再生できるようにする
  if (state.voiceEnabled) {
    for (const item of items) {
      if (isSpokenMessage(item.message)) {
        void fetchTtsAudio(item.message.text, speakerOfMessage(item.message));
      }
    }
  }
  messageQueue.push(...items);
  if (!isQueueRunning) {
    isQueueRunning = true;
    void runQueueLoop(++queueRunId);
  }
}

async function waitAfterQueuedMessage(
  runId: number,
  msg: TimelineMessage,
  shouldSpeak: boolean,
): Promise<boolean> {
  if (shouldSpeak) {
    // 現在の吹き出しの音声読み上げが終わるのを待ってから、次の発言の書き込みに進む
    await playTtsAndWait(msg);
    if (runId !== queueRunId) return false;
    if (finishSpeakingMessage()) renderStudio();
    if (messageQueue.length > 0) {
      await sleepWithQueueTimer(220);
    }
  } else {
    if (finishSpeakingMessage()) renderStudio();
    if (messageQueue.length > 0) {
      await sleepWithQueueTimer(QUEUE_PAUSE_MS);
    }
  }
  return runId === queueRunId;
}

async function runQueueLoop(runId: number): Promise<void> {
  while (runId === queueRunId && messageQueue.length > 0) {
    const next = messageQueue.shift()!;
    activeQueuedItem = next;
    const spoken = isSpokenMessage(next.message);

    if (next.typingMs > 0 && next.message.role !== 'system') {
      state.typingMessage = { role: next.message.role, personaId: next.message.personaId };
      renderStudio();
      scrollToLatestMessage();

      if (state.voiceEnabled && spoken) {
        // 音声ONのときは、最小限の「…」表示（550ms）と音声先読み完了（最大1800ms）が揃い次第すぐ吹き出しを出す
        const speaker = speakerOfMessage(next.message);
        await Promise.all([
          sleepWithQueueTimer(550),
          Promise.race([fetchTtsAudio(next.message.text, speaker), sleepWithQueueTimer(1800)]),
        ]);
      } else {
        await sleepWithQueueTimer(next.typingMs);
      }
    } else {
      state.typingMessage = undefined;
      renderStudio();
      await sleepWithQueueTimer(next.typingMs > 0 ? next.typingMs : 260);
    }

    if (runId !== queueRunId) return;

    activeQueuedItem = null;
    state.typingMessage = undefined;
    const shouldSpeak = state.voiceEnabled && spoken;
    const full = pushMessage(next.message, { skipAutoTts: true });
    speakingFinishedCallback = next.onFinished ?? null;
    renderStudio();
    scrollToLatestMessage();

    if (!(await waitAfterQueuedMessage(runId, full, shouldSpeak))) return;
  }

  if (runId === queueRunId) {
    activeQueuedItem = null;
    clearQueueTimers();
    state.typingMessage = undefined;
    isQueueRunning = false;
    renderStudio();
  }
}

/** チャット欄をタップしたときは、いま待っている発言を1つだけ即座に出してその音声へ切り替える */
function advanceOneQueuedMessage(): void {
  // 話している途中で次へ進めた発言は、言い終えたことにする
  finishSpeakingMessage();
  const item = activeQueuedItem ?? messageQueue.shift() ?? null;
  activeQueuedItem = null;
  clearQueueTimers();
  stopAllTtsPlayback();
  state.typingMessage = undefined;
  const myRunId = ++queueRunId;

  if (!item) {
    isQueueRunning = false;
    renderStudio();
    return;
  }

  const spoken = isSpokenMessage(item.message);
  const shouldSpeak = state.voiceEnabled && spoken;
  const full = pushMessage(item.message, { skipAutoTts: true });
  speakingFinishedCallback = item.onFinished ?? null;
  renderStudio();
  scrollToLatestMessage();

  isQueueRunning = true;
  void (async () => {
    if (!(await waitAfterQueuedMessage(myRunId, full, shouldSpeak))) return;
    await runQueueLoop(myRunId);
  })();
}

/** 学習者が先に操作したときは、残りの発言を待たずにすべて出して順序を保つ */
function flushMessageQueue(): void {
  ++queueRunId;
  isQueueRunning = false;
  clearQueueTimers();
  stopAllTtsPlayback();
  state.typingMessage = undefined;
  finishSpeakingMessage();
  if (activeQueuedItem) {
    const current = activeQueuedItem;
    activeQueuedItem = null;
    pushMessage(current.message, { skipAutoTts: true });
    current.onFinished?.();
  }
  while (messageQueue.length > 0) {
    const item = messageQueue.shift()!;
    pushMessage(item.message, { skipAutoTts: true });
    item.onFinished?.();
  }
}

function isTransientChoiceMessage(m: TimelineMessage): boolean {
  return (
    m.role === 'system' &&
    Boolean(
      (m.followUps && m.followUps.length > 0) || m.showEvaluateAction || m.predictionPrompt,
    )
  );
}

/** 一時的な返信候補・反応確認ボタン・未実行の予想パネルのシステムメッセージを履歴から整理する */
function clearTransientChoiceMessages(): void {
  const hadOpenPrediction = state.messages.some((m) => Boolean(m.predictionPrompt));
  state.messages = state.messages.filter(
    (m) => !isTransientChoiceMessage(m) && !(hadOpenPrediction && m.predictionStepDone),
  );
}

function clearPredictionStepMessages(): void {
  state.messages = state.messages.filter((m) => !m.predictionStepDone);
}

/** 冒頭の会話。店長とお助けエージェントの行は書き込み中の表示を挟んで話す */
function playInitialWelcomeSequence(): void {
  // 開いた瞬間にチャットが空だと何が始まるか分からないので、最初の場面の説明だけはすぐ出す
  const [first, ...rest] = ACTIVE_THEME.introScript;
  const lines = first?.speaker === 'system' && !first.showCohortSelector ? rest : ACTIVE_THEME.introScript;
  if (lines !== ACTIVE_THEME.introScript) {
    pushMessage({ role: 'system', text: first.text });
  }
  const queued: QueuedMessage[] = [];
  for (const line of lines) {
    queued.push({
      message: {
        role: line.speaker,
        text: line.text,
        showCohortSelector: line.showCohortSelector,
      },
      typingMs:
        line.speaker === 'owner' || line.speaker === 'assistant'
          ? typingDurationFor(line.text)
          : 0,
    });
  }
  enqueueMessages(queued);
}

function scrollToLatestMessage(): void {
  window.requestAnimationFrame(() => {
    if (chatScrollEl) {
      chatScrollEl.scrollTop = chatScrollEl.scrollHeight;
    }
  });
}

/**
 * スマホの中で起きたこと（押しても反応がない等）を、画面を邪魔しないよう進行役の発言としてチャットに出す。
 * ボタンを連打しても埋まらないよう、直前と同じ内容なら重ねて出さない。
 */
function postSandboxNotice(message: string, type: 'info' | 'success' | 'warning'): void {
  const last = state.messages[state.messages.length - 1];
  if (last?.role === 'system' && last.text === message) return;
  pushMessage({ role: 'system', tone: type === 'warning' ? 'warning' : 'default', text: message });
  renderStudio();
  scrollToLatestMessage();
}

/**
 * 部品が画面に見えているか。画面下に固定したバーの裏やスクロールの外に隠れている部品は、
 * 赤ペンの線がかかっても選ばないようにする。
 * ポップアップの半透明の背景幕（.inapp-popup-backdrop）は透けて裏の部品が見えるため無視し、
 * ポップアップ本体（.inapp-popup-card）や固定バーの裏に完全に隠れている部品だけを除外する。
 */
function isPartlyVisible(el: HTMLElement, r: DOMRect): boolean {
  const doc = el.ownerDocument;
  const points: Array<[number, number]> = [
    [0.5, 0.5],
    [0.2, 0.2],
    [0.8, 0.2],
    [0.2, 0.8],
    [0.8, 0.8],
  ];
  return points.some(([fx, fy]) => {
    const x = r.left + r.width * fx;
    const y = r.top + r.height * fy;
    const stack = doc.elementsFromPoint
      ? doc.elementsFromPoint(x, y)
      : [doc.elementFromPoint(x, y)].filter((n): n is Element => n !== null);
    const hit = stack.find((node) => !node.classList.contains('inapp-popup-backdrop'));
    return !!hit && (hit === el || el.contains(hit));
  });
}

function collectComponentRects(container: HTMLElement): Rect2D[] {
  const containerRect = container.getBoundingClientRect();
  const elements = querySandboxComponentElements(container);
  const rects: LayeredRect2D[] = [];
  let viewport = { width: containerRect.width, height: containerRect.height };
  for (const el of elements) {
    const id = el.dataset.componentId;
    if (!id) continue;
    const r = el.getBoundingClientRect();
    if (!isPartlyVisible(el, r)) continue;
    const view = el.ownerDocument.defaultView;
    if (el.ownerDocument !== document && view) {
      viewport = { width: view.innerWidth, height: view.innerHeight };
    }
    const rel = getStageRelativeRect(el, containerRect);
    rects.push({
      id,
      left: rel.left,
      top: rel.top,
      right: rel.right,
      bottom: rel.bottom,
      // 画面下に固定した帯と広告は、その下の部品を隠す
      occludes: el.classList.contains('is-sticky') || el.classList.contains('inapp-popup-card'),
      pinned: el.classList.contains('is-sticky'),
    });
  }
  // 隠れた部分や画面の外の部分は囲みの判定に使わない
  return clipToVisibleRects(rects, viewport);
}

const SPEAKER_PEN_COLORS: Record<'customer' | 'assistant' | 'owner', string> = {
  customer: '#f97316', // お客さん：鮮やかなオレンジ
  assistant: '#2563eb', // お助けAI：鮮やかな青
  owner: '#16a34a', // 店長：鮮やかな緑
};
const USER_RED_PEN_COLOR = '#dc2626';

const SPEAKER_PEN_HOLD_MS = 3600;
const SPEAKER_PEN_FADE_MS = 1400;
const SPEAKER_PEN_TOTAL_MS = SPEAKER_PEN_HOLD_MS + SPEAKER_PEN_FADE_MS;

const USER_PEN_HOLD_HIT_MS = 2600;
const USER_PEN_HOLD_MISS_MS = 450;
const USER_PEN_FADE_MS = 1100;

let speakerPenAnimFrame: number | null = null;
let speakerPenWakeTimer: number | null = null;
let userPenFinishedAt: number | null = null;
let userPenAnimFrame: number | null = null;
let userPenWakeTimer: number | null = null;
let lastUserPenSnapshotBase64: string | undefined;

function clearPenFadeTimers(): void {
  if (speakerPenAnimFrame !== null) {
    window.cancelAnimationFrame(speakerPenAnimFrame);
    speakerPenAnimFrame = null;
  }
  if (speakerPenWakeTimer !== null) {
    window.clearTimeout(speakerPenWakeTimer);
    speakerPenWakeTimer = null;
  }
  if (userPenAnimFrame !== null) {
    window.cancelAnimationFrame(userPenAnimFrame);
    userPenAnimFrame = null;
  }
  if (userPenWakeTimer !== null) {
    window.clearTimeout(userPenWakeTimer);
    userPenWakeTimer = null;
  }
}

function buildSpeakerLoopPoints(
  rect: { left: number; top: number; right: number; bottom: number },
  progress: number,
  canvasWidth: number,
  canvasHeight: number,
  seedOffset = 0,
): Point2D[] {
  const pad = 10;
  const safeLeft = Math.max(pad, rect.left);
  const safeRight = Math.min(canvasWidth - pad, rect.right);
  const safeTop = Math.max(pad, rect.top);
  const safeBottom = Math.min(canvasHeight - pad, rect.bottom);
  const cx = (safeLeft + safeRight) / 2;
  const cy = (safeTop + safeBottom) / 2;
  const hw = Math.max(20, (safeRight - safeLeft) / 2 + 4);
  const hh = Math.max(16, (safeBottom - safeTop) / 2 + 4);
  const totalSteps = 84;
  const activeSteps = Math.max(2, Math.round(totalSteps * progress));
  const startAngle = -Math.PI * 0.72 + seedOffset * 0.12;
  const totalSweep = Math.PI * 2.32;
  const pts: Point2D[] = [];

  for (let i = 0; i <= activeSteps; i++) {
    const u = i / totalSteps;
    const angle = startAngle + u * totalSweep;
    const cosA = Math.cos(angle);
    const sinA = Math.sin(angle);
    const nx = Math.sign(cosA) * Math.pow(Math.abs(cosA), 0.56);
    const ny = Math.sign(sinA) * Math.pow(Math.abs(sinA), 0.56);
    const wobble =
      1 +
      0.024 * Math.sin(angle * 2 + 0.8 + seedOffset) +
      0.016 * Math.cos(angle * 3 - 0.5 + seedOffset);
    const secondPassOffset = u > 0.78 ? (u - 0.78) * 18 : 0;
    const rawX = cx + nx * (hw * wobble + secondPassOffset);
    const rawY = cy + ny * (hh * wobble + secondPassOffset * 0.7);
    pts.push({
      x: Math.min(canvasWidth - pad, Math.max(pad, rawX)),
      y: Math.min(canvasHeight - pad, Math.max(pad, rawY)),
    });
  }
  return pts;
}

function traceSmoothPenPath(ctx: CanvasRenderingContext2D, pts: Point2D[]): void {
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  if (pts.length === 2) {
    ctx.lineTo(pts[1].x, pts[1].y);
    return;
  }
  for (let i = 1; i < pts.length - 1; i++) {
    const midX = (pts[i].x + pts[i + 1].x) / 2;
    const midY = (pts[i].y + pts[i + 1].y) / 2;
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, midX, midY);
  }
  const last = pts[pts.length - 1];
  ctx.lineTo(last.x, last.y);
}

/**
 * どんな背景色の上でもはっきり見分けられるよう、白縁＋影を下敷きにして太めの手書きマーカー線を描く。
 * ユーザーの赤ペンと、お客さん・お助けAI・店長の色ペンで共通の描画スタイルを使う。
 */
function drawHighlightedPenPath(
  ctx: CanvasRenderingContext2D,
  pts: Point2D[],
  color: string,
  opacity = 1,
): void {
  if (pts.length < 2 || opacity <= 0.01) return;

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.96)';
  ctx.lineWidth = 10;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.shadowColor = `rgba(0, 0, 0, ${0.38 * opacity})`;
  ctx.shadowBlur = 6;
  ctx.shadowOffsetY = 1.5;
  traceSmoothPenPath(ctx, pts);
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.strokeStyle = color;
  ctx.lineWidth = 5;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  traceSmoothPenPath(ctx, pts);
  ctx.stroke();
  ctx.restore();
}

function drawSpeakerPenLoop(
  ctx: CanvasRenderingContext2D,
  rect: { left: number; top: number; right: number; bottom: number },
  role: 'customer' | 'assistant' | 'owner',
  progress: number,
  opacity: number,
  canvasWidth: number,
  canvasHeight: number,
  seedOffset = 0,
): void {
  const color = SPEAKER_PEN_COLORS[role];
  const pts = buildSpeakerLoopPoints(rect, progress, canvasWidth, canvasHeight, seedOffset);
  drawHighlightedPenPath(ctx, pts, color, opacity);
}

function resolveSpeakerTargetRect(
  container: HTMLElement,
  componentId: string,
  collectedRects: Rect2D[],
  canvasWidth: number,
  canvasHeight: number,
): { left: number; top: number; right: number; bottom: number } | null {
  const matched = collectedRects.filter((r) => r.id === componentId);
  if (matched.length > 0) {
    return {
      left: Math.min(...matched.map((r) => r.left)),
      top: Math.min(...matched.map((r) => r.top)),
      right: Math.max(...matched.map((r) => r.right)),
      bottom: Math.max(...matched.map((r) => r.bottom)),
    };
  }
  const el = findSandboxComponentElement(container, componentId);
  if (!el) return null;
  const rel = getStageRelativeRect(el, container.getBoundingClientRect());
  const left = Math.max(0, rel.left);
  const top = Math.max(0, rel.top);
  const right = Math.min(canvasWidth, rel.right);
  const bottom = Math.min(canvasHeight, rel.bottom);
  if (right - left < 8 || bottom - top < 8) return null;
  return { left, top, right, bottom };
}

function redrawPenCanvas(): void {
  if (!penCanvasEl || !sandboxStageEl) return;
  const rect = sandboxStageEl.getBoundingClientRect();
  if (
    penCanvasEl.width !== Math.round(rect.width) ||
    penCanvasEl.height !== Math.round(rect.height)
  ) {
    penCanvasEl.width = Math.max(1, Math.round(rect.width));
    penCanvasEl.height = Math.max(1, Math.round(rect.height));
  }
  const ctx = penCanvasEl.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, penCanvasEl.width, penCanvasEl.height);

  // お客さん・お助けAI・店長が指し示している部品を、その人のペン色で手書き風に丸く囲み、時間経過でフェードアウトさせる
  if (state.speakerPointed && state.speakerPointed.componentIds.length > 0) {
    const elapsed = Math.max(0, performance.now() - state.speakerPointed.startedAt);
    if (elapsed >= SPEAKER_PEN_TOTAL_MS) {
      state.speakerPointed = null;
      return;
    }
    const rawProgress = Math.min(1, elapsed / 380);
    const easedProgress = Math.max(0.08, 1 - Math.pow(1 - rawProgress, 3));
    const fadeProgress =
      elapsed <= SPEAKER_PEN_HOLD_MS
        ? 0
        : Math.min(1, (elapsed - SPEAKER_PEN_HOLD_MS) / SPEAKER_PEN_FADE_MS);
    const opacity = 0.5 * (1 + Math.cos(Math.PI * fadeProgress));

    const collected = collectComponentRects(sandboxStageEl);
    state.speakerPointed.componentIds.forEach((id, idx) => {
      const targetRect = resolveSpeakerTargetRect(
        sandboxStageEl!,
        id,
        collected,
        penCanvasEl!.width,
        penCanvasEl!.height,
      );
      if (!targetRect) return;
      drawSpeakerPenLoop(
        ctx,
        targetRect,
        state.speakerPointed!.role,
        easedProgress,
        opacity,
        penCanvasEl!.width,
        penCanvasEl!.height,
        idx * 1.3,
      );
    });

    if (elapsed < 600 || elapsed >= SPEAKER_PEN_HOLD_MS) {
      if (speakerPenAnimFrame === null) {
        speakerPenAnimFrame = window.requestAnimationFrame(() => {
          speakerPenAnimFrame = null;
          redrawPenCanvas();
        });
      }
    } else if (speakerPenWakeTimer === null) {
      speakerPenWakeTimer = window.setTimeout(() => {
        speakerPenWakeTimer = null;
        redrawPenCanvas();
      }, Math.max(16, SPEAKER_PEN_HOLD_MS - elapsed));
    }
    return;
  }

  if (state.penPoints.length < 2) return;

  let userOpacity = 1;
  if (userPenFinishedAt !== null) {
    const holdMs =
      state.circledComponentIds.length > 0 ? USER_PEN_HOLD_HIT_MS : USER_PEN_HOLD_MISS_MS;
    const totalMs = holdMs + USER_PEN_FADE_MS;
    const elapsed = Math.max(0, performance.now() - userPenFinishedAt);
    if (elapsed >= totalMs) {
      state.penPoints = [];
      userPenFinishedAt = null;
      return;
    }
    if (elapsed > holdMs) {
      const fadeProgress = Math.min(1, (elapsed - holdMs) / USER_PEN_FADE_MS);
      userOpacity = 0.5 * (1 + Math.cos(Math.PI * fadeProgress));
      if (userPenAnimFrame === null) {
        userPenAnimFrame = window.requestAnimationFrame(() => {
          userPenAnimFrame = null;
          redrawPenCanvas();
        });
      }
    } else if (userPenWakeTimer === null) {
      userPenWakeTimer = window.setTimeout(() => {
        userPenWakeTimer = null;
        redrawPenCanvas();
      }, Math.max(16, holdMs - elapsed));
    }
  }

  drawHighlightedPenPath(ctx, state.penPoints, USER_RED_PEN_COLOR, userOpacity);
}

function attachPenEvents(canvas: HTMLCanvasElement, stage: HTMLElement): void {
  let drawing = false;

  const getLocalPoint = (e: PointerEvent): Point2D => {
    const r = stage.getBoundingClientRect();
    return {
      x: e.clientX - r.left,
      y: e.clientY - r.top,
    };
  };

  canvas.addEventListener('pointerdown', (e) => {
    if (state.interactionMode !== 'pen') return;
    drawing = true;
    canvas.setPointerCapture(e.pointerId);
    clearPenFadeTimers();
    userPenFinishedAt = null;
    lastUserPenSnapshotBase64 = undefined;
    state.speakerPointed = null;
    state.justChangedIds = [];
    state.penPoints = [getLocalPoint(e)];
    redrawPenCanvas();
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!drawing) return;
    state.penPoints.push(getLocalPoint(e));
    redrawPenCanvas();
  });

  const finishStroke = () => {
    if (!drawing) return;
    drawing = false;
    const rects = collectComponentRects(stage);
    const hits = detectHitComponents(state.penPoints, rects);
    state.circledComponentIds = hits;
    userPenFinishedAt = null;
    renderStudio();
    redrawPenCanvas();
    if (hits.length > 0 && sandboxStageEl) {
      lastUserPenSnapshotBase64 = captureSandboxSnapshot(
        getCurrentSpec(),
        state.liveAppState,
        sandboxStageEl,
        penCanvasEl,
      );
    } else {
      lastUserPenSnapshotBase64 = undefined;
    }
    userPenFinishedAt = performance.now();
    redrawPenCanvas();
    // 囲めたら、お助けエージェントがその部品を受け止めて、どうしたいかを尋ねる
    if (hits.length > 0) void reactToCircle(hits);
  };

  canvas.addEventListener('pointerup', finishStroke);
  canvas.addEventListener('pointercancel', finishStroke);

  // 赤ペンのときも画面をスクロールできるようにする。描いた線は画面と一緒に動かないので消し、
  // 囲んだ部品の印（画面側の赤枠）は残す
  canvas.addEventListener(
    'wheel',
    (e) => {
      if (state.interactionMode !== 'pen' || drawing) return;
      e.preventDefault();
      if (!sandboxAppMountEl) return;
      const scrolled = scrollSandboxBy(sandboxAppMountEl, e.deltaY);
      if (scrolled && state.penPoints.length > 0) {
        state.penPoints = [];
        redrawPenCanvas();
      }
    },
    { passive: false },
  );
}

/**
 * 改造後の画面に状態を移す際、初期値が変わっていない状態キーについては
 * 学習者がすでに選んだ値（支払い方法・閉じた広告など）を引き継ぐ。
 */
function preserveUnchangedLiveState(
  prevSpec: AppSpec,
  nextSpec: AppSpec,
  prevLiveState: Record<string, StateValue>,
): Record<string, StateValue> {
  const base = createInitialState(nextSpec);
  for (const s of nextSpec.state) {
    const prevDef = prevSpec.state.find((p) => p.key === s.key);
    if (
      prevDef &&
      prevDef.initialValue === s.initialValue &&
      prevLiveState[s.key] !== undefined
    ) {
      base[s.key] = prevLiveState[s.key];
    }
  }
  return withDerivedValues(nextSpec, base);
}

function resetPenAndPointingState(): void {
  clearPenFadeTimers();
  userPenFinishedAt = null;
  lastUserPenSnapshotBase64 = undefined;
  state.penPoints = [];
  state.circledComponentIds = [];
  state.topicTargetIds = [];
  state.justChangedIds = [];
  state.speakerPointed = null;
}

function pushNewCheckpoint(commitMessage: string, updatedSpec: AppSpec): void {
  const prevSpec = getCurrentSpec();
  // 過去の画面に戻った状態で新しく手直ししたときは、その画面から分岐して以降の履歴を置き換える
  if (state.activeCheckpointIndex < state.checkpoints.length - 1) {
    state.checkpoints = state.checkpoints.slice(0, state.activeCheckpointIndex + 1);
  }
  const nextStepNumber = state.checkpoints.length;
  const newCheckpoint: Checkpoint = {
    id: `cp-${nextStepNumber}-${Date.now()}`,
    stepNumber: nextStepNumber,
    commitMessage,
    createdAt: new Date().toISOString(),
    spec: updatedSpec,
  };

  state.checkpoints.push(newCheckpoint);
  state.activeCheckpointIndex = state.checkpoints.length - 1;
  state.liveAppState = preserveUnchangedLiveState(prevSpec, updatedSpec, state.liveAppState);
  resetPenAndPointingState();
  state.interactionMode = 'play';
}

async function recordDecision(body: RecordDecisionRequest): Promise<void> {
  try {
    await postJson('/api/decision', body);
  } catch (err) {
    // 記録に失敗しても体験の進行は止めない
    console.warn('[studio] 判断の記録に失敗しました:', err);
  }
}

function handleSelectCohort(cohortId: TargetCohortId): void {
  if (state.selectedCohortId) return;
  lastUserTurn = null;
  flushMessageQueue();
  clearTransientChoiceMessages();
  const selector = [...state.messages].reverse().find((m) => m.showCohortSelector);
  if (selector) selector.chosenCohortId = cohortId;
  startCohort(cohortId);
}

/**
 * クリア後に、画面を最初に戻してやり直す。
 * 1回の挑戦は1つの客層だけなので、直した画面を引き継いで別の客層を呼ぶことはしない。
 * sameCohortId を渡すと同じ客層でそのまま始め、渡さなければ客層を選び直す
 */
function handleRestartFromScratch(sameCohortId?: TargetCohortId): void {
  if (isRunInProgress()) sendRunEvent('end', 'reset');
  lastUserTurn = null;
  abortActiveUserSend();
  flushMessageQueue();
  clearTransientChoiceMessages();
  state.runFinished = false;
  state.selectedCohortId = undefined;
  state.predictionRounds = [];
  state.recipient = 'assistant';
  recipientFirstSeenAt.clear();
  state.checkpoints = [
    {
      id: `cp-0-${Date.now()}`,
      stepNumber: 0,
      commitMessage: 'かいぞう前',
      createdAt: new Date().toISOString(),
      spec: structuredClone(INITIAL_SPEC),
    },
  ];
  state.activeCheckpointIndex = 0;
  state.liveAppState = createInitialState(INITIAL_SPEC);
  resetPenAndPointingState();
  pushMessage({ role: 'system', text: '画面を最初の状態に戻しました' });
  if (sameCohortId) {
    startCohort(sameCohortId);
    return;
  }
  const selectorLine = ACTIVE_THEME.introScript.find((l) => l.showCohortSelector);
  pushMessage({
    role: 'system',
    text: selectorLine?.text ?? '誰のために直しますか',
    showCohortSelector: true,
  });
  renderStudio();
  scrollToLatestMessage();
}

function startCohort(cohortId: TargetCohortId): void {
  const cohort = TARGET_COHORTS.find((c) => c.id === cohortId) ?? TARGET_COHORTS[0];
  state.selectedCohortId = cohort.id;
  state.runId = `run-${crypto.randomUUID()}`;
  state.runTurnTexts = [];
  state.predictionRounds = [];
  // 同意の切り替えは、次の挑戦から効く
  state.runShareForAggregate = state.aggregateConsent === true;
  state.runEnded = false;
  lastActivityAt = Date.now();
  sendRunEvent('start');
  state.recipient = 'assistant';
  recipientFirstSeenAt.clear();

  void recordDecision({
    stage: 'select_cohort',
    targetId: cohort.id,
    chosenLabel: cohort.label,
    reason: '',
  });

  void playCohortHearing(cohort, getCurrentSpec());
}

/**
 * 選んだ客層の3人に、今の画面で困ったことを聞く。お客さんの反応を見るときと同じく、
 * 決まった条件で描画して測った座標で判定する
 */
async function playCohortHearing(cohort: (typeof TARGET_COHORTS)[number], currentSpec: AppSpec): Promise<void> {
  const componentBounds = await measureFixedLayoutBounds(currentSpec);
  // 測っている間に、やり直しなどで別の挑戦が始まっていたら話さない
  if (state.selectedCohortId !== cohort.id || getCurrentSpec() !== currentSpec) return;
  const reviewers = getReviewersByCohort(cohort.id);
  const hearingQueue: QueuedMessage[] = [
    {
      message: {
        role: 'system',
        text: `${cohort.label}の3人に、今の画面で困ったことをヒアリングしました`,
      },
      typingMs: 0,
    },
  ];

  let abandonedCount = 0;
  const unhappyNames: string[] = [];
  for (const persona of reviewers) {
    const sim = simulatePersonaComputerUse(currentSpec, persona, componentBounds);
    hearingQueue.push({
      message: { role: 'customer', personaId: persona.id, text: persona.greeting },
      typingMs: typingDurationFor(persona.greeting),
    });
    if (sim.outcome !== 'achieved') {
      abandonedCount++;
      unhappyNames.push(persona.name);
      hearingQueue.push({
        message: {
          role: 'customer',
          personaId: persona.id,
          text: sim.simulatedComment,
          pointedComponentIds: sim.frictionComponentIds,
        },
        typingMs: typingDurationFor(sim.simulatedComment),
      });
    }
  }

  const guideLines =
    abandonedCount > 0
      ? [
          abandonedCount === reviewers.length
            ? '3人とも満足して買えずに帰ってしまっていますね…'
            : `${unhappyNames.map((n) => (n.startsWith('客') ? n : `${n}さん`)).join('と')}が、満足して買えずに帰ってしまっていますね…`,
          '左のスマホを実際に触って確かめながら、直したい場所を赤ペンで囲んで指示してください！',
        ]
      : [
          'この3人が満足して買える画面に直しましょう！',
          '左のスマホを触って、直したい場所を赤ペンで囲んで指示してくださいね',
        ];

  for (const line of guideLines) {
    hearingQueue.push({
      message: { role: 'assistant', text: line },
      typingMs: typingDurationFor(line),
    });
  }

  enqueueMessages(hearingQueue);
}

/** AIに渡す直近の会話。スマホ画面の反応なし通知も含めて文脈が伝わるようにする */
/** 店長が同じ言い回しを繰り返さないよう、これまでの店長のせりふだけを渡す（会話の古い部分も含める） */
function buildOwnerHistory(): ChatHistoryEntry[] {
  return state.messages
    .filter((m) => m.role === 'owner')
    .slice(-10)
    .map((m) => ({ speaker: OWNER.name, text: m.text }));
}

function buildChatHistory(): ChatHistoryEntry[] {
  return state.messages
    .filter((m) => m.text.trim().length > 0 && !isTransientChoiceMessage(m))
    .slice(-12)
    .map((m) => ({
      speaker:
        m.role === 'user'
          ? 'ユーザー'
          : m.role === 'assistant'
            ? 'お助けAI'
            : m.role === 'owner'
              ? OWNER.name
              : m.role === 'system'
                ? 'スマホ画面の通知'
                : `お客さんの${findPersona(m.personaId)?.name ?? ''}`,
      text: m.text,
    }));
}

/**
 * 発言を送る。送信と同時にスマホ画面の囲みは消すが、囲んだ部品は話題として次の発言にも引き継ぐ。
 * 改造の指示であれば改造案カードを出さずに直接パッチを適用し、店長が変更にリアクションする。
 */
/** 赤ペンで囲んだ部品について、お助けエージェントが受け止めて尋ねる。画面は変えず、囲みも残す */
let circleRequestSeq = 0;
let circleInFlight = false;

// ============================================================================
// 待ち時間の表示
// ============================================================================

/** 返事を待ち始めてからこの秒数を過ぎたら、経過した秒数を出す */
const WAIT_NOTE_AFTER_SEC = 5;
/** この秒数を過ぎたら、混み合っているかもしれないことと、送り直してよいことを添える */
const WAIT_BUSY_AFTER_SEC = 15;
/** 今の待ちが始まった時刻。待っていないときは null */
let waitStartedAt: number | null = null;

function isWaitingForAi(): boolean {
  return state.chatLoading || state.evaluateLoading;
}

/** 待っている間の1行。止まったのか動いているのかが分かるよう、経過した秒数を出す */
function waitNoteText(elapsedSec: number): string {
  const seconds = `\`${elapsedSec}秒\``;
  if (elapsedSec < WAIT_BUSY_AFTER_SEC) return `待っています ${seconds}`;
  const retry = state.evaluateLoading
    ? '少しして、もう一度お客さんの反応を見てください'
    : '少しして、もう一度送ってください';
  return `混み合っているようです。このまま待つか、${retry} ${seconds}`;
}

function waitElapsedSec(): number {
  return waitStartedAt === null ? 0 : Math.floor((performance.now() - waitStartedAt) / 1000);
}

function buildWaitNoteRow(): HTMLElement {
  const row = document.createElement('div');
  row.className = 'chat-system-row chat-wait-note';
  const line = document.createElement('div');
  line.className = 'chat-system-line is-wait';
  appendFormattedSystemText(line, waitNoteText(waitElapsedSec()));
  row.appendChild(line);
  return row;
}

/** 1秒ごとに待ち時間を数え、チャットの表示を作り直さずに、待っている間の1行だけを書き換える */
function tickWaitNote(): void {
  if (!isWaitingForAi()) {
    waitStartedAt = null;
    chatScrollEl?.querySelector('.chat-wait-note')?.remove();
    return;
  }
  if (waitStartedAt === null) waitStartedAt = performance.now();
  if (waitElapsedSec() < WAIT_NOTE_AFTER_SEC) return;
  const stream = chatScrollEl;
  if (!stream) return;
  const existing = stream.querySelector('.chat-wait-note');
  const next = buildWaitNoteRow();
  if (existing) {
    existing.replaceWith(next);
  } else {
    stream.appendChild(next);
    scrollToLatestMessage();
  }
}

window.setInterval(tickWaitNote, 1000);

const FOLLOW_UP_PROMPT_TEXT = '気になることを聞くか、直したい内容をチャットで伝えてください';

async function reactToCircle(targetIds: string[]): Promise<void> {
  // 誰のために直すかを選ぶ前や、改造中・質問への回答待ちのときは割り込まない
  if (!state.selectedCohortId) return;
  if (state.chatLoading && !circleInFlight) return;
  lastUserTurn = null;
  const seq = ++circleRequestSeq;
  circleInFlight = true;
  flushMessageQueue();
  // 前に囲んだときの反応がチャットの最後に残っていれば、置き換える
  while (state.messages.length > 0 && state.messages[state.messages.length - 1].circleReaction) {
    state.messages.pop();
  }
  clearTransientChoiceMessages();
  state.topicTargetIds = [...targetIds];
  state.chatLoading = true;
  renderStudio();
  scrollToLatestMessage();
  try {
    const spec = getCurrentSpec();
    const annotatedImageBase64 =
      lastUserPenSnapshotBase64 ??
      (sandboxStageEl
        ? captureSandboxSnapshot(spec, state.liveAppState, sandboxStageEl, penCanvasEl)
        : undefined);
    const data = await postJson<ChatResponse>('/api/chat', {
      currentSpec: spec,
      latestReviews: buildLatestReviewRecord(),
      targetComponentIds: targetIds,
      userMessage: '',
      cohortId: state.selectedCohortId,
      runId: state.runId,
      annotatedImageBase64,
      history: buildChatHistory(),
      mode: 'circled',
    });
    if (seq !== circleRequestSeq) return;
    // 囲んだ部品は、画面の文字を読み上げず、役割の名前でシステム行に出す
    pushMessage({
      role: 'system',
      text: '',
      circledNoticeIds: [...targetIds],
      targetComponentIds: targetIds,
      circleReaction: true,
    });
    const chunks = splitIntoChatChunks(data.reply);
    for (const chunk of chunks) {
      pushMessage({
        role: 'assistant',
        text: chunk,
        targetComponentIds: targetIds,
        circleReaction: true,
      });
    }
    if (data.suggestedFollowUps && data.suggestedFollowUps.length > 0) {
      pushMessage({
        role: 'system',
        text: FOLLOW_UP_PROMPT_TEXT,
        targetComponentIds: targetIds,
        circleReaction: true,
        followUps: data.suggestedFollowUps,
      });
    }
  } catch (err) {
    if (seq === circleRequestSeq) {
      pushMessage({ role: 'system', tone: 'warning', text: describeError(err) });
    }
  } finally {
    if (seq === circleRequestSeq) {
      circleInFlight = false;
      state.chatLoading = false;
      renderStudio();
      scrollToLatestMessage();
      sendPendingIfReady();
    }
  }
}

const sleep = (ms: number) => new Promise<void>((resolve) => window.setTimeout(resolve, ms));
const WORK_STEP_INTERVAL_MS = 450;

/** 準備中の進捗文言を更新する */
function setPrepWorkStep(log: NonNullable<TimelineMessage['workLog']>, text: string): void {
  log.steps = [{ text, done: false }];
  renderStudio();
  scrollToLatestMessage();
}

/** 作業ログの行を足す。前の行は終わったことにする */
function addWorkStep(
  log: NonNullable<TimelineMessage['workLog']>,
  text: string,
  done = false,
  componentId?: string,
): void {
  for (const step of log.steps) step.done = true;
  log.steps.push({ text, done, componentId });
  renderStudio();
  scrollToLatestMessage();
}

function startWorkLog(targetIds: string[]): TimelineMessage {
  return pushMessage({
    role: 'system',
    text: '',
    targetComponentIds: targetIds,
    workLog: { steps: [{ text: '指示を読んでいます', done: false }], finished: false },
  });
}

let highlightClearTimer: number | null = null;
/**
 * 指し示すために、隠れている部品を出そうとして切り替えた画面の状態。
 * 確認画面のボタンを指すと確認画面へ切り替わるので、指し終えたら元に戻す。
 * 指している間に学習者がスマホを操作して値が変わっていれば、戻さない
 */
let forcedVisibility: { key: string; previous: StateValue | undefined; forced: StateValue } | null = null;

function restoreForcedVisibility(): void {
  if (!forcedVisibility) return;
  const { key, previous, forced } = forcedVisibility;
  forcedVisibility = null;
  if (String(state.liveAppState[key]) !== String(forced)) return;
  state.liveAppState = withDerivedValues(getCurrentSpec(), {
    ...state.liveAppState,
    [key]: previous as StateValue,
  });
}

/**
 * スマホ画面の本文だけをスクロールして、部品を真ん中に出す。
 * scrollIntoView だとページ全体まで動くことがあるので、本文のスクロール位置を直接変える。
 * 画面下に固定した帯や広告は本文の外にあるので、スクロールしない
 */
function scrollSandboxElementIntoView(el: HTMLElement): void {
  const scroller = el.closest<HTMLElement>('.sandbox-app-body');
  if (!scroller) return;
  const elRect = el.getBoundingClientRect();
  const scrollerRect = scroller.getBoundingClientRect();
  const offset = elRect.top - scrollerRect.top - (scrollerRect.height - elRect.height) / 2;
  const maxTop = scroller.scrollHeight - scroller.clientHeight;
  const top = Math.max(0, Math.min(maxTop, scroller.scrollTop + offset));
  scroller.scrollTo({ top, behavior: 'smooth' });
}

/**
 * 発言やシステムメッセージ内の部品名・作業行が指し示すスマホ内の部品を表示し、
 * 発言者の場合はそのペン色（お客さん＝オレンジ、お助けAI＝青、店長＝緑）で手書き風に丸く囲む。
 * （手直しするには学習者自身が赤ペンで囲む必要があるため、circledComponentIds は変更しない）
 */
function highlightSandboxComponent(
  componentIdOrIds: string | string[],
  role?: 'customer' | 'assistant' | 'owner',
): void {
  const ids = Array.from(
    new Set((Array.isArray(componentIdOrIds) ? componentIdOrIds : [componentIdOrIds]).filter(Boolean)),
  ).slice(0, 1);
  if (ids.length === 0) return;

  if (highlightClearTimer !== null) {
    window.clearTimeout(highlightClearTimer);
    highlightClearTimer = null;
  }
  clearPenFadeTimers();
  restoreForcedVisibility();
  const spec = getCurrentSpec();
  const primaryId = ids[0];
  const comp = spec.components.find((c) => c.id === primaryId);
  if (comp) {
    if (
      comp.visibleWhen &&
      comp.visibleWhen.operator === 'eq' &&
      isSafeStateKey(comp.visibleWhen.stateKey) &&
      String(state.liveAppState[comp.visibleWhen.stateKey]) !== String(comp.visibleWhen.value)
    ) {
      forcedVisibility = {
        key: comp.visibleWhen.stateKey,
        previous: state.liveAppState[comp.visibleWhen.stateKey],
        forced: comp.visibleWhen.value,
      };
      state.liveAppState = {
        ...state.liveAppState,
        [comp.visibleWhen.stateKey]: comp.visibleWhen.value,
      };
    }
    if (comp.type === 'popup' && comp.stateKey && isSafeStateKey(comp.stateKey)) {
      state.liveAppState = {
        ...state.liveAppState,
        [comp.stateKey]: 'open',
      };
    } else if (comp.type !== 'popup') {
      for (const p of spec.components) {
        if (
          p.type === 'popup' &&
          p.stateKey &&
          isSafeStateKey(p.stateKey) &&
          state.liveAppState[p.stateKey] === 'open'
        ) {
          state.liveAppState = {
            ...state.liveAppState,
            [p.stateKey]: 'closed',
          };
        }
      }
    }
  }
  state.justChangedIds = role ? [] : ids;
  state.speakerPointed = role
    ? {
        componentIds: ids,
        role,
        startedAt: performance.now(),
      }
    : null;
  renderStudio();
  // 描き直した直後のフレームで、スマホ画面は元のスクロール位置に戻される。
  // そのあとでスクロールしないと、広告を閉じたときなどに戻されて、画面外の部品が見えないままになる
  const stage = sandboxStageEl;
  if (stage) {
    window.requestAnimationFrame(() => {
      const el = findSandboxComponentElement(stage, primaryId);
      if (el) scrollSandboxElementIntoView(el);
    });
  }
  const durationMs = role ? SPEAKER_PEN_TOTAL_MS : 1800;
  highlightClearTimer = window.setTimeout(() => {
    highlightClearTimer = null;
    state.justChangedIds = [];
    state.speakerPointed = null;
    // 指し終えたら、隠れていた部品を出すために切り替えた画面を元に戻す
    restoreForcedVisibility();
    renderStudio();
  }, durationMs);
}

/**
 * 改造の結果を、システムのメッセージとして1行ずつ見せ、完了後もそのまま残す。
 * 行ごとにスマホのその部品を光らせる。
 */
async function playWorkLog(
  logMessage: TimelineMessage,
  steps: Array<{ text: string; componentId?: string }>,
  rolledBack: boolean,
  signal?: AbortSignal,
): Promise<void> {
  const log = logMessage.workLog!;
  if (rolledBack) {
    log.steps = [{ text: '画面がうまく表示できなかったので、元に戻しました', done: true }];
    logMessage.text = log.steps[0].text;
    renderStudio();
    scrollToLatestMessage();
  } else {
    // 準備中の行は消して、実際に行った作業の行だけをシステムメッセージとして残す
    log.steps = [];
    state.speakerPointed = null;
    for (const step of steps) {
      if (signal?.aborted) return;
      state.justChangedIds = step.componentId ? [step.componentId] : [];
      addWorkStep(log, step.text, false, step.componentId);
      if (step.componentId && sandboxStageEl) {
        const el = findSandboxComponentElement(sandboxStageEl, step.componentId);
        el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
      await sleep(WORK_STEP_INTERVAL_MS);
      if (signal?.aborted) return;
      log.steps[log.steps.length - 1].done = true;
    }
    state.justChangedIds = [];
    if (log.steps.length === 0) {
      addWorkStep(log, '画面を書き換えました', true);
    }
    logMessage.text = log.steps.map((s) => s.text).join(' / ');
  }
  log.finished = true;
  renderStudio();
}

/** 考え中に送ろうとした文を、返事が届いたあとに送る */
function sendPendingIfReady(): void {
  if (!state.pendingSend || state.chatLoading) return;
  state.pendingSend = false;
  if (state.inputText.trim()) void handleSendMessage();
  else renderStudio();
}

let activeMediaRecorder: MediaRecorder | null = null;
let recordedAudioChunks: Blob[] = [];

function pickSupportedAudioMimeType(): string {
  if (typeof MediaRecorder === 'undefined') return '';
  for (const candidate of ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']) {
    if (MediaRecorder.isTypeSupported(candidate)) return candidate;
  }
  return '';
}

async function toggleVoiceRecording(): Promise<void> {
  if (state.isRecordingAudio && activeMediaRecorder) {
    activeMediaRecorder.stop();
    return;
  }
  if (!state.selectedCohortId || state.chatLoading || state.isTranscribingAudio) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedAudioChunks = [];
    const mimeType = pickSupportedAudioMimeType();
    const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
    activeMediaRecorder = recorder;
    state.isRecordingAudio = true;
    renderStudio();

    recorder.addEventListener('dataavailable', (e) => {
      if (e.data.size > 0) recordedAudioChunks.push(e.data);
    });
    recorder.addEventListener('stop', () => {
      for (const track of stream.getTracks()) track.stop();
      state.isRecordingAudio = false;
      activeMediaRecorder = null;
      const resolvedMime = recorder.mimeType || mimeType || 'audio/webm';
      const blob = new Blob(recordedAudioChunks, { type: resolvedMime });
      recordedAudioChunks = [];
      if (blob.size < 100) {
        renderStudio();
        return;
      }
      const reader = new FileReader();
      reader.onloadend = async () => {
        const dataUrl = typeof reader.result === 'string' ? reader.result : '';
        if (!dataUrl) {
          renderStudio();
          return;
        }
        state.isTranscribingAudio = true;
        renderStudio();
        try {
          const spec = getCurrentSpec();
          const hasPenAnnotation =
            state.circledComponentIds.length > 0 || state.penPoints.length > 0;
          const targetComponentIds = hasPenAnnotation
            ? [...state.circledComponentIds]
            : [...state.topicTargetIds];
          const annotatedImageBase64 =
            hasPenAnnotation && sandboxStageEl
              ? captureSandboxSnapshot(spec, state.liveAppState, sandboxStageEl, penCanvasEl)
              : undefined;
          const res = await postJson<TranscribeResponse>('/api/transcribe', {
            audioBase64: dataUrl,
            audioMimeType: resolvedMime,
            currentSpec: spec,
            targetComponentIds,
            annotatedImageBase64,
          });
          if (res.hasMeaningfulSpeech && res.cleanedText.trim()) {
            const cleaned = res.cleanedText.trim();
            state.inputText = state.inputText.trim()
              ? `${state.inputText.trim()} ${cleaned}`
              : cleaned;
            const switched = detectExplicitRecipientMention(state.inputText);
            if (switched) state.recipient = switched;
          } else {
            pushMessage({
              role: 'system',
              text: '音声が聞き取れなかったか、手直しに関係する言葉が見つかりませんでした',
            });
          }
        } catch (err) {
          pushMessage({ role: 'system', tone: 'warning', text: describeError(err) });
        } finally {
          state.isTranscribingAudio = false;
          renderStudio();
          focusComposerInput(true);
        }
      };
      reader.readAsDataURL(blob);
    });
    recorder.start();
  } catch {
    state.isRecordingAudio = false;
    pushMessage({
      role: 'system',
      tone: 'warning',
      text: 'マイクの使用が許可されていないか、音声入力に対応していません',
    });
    renderStudio();
  }
}

/** 赤ペンで囲んだときに出る質問ボタンの文言から、送り先の相手を決める */
function resolveRecipientForFollowUp(text: string): 'assistant' | 'owner' | 'customer' {
  if (text.includes('店長')) return 'owner';
  if (text.includes('お客さん')) return 'customer';
  return 'assistant';
}

interface LastUserTurnSnapshot {
  /** 送信したユーザー発言メッセージのID */
  userMessageId: string;
  /** 入力欄に戻すテキスト（自由入力から送ったときのみ復元） */
  restoreInputText?: string;
  /** 送信前のメッセージ履歴（一時的な選択肢も含む） */
  messagesBefore: TimelineMessage[];
  /** 送信前のチェックポイント一覧と現在位置 */
  checkpointsBefore: Checkpoint[];
  activeCheckpointIndexBefore: number;
  liveAppStateBefore: Record<string, StateValue>;
  /** 送信前の発言一覧の長さ */
  turnTextsLengthBefore: number;
  /** 送信前の赤ペン・話題の対象部品 */
  circledComponentIdsBefore: string[];
  topicTargetIdsBefore: string[];
  penPointsBefore: Point2D[];
  /** 送信前の宛先 */
  recipientBefore: string;
}

let lastUserTurn: LastUserTurnSnapshot | null = null;
let activeUserSendAbort: AbortController | null = null;
let activeWorkLogTimers: number[] = [];

function abortActiveUserSend(): void {
  if (activeUserSendAbort) {
    activeUserSendAbort.abort();
    activeUserSendAbort = null;
  }
  for (const t of activeWorkLogTimers) {
    window.clearTimeout(t);
  }
  activeWorkLogTimers = [];
}

function canCancelOrUndoLastUserTurn(): boolean {
  if (!lastUserTurn || state.evaluateLoading || state.runFinished) return false;
  return state.messages.some((m) => m.id === lastUserTurn!.userMessageId);
}

/**
 * 送信中（AIの応答待ち）の中断、または直前に送信した発言の取り消しを行う。
 * 発言とそれに続く返事・画面変更を送信前の状態に戻し、入力していた文面を入力欄に復元する。
 */
function handleCancelLastUserMessage(): void {
  if (!canCancelOrUndoLastUserTurn() || !lastUserTurn) return;
  const snap = lastUserTurn;
  lastUserTurn = null;
  abortActiveUserSend();
  flushMessageQueue();
  stopAllTtsPlayback();

  state.messages = snap.messagesBefore;
  state.checkpoints = snap.checkpointsBefore;
  state.activeCheckpointIndex = snap.activeCheckpointIndexBefore;
  state.liveAppState = snap.liveAppStateBefore;
  state.runTurnTexts.length = snap.turnTextsLengthBefore;
  state.circledComponentIds = snap.circledComponentIdsBefore;
  state.topicTargetIds = snap.topicTargetIdsBefore;
  state.penPoints = snap.penPointsBefore;
  state.recipient = snap.recipientBefore;
  if (snap.restoreInputText !== undefined) {
    state.inputText = snap.restoreInputText;
  }
  state.chatLoading = false;
  state.pendingSend = false;
  state.justChangedIds = [];
  state.typingMessage = undefined;

  renderStudio();
  scrollToLatestMessage();
  focusComposerInput(true);
}

async function handleSendMessage(
  overrideText?: string,
  targetIdsOverride?: string[],
): Promise<void> {
  if (!state.selectedCohortId) return;
  const text = (overrideText ?? state.inputText).trim();
  if (!text) return;
  if (state.chatLoading) {
    // 入力欄から送ろうとしたときは、取りこぼさずに返事のあとで送る
    if (overrideText === undefined) {
      state.pendingSend = true;
      renderStudio();
    }
    return;
  }
  abortActiveUserSend();
  flushMessageQueue();
  const snapshotBeforeSend: Omit<LastUserTurnSnapshot, 'userMessageId'> = {
    restoreInputText: overrideText === undefined ? text : undefined,
    messagesBefore: structuredClone(state.messages),
    checkpointsBefore: structuredClone(state.checkpoints),
    activeCheckpointIndexBefore: state.activeCheckpointIndex,
    liveAppStateBefore: structuredClone(state.liveAppState),
    turnTextsLengthBefore: state.runTurnTexts.length,
    circledComponentIdsBefore: [...state.circledComponentIds],
    topicTargetIdsBefore: [...state.topicTargetIds],
    penPointsBefore: state.penPoints.map((p) => ({ ...p })),
    recipientBefore: state.recipient,
  };
  clearTransientChoiceMessages();

  if (overrideText === undefined) {
    const switched = detectExplicitRecipientMention(text);
    if (switched) state.recipient = switched;
  }

  const spec = getCurrentSpec();
  const recipientSnapshot =
    overrideText !== undefined ? resolveRecipientForFollowUp(text) : state.recipient;
  // 挑戦が終わったあとは、店長やお客さんへの質問だけを受け付ける
  if (state.runFinished && recipientSnapshot === 'assistant') {
    pushMessage({
      role: 'system',
      tone: 'warning',
      text: 'この挑戦は終わりました。上の選択肢からやり直すと、また手直しできます。店長やお客さんには質問できます',
    });
    renderStudio();
    scrollToLatestMessage();
    return;
  }
  const hasPenAnnotation = state.circledComponentIds.length > 0 || state.penPoints.length > 0;
  const targetIdsSnapshot = hasPenAnnotation
    ? [...state.circledComponentIds]
    : [...(targetIdsOverride ?? state.topicTargetIds)];
  state.topicTargetIds = targetIdsSnapshot;
  const history = buildChatHistory();
  // 囲みを消す前に、赤ペン付きの画面を撮っておく（手書き線がフェードアウト済みの場合は囲んだ直後の撮影を使う）
  const annotatedImageBase64 = hasPenAnnotation
    ? (lastUserPenSnapshotBase64 ??
      (sandboxStageEl
        ? captureSandboxSnapshot(spec, state.liveAppState, sandboxStageEl, penCanvasEl)
        : undefined))
    : undefined;

  const userMessage = pushMessage({
    role: 'user',
    text,
    targetComponentIds: targetIdsSnapshot,
  });
  lastUserTurn = {
    ...snapshotBeforeSend,
    userMessageId: userMessage.id,
  };
  if (overrideText === undefined) {
    state.inputText = '';
  }
  state.penPoints = [];
  userPenFinishedAt = null;
  lastUserPenSnapshotBase64 = undefined;
  state.circledComponentIds = [];

  // チャットから「1つ前に戻して」「最初の画面に戻して」と指示されたときは、会話の流れを保ったまま即座に画面を戻す
  const revertCommand =
    recipientSnapshot === 'assistant' && !hasPenAnnotation
      ? detectCheckpointRevertCommand(text)
      : null;
  if (revertCommand) {
    if (state.checkpoints.length <= 1) {
      pushMessage({
        role: 'assistant',
        text: 'いまは一番最初の画面なので、これ以上前の画面はありません。気になるところを赤ペンで囲むか、直したい場所を教えてくださいね。',
      });
    } else if (revertCommand === 'reset') {
      const prevSpec = getCurrentSpec();
      state.checkpoints = [state.checkpoints[0]];
      state.activeCheckpointIndex = 0;
      state.liveAppState = preserveUnchangedLiveState(
        prevSpec,
        getCurrentSpec(),
        state.liveAppState,
      );
      state.topicTargetIds = [];
      pushMessage({ role: 'system', text: '一番最初の画面に戻しました' });
      pushMessage({
        role: 'assistant',
        text: '一番最初の画面に戻しました。どこから手直ししますか？',
      });
    } else {
      const prevSpec = getCurrentSpec();
      state.checkpoints.pop();
      state.activeCheckpointIndex = state.checkpoints.length - 1;
      state.liveAppState = preserveUnchangedLiveState(
        prevSpec,
        getCurrentSpec(),
        state.liveAppState,
      );
      state.topicTargetIds = [];
      pushMessage({ role: 'system', text: '1つ前の画面に戻しました' });
      pushMessage({
        role: 'assistant',
        text: '1つ前の画面に戻しました。ここからどう手直ししますか？',
      });
    }
    renderStudio();
    scrollToLatestMessage();
    return;
  }

  const abortController = new AbortController();
  activeUserSendAbort = abortController;
  state.chatLoading = true;
  // はっきりした改造の指示（かつ赤ペンで部品を囲んでいて、店長やお客さん宛てでないとき）なら、返事を待つ間から作業の過程をシステムメッセージで見せる
  const canApplyDirectly = recipientSnapshot === 'assistant' && targetIdsSnapshot.length > 0;
  let logMessage: TimelineMessage | undefined =
    canApplyDirectly && text && classifyUserMessageIntent(text) === 'apply'
      ? startWorkLog(targetIdsSnapshot)
      : undefined;
  const logTimers: number[] = [];
  if (logMessage) {
    const log = logMessage.workLog!;
    logTimers.push(
      window.setTimeout(() => !log.finished && setPrepWorkStep(log, '画面の部品を調べています'), 900),
      window.setTimeout(() => !log.finished && setPrepWorkStep(log, '変え方を考えています'), 2200),
    );
  }
  activeWorkLogTimers = logTimers;
  renderStudio();
  scrollToLatestMessage();

  try {
    // 発言の本文はこのブラウザに残し、サーバーには何番目の発言かだけを伝える
    const turnIndex = state.runTurnTexts.push(text) - 1;
    // お客さんへの質問の返事も、反応を見るときと同じ条件で測った座標で判定する
    const componentBounds = await measureFixedLayoutBounds(spec);
    if (abortController.signal.aborted) return;
    const data = await postJson<ChatResponse>(
      '/api/chat',
      {
        componentBounds,
        currentSpec: spec,
        latestReviews: buildLatestReviewRecord(),
        targetComponentIds: targetIdsSnapshot,
        userMessage: text,
        cohortId: state.selectedCohortId,
        runId: state.runId,
        turnIndex,
        shareForAggregate: state.runShareForAggregate,
        annotatedImageBase64,
        history,
        recipient: recipientSnapshot,
      },
      abortController.signal,
    );
    if (abortController.signal.aborted) return;

    logTimers.forEach((t) => window.clearTimeout(t));
    if (data.applied) {
      const applied = data.applied;
      // お助けエージェントが判断して改造に進んだときは、ここで作業ログを出す
      logMessage ??= startWorkLog(targetIdsSnapshot);
      if (!applied.rolledBack) {
        // 手直しが完了したら赤ペンの選択状態を解除し、次の手直しでは再び赤ペンで囲んでもらうようにする
        state.topicTargetIds = [];
        state.circledComponentIds = [];
        // 新しい画面にしてから作業を1行ずつ流し、変えた部品を光らせる
        pushNewCheckpoint(applied.commitMessage, applied.updatedSpec);
      }
      await playWorkLog(
        logMessage,
        applied.workSteps ?? [],
        Boolean(applied.rolledBack),
        abortController.signal,
      );
      if (abortController.signal.aborted) return;
      if (applied.rolledBack) {
        pushMessage({
          role: 'assistant',
          text: applied.commitMessage,
          targetComponentIds: targetIdsSnapshot,
        });
      } else {
        const hasWorkingOrder = applied.updatedSpec.components.some(
          (c) => c.type === 'button' && hasCompletionAction(c),
        );
        pushMessage({
          role: 'assistant',
          text: applied.commitMessage,
          targetComponentIds: targetIdsSnapshot,
        });
        const evalChoiceMsg: Omit<TimelineMessage, 'id'> | undefined = hasWorkingOrder
          ? {
              role: 'system',
              text: '手直しした画面をお客さんに試してもらいますか',
              targetComponentIds: targetIdsSnapshot,
              showEvaluateAction: true,
            }
          : undefined;

        if (applied.ownerFacts && applied.ownerFacts.length > 0) {
          state.chatLoading = false;
          void speakOwner(
            {
              situation: 'change',
              facts: applied.ownerFacts,
              fallback: applied.ownerReaction ?? '',
            },
            evalChoiceMsg,
            abortController.signal,
            applied.ownerComponentIds,
          );
          return;
        }
        if (evalChoiceMsg) {
          pushMessage(evalChoiceMsg);
        }
      }
    } else if (data.intent === 'owner') {
      // 店長に話しかけたときは、店長本人が返事をする
      if (logMessage) state.messages = state.messages.filter((m) => m !== logMessage);
      if (targetIdsSnapshot.length > 0) {
        state.circledComponentIds = [...targetIdsSnapshot];
        state.topicTargetIds = [...targetIdsSnapshot];
      }
      state.chatLoading = false;
      const chunks = splitIntoChatChunks(data.reply);
      enqueueMessages(
        chunks.map((chunk, i) => ({
          message: {
            role: 'owner' as const,
            text: chunk,
            pointedComponentIds: i === chunks.length - 1 ? data.pointedComponentIds : undefined,
          },
          typingMs: i === 0 ? 400 : typingDurationFor(chunk),
        })),
      );
      return;
    } else if (data.intent === 'customer') {
      // お客さんに話しかけたときは、選んだ客層の3人が順番に一言ずつ返事をする
      if (logMessage) state.messages = state.messages.filter((m) => m !== logMessage);
      if (targetIdsSnapshot.length > 0) {
        state.circledComponentIds = [...targetIdsSnapshot];
        state.topicTargetIds = [...targetIdsSnapshot];
      }
      state.chatLoading = false;
      const replies = data.customerReplies ?? [];
      const queued: QueuedMessage[] = [];
      replies.forEach((item, idx) => {
        const chunks = splitIntoChatChunks(item.text);
        for (const [chunkIdx, chunk] of chunks.entries()) {
          queued.push({
            message: {
              role: 'customer' as const,
              personaId: item.personaId,
              text: chunk,
              pointedComponentIds:
                chunkIdx === chunks.length - 1
                  ? (item.pointedComponentIds ?? data.pointedComponentIds)
                  : undefined,
            },
            typingMs: idx === 0 && chunkIdx === 0 ? 400 : typingDurationFor(chunk),
          });
        }
      });
      enqueueMessages(queued);
      return;
    } else {
      // 改造だと思って作業ログを出したが、質問や聞き返しだったときは消す
      if (logMessage) state.messages = state.messages.filter((m) => m !== logMessage);
      if (targetIdsSnapshot.length > 0) {
        state.circledComponentIds = [...targetIdsSnapshot];
        state.topicTargetIds = [...targetIdsSnapshot];
      }
      const replyChunks = splitIntoChatChunks(data.reply);
      const [firstChunk, ...restChunks] = replyChunks;
      pushMessage({
        role: 'assistant',
        text: firstChunk,
        targetComponentIds: targetIdsSnapshot,
        pointedComponentIds: restChunks.length === 0 ? data.pointedComponentIds : undefined,
      });
      if (restChunks.length > 0) {
        state.chatLoading = false;
        const queued: QueuedMessage[] = restChunks.map((chunk, idx) => ({
          message: {
            role: 'assistant' as const,
            text: chunk,
            targetComponentIds: targetIdsSnapshot,
            pointedComponentIds:
              idx === restChunks.length - 1 ? data.pointedComponentIds : undefined,
          },
          typingMs: typingDurationFor(chunk),
        }));
        enqueueMessages(queued);
        return;
      }
    }
  } catch (err) {
    logTimers.forEach((t) => window.clearTimeout(t));
    if (abortController.signal.aborted) return;
    if (logMessage?.workLog && !logMessage.workLog.finished) {
      state.messages = state.messages.filter((m) => m !== logMessage);
    }
    if ((err as { code?: string }).code === 'input_rejected') {
      // 送らなかった発言は、会話の履歴から外して以後 Gemini に渡さない。直せるように入力欄へ戻す
      lastUserTurn = null;
      state.messages = state.messages.filter((m) => m !== userMessage);
      const sentIndex = state.runTurnTexts.lastIndexOf(text);
      if (sentIndex >= 0) state.runTurnTexts[sentIndex] = '';
      if (!state.inputText.trim()) state.inputText = text;
    }
    pushMessage({ role: 'system', tone: 'warning', text: describeError(err) });
  } finally {
    if (!abortController.signal.aborted) {
      if (activeUserSendAbort === abortController) {
        activeUserSendAbort = null;
      }
      activeWorkLogTimers = [];
      if (!state.speakerPointed) {
        state.justChangedIds = [];
      }
      state.chatLoading = false;
      renderStudio();
      scrollToLatestMessage();
      sendPendingIfReady();
    }
  }
}

/**
 * 店長に話してもらう。せりふはAIが作るので、届くまで店長の書き込み中の表示を出しておく。
 * 届かなかったときは決まったせりふを使う
 */
async function speakOwner(
  req: Omit<OwnerLineRequest, 'history'>,
  afterMessage?: Omit<TimelineMessage, 'id'>,
  signal?: AbortSignal,
  pointedComponentIds?: string[],
): Promise<void> {
  if (signal?.aborted) return;
  state.typingMessage = { role: 'owner' };
  renderStudio();
  scrollToLatestMessage();
  let text = req.fallback;
  try {
    const data = await postJson<OwnerLineResponse>(
      '/api/owner-line',
      {
        ...req,
        history: buildOwnerHistory(),
      },
      signal,
    );
    text = data.text || req.fallback;
  } catch {
    // 決まったせりふを使う
  }
  if (signal?.aborted) return;
  if (state.typingMessage?.role === 'owner') state.typingMessage = undefined;
  const chunks = text ? splitIntoChatChunks(text) : [];
  const queued: QueuedMessage[] = chunks.map((chunk, i) => ({
    message: {
      role: 'owner' as const,
      text: chunk,
      pointedComponentIds: i === chunks.length - 1 ? pointedComponentIds : undefined,
    },
    typingMs: i === 0 ? 300 : typingDurationFor(chunk),
  }));
  if (afterMessage) {
    queued.push({ message: afterMessage, typingMs: 0 });
  }
  if (queued.length === 0) {
    renderStudio();
    return;
  }
  // 1つ目はすでに書き込み中を見せて待たせたので、すぐ出す
  enqueueMessages(queued);
}

/** 「お客さんの反応を見る」を押したとき、まず1人目のお客さんがどう思うかを予想する選択肢をチャットに出す */
function handleRequestEvaluate(): void {
  if (state.evaluateLoading || !state.selectedCohortId || state.runFinished) return;
  lastUserTurn = null;
  flushMessageQueue();
  const existing = [...state.messages].reverse().find((m) => m.predictionPrompt);
  if (existing && state.messages[state.messages.length - 1] === existing) {
    scrollToLatestMessage();
    return;
  }
  clearTransientChoiceMessages();
  clearPredictionStepMessages();
  const personas = getReviewersByCohort(state.selectedCohortId);
  const firstPersona = personas[0];
  if (!firstPersona) {
    void handleEvaluate();
    return;
  }
  pushMessage({
    role: 'system',
    // 予想を選ぶまで結果が出ないので、最初の1人のときに、予想の役割を添える
    text: `\`${firstPersona.name}\` はどう思うか予想しよう 予想はクリアに関係しません。当たった数が振り返りに残ります`,
    predictionPrompt: { draft: {} },
  });
  renderStudio();
  scrollToLatestMessage();
}

async function handleEvaluate(predictions?: Record<string, ReviewOutcome>): Promise<void> {
  if (state.evaluateLoading || !state.selectedCohortId) return;
  lastUserTurn = null;
  flushMessageQueue();
  clearTransientChoiceMessages();
  if (!predictions) {
    clearPredictionStepMessages();
  }
  state.evaluateLoading = true;
  renderStudio();
  scrollToLatestMessage();

  try {
    const activePersonas = getReviewersByCohort(state.selectedCohortId);
    const currentSpec = getCurrentSpec();
    // 判定用の座標は、見えている画面ではなく、決まった大きさで画面外に描画した画面から測る
    const componentBounds = await measureFixedLayoutBounds(currentSpec);
    const screenshotBase64 = sandboxStageEl
      ? captureSandboxSnapshot(currentSpec, state.liveAppState, sandboxStageEl, null)
      : undefined;

    // 前に反応を見たときの記録。画面を直して満足が増えず、気にしている部品が前と変わったときだけ次の困りごとが見えてきたと伝える
    const previousFound = findLatestCohortEvaluation();
    const changedSinceLastReview = previousFound
      ? previousFound.index !== state.activeCheckpointIndex
      : false;
    const evaluation = await postJson<EvaluationResponse>('/api/evaluate', {
      currentSpec,
      cohortId: state.selectedCohortId,
      runId: state.runId,
      turnCount: state.runTurnTexts.length,
      screenshotBase64,
      componentBounds,
      history: buildOwnerHistory(),
    });
    // 満足の欄は、お客さんが言い終えるまで空けておく
    const revealed = new Set<string>();
    revealedReviewerIds.set(evaluation, revealed);
    getCurrentCheckpoint().evaluation = evaluation;
    state.predictionRounds.push(scorePredictionRound(evaluation.results, predictions));
    // 選ばなかった客層も、サーバーと同じ独自の部品の読み取り結果と座標で判定する
    const simInput = prepareSimulationInput(
      currentSpec,
      evaluation.customComponentProfiles,
      componentBounds,
    );
    // 各お客さんの吹き出しの下に予想の答え合わせバッジが出るため、途中の予想確認行はここで片付ける
    clearPredictionStepMessages();

    // お客さん3人の反応を書き込み中の表示を挟んで1人ずつ出す（店長は変更直後に反応済みのためここでは話さない）
    const queued: QueuedMessage[] = [];
    if ((evaluation.customComponentFallbackIds ?? []).length > 0) {
      queued.push({
        message: {
          role: 'system',
          tone: 'warning',
          text: '独自の部品を読み取れなかったので、今回は部品の中のボタンの動きと文字だけで判定しました。もう一度反応を見ると読み取り直します',
        },
        typingMs: 0,
      });
    }
    let achievedCount = 0;
    for (const persona of activePersonas) {
      const result = evaluation.results.find((r) => r.reviewerId === persona.id);
      if (!result) continue;
      if (result.outcome === 'achieved') achievedCount++;
      queued.push({
        message: {
          role: 'customer',
          personaId: persona.id,
          text: result.comment,
          pointedComponentIds: result.frictionComponentIds,
          review: {
            outcome: result.outcome,
            predictedOutcome: predictions?.[persona.id],
          },
        },
        typingMs: typingDurationFor(result.comment),
        onFinished: () => revealed.add(persona.id),
      });
    }
    // 選んだ3人のうち誰かが買えるようになったら、選ばなかった客層への影響も見せる。
    // 誰のために直したかで別の誰かが困ることに、その場で気づけるようにする
    const otherPersonas = ALL_REVIEWER_PERSONAS.filter(
      (p) => !activePersonas.some((a) => a.id === p.id),
    );
    if (
      shouldShowNextFrictionNotice({
        changedSinceLastReview,
        previousResults: previousFound?.evaluation.results,
        currentResults: evaluation.results,
        activeReviewerIds: activePersonas.map((p) => p.id),
      })
    ) {
      queued.push({
        message: {
          role: 'system',
          text: '直すと、隠れていた次の困りごとが見えてきます。お客さんの声を手がかりに続けましょう',
        },
        typingMs: 0,
      });
    }
    // クリアしたときは、あとのレポートで1人ずつ詳しく出す
    if (achievedCount > 0 && achievedCount < activePersonas.length) {
      queued.push({
        message: {
          role: 'system',
          text: describeOtherCohortImpact(simInput.spec, otherPersonas, simInput.bounds).line,
        },
        typingMs: 0,
      });
    }
    if (achievedCount === activePersonas.length) {
      queued.push({
        message: {
          role: 'system',
          text: '🎉 選んだお客さん3人全員が、気持ちよく注文できました！',
        },
        typingMs: 0,
      });
      for (const ownerLine of evaluation.owner?.closingLines ??
        buildOwnerClosingLines(OWNER, evaluation.owner)) {
        queued.push({
          message: { role: 'owner', text: ownerLine },
          typingMs: typingDurationFor(ownerLine),
        });
      }
      if (state.selectedCohortId && !state.clearedCohortIds.includes(state.selectedCohortId)) {
        state.clearedCohortIds.push(state.selectedCohortId);
      }
      // 1回の挑戦は、選んだ客層の3人が全員達成したところで終わる。やり直すまで手直しは受け付けない
      state.runFinished = true;
      sendRunEvent('end', 'cleared', {
        achievedCount: achievedCount,
        totalCustomers: activePersonas.length,
      });
      const remaining = TARGET_COHORTS.filter((c) => !state.clearedCohortIds.includes(c.id));
      const assistantFeedback = evaluation.telemetrySummary?.assistantFeedbackLines ?? [];
      const allClearedLine = 'おつかれさまでした！これで3つの客層すべてをクリアしました';
      const assistantLines =
        assistantFeedback.length > 0
          ? [...(remaining.length === 0 ? [allClearedLine] : []), ...assistantFeedback]
          : remaining.length > 0
            ? ['おつかれさまでした！今回の挑戦のレポートです']
            : [allClearedLine, '今回の挑戦のレポートです'];
      for (const rawLine of assistantLines) {
        // 指示の引用は目印で届くので、このブラウザにある本文に置き換える
        const filled = fillTurnRefs(rawLine, state.runTurnTexts);
        queued.push({
          message: { role: 'assistant', text: filled },
          typingMs: typingDurationFor(filled),
        });
      }
      // お助けAIが2通目の吹き出しで指示の引用を伝えるため、システム行は数値の記録行と、
      // 目的を伝えたと数えた指示の行だけ出す
      const rawSystemLines = evaluation.telemetrySummary?.systemLogLines ?? [];
      const systemLinesToShow =
        assistantFeedback.length > 0 ? rawSystemLines.slice(0, 2) : rawSystemLines;
      for (const telemetryLine of systemLinesToShow) {
        queued.push({
          message: { role: 'system', text: fillTurnRefs(telemetryLine, state.runTurnTexts) },
          typingMs: 0,
        });
      }
      const predictionLine = formatPredictionReportLine(state.predictionRounds);
      if (predictionLine) {
        queued.push({
          message: { role: 'system', text: predictionLine },
          typingMs: 0,
        });
      }
      const report = describeOtherCohortReport(
        simInput.spec,
        otherPersonas,
        simInput.bounds,
        TARGET_COHORTS,
      );
      queued.push({
        message: {
          role: 'system',
          text: `ほかの客層のお客さんは、この画面をこう思いそうです 満足しそうなのは \`${report.total}人中${report.achieved}人\``,
        },
        typingMs: 0,
      });
      // 客層ごとに見出しを付け、次にどの客層を選べばその人のために直せるかが分かるようにする
      for (const group of report.groups) {
        queued.push({
          message: {
            role: 'system',
            text: `\`${group.label}\` の客層 満足しそうなのは \`${group.total}人中${group.achieved}人\``,
          },
          typingMs: 0,
        });
        for (const line of group.lines) {
          queued.push({ message: { role: 'system', text: line }, typingMs: 0 });
        }
      }
      queued.push({
        message: {
          role: 'system',
          text: '次はどうしますか どちらも画面を最初に戻してやり直します',
          showNextChallenge: true,
        },
        typingMs: 0,
      });
    } else {
      for (const line of [
        `3人中${achievedCount}人が満足して買えました`,
        '気になるお客さんに話を聞くか、直したい場所を指示してください',
      ]) {
        queued.push({
          message: { role: 'assistant', text: line },
          typingMs: typingDurationFor(line),
        });
      }
    }
    state.evaluateLoading = false;
    enqueueMessages(queued);
  } catch (err) {
    clearPredictionStepMessages();
    pushMessage({ role: 'system', tone: 'warning', text: describeError(err) });
  } finally {
    state.evaluateLoading = false;
    renderStudio();
    scrollToLatestMessage();
  }
}

/**
 * 誰のために直すかを選ぶ前の案内。紹介の会話が流れている間は、まだ選ぶボタンが出ていないので待つように伝え、
 * ボタンが出たら、チャットの中のボタンから選ぶように伝える
 */
function cohortSelectionGuidance(): string {
  const selectorShown = state.messages.some((m) => m.showCohortSelector);
  return selectorShown
    ? 'チャットのボタンから、誰のために直すかを選んでください'
    : '店長の話のあとで、誰のために直すかを選びます';
}

/** 「対象 未選択」を押したとき、チャットの選ぶボタンまでスクロールして光らせる。まだ出ていなければ会話の最後へ */
function focusCohortSelector(): void {
  const rows = chatScrollEl?.querySelectorAll<HTMLElement>('.cohort-pill-row');
  const row = rows && rows.length > 0 ? rows[rows.length - 1] : null;
  if (!row) {
    scrollToLatestMessage();
    return;
  }
  row.scrollIntoView({ behavior: 'smooth', block: 'center' });
  row.classList.remove('is-attention');
  void row.offsetWidth;
  row.classList.add('is-attention');
}

function renderCohortSelector(msg: TimelineMessage, isLatestSelector: boolean): HTMLElement {
  const pillRow = document.createElement('div');
  pillRow.className = 'cohort-pill-row';
  for (const c of TARGET_COHORTS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    const isSelected = msg.chosenCohortId === c.id;
    btn.className = `cohort-pill-btn ${isSelected ? 'is-selected' : ''}`;
    // やり直したあとは、一番新しい選択肢だけを押せるようにする
    btn.disabled = state.selectedCohortId !== undefined || !isLatestSelector;
    btn.textContent = state.clearedCohortIds.includes(c.id) ? `${c.label} クリア済み` : c.label;
    btn.addEventListener('click', () => handleSelectCohort(c.id));
    pillRow.appendChild(btn);
  }
  return pillRow;
}

/** クリア後の「ほかのお客さんにもチャレンジしますか」の選択肢（システムメッセージ表示） */
function renderNextChallengeSelector(msg: TimelineMessage, isLatestChallenge: boolean): HTMLElement {
  const pillRow = document.createElement('div');
  pillRow.className = 'cohort-pill-row';
  const currentCohortId = state.selectedCohortId;
  const options: Array<{ key: string; label: string; onSelect: () => void }> = [
    {
      key: 'same',
      label: '同じ客層でやり直す',
      onSelect: () => handleRestartFromScratch(currentCohortId),
    },
    {
      key: 'other',
      label: '別の客層でやり直す',
      onSelect: () => handleRestartFromScratch(),
    },
  ];
  for (const opt of options) {
    const btn = document.createElement('button');
    btn.type = 'button';
    const isSelected = msg.chosenOption === opt.key;
    btn.className = `cohort-pill-btn ${isSelected ? 'is-selected' : ''}`;
    btn.disabled = Boolean(msg.chosenOption) || !isLatestChallenge;
    btn.textContent = opt.label;
    btn.addEventListener('click', () => {
      msg.chosenOption = opt.key;
      opt.onSelect();
    });
    pillRow.appendChild(btn);
  }
  return pillRow;
}

const PREDICTION_NEXT_DELAY_MS = 480;

/** お客さんの反応を見る前に、いま順番が来ている1人の予想選択肢ピルを描画する */
function renderPredictionPrompt(msg: TimelineMessage): HTMLElement | null {
  const draft = msg.predictionPrompt!.draft;
  const personas = getReviewersByCohort(state.selectedCohortId);
  const personaIndex = personas.findIndex((p) => !draft[p.id]);
  if (personaIndex < 0) return null;
  const currentPersona = personas[personaIndex];

  const pillRow = document.createElement('div');
  pillRow.className = 'cohort-pill-row';
  for (const opt of PREDICTION_OPTIONS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.disabled = state.evaluateLoading;
    btn.className = `cohort-pill-btn is-prediction-opt is-${opt.outcome}`;
    btn.textContent = opt.label;
    btn.addEventListener('click', () => {
      const nextDraft: Record<string, ReviewOutcome> = {
        ...draft,
        [currentPersona.id]: opt.outcome,
      };
      // 選んだ瞬間に今の選択肢ボタンを消し、選んだ予想の1行だけを残す
      state.messages = state.messages.filter((m) => !m.predictionPrompt);
      pushMessage({
        role: 'system',
        text: `\`${currentPersona.name}\` の予想：\`${outcomePredictionLabel(opt.outcome)}\``,
        predictionStepDone: {
          personaIndex,
          draftBefore: { ...draft },
        },
      });
      renderStudio();
      scrollToLatestMessage();

      const nextPersona = personas.find((p) => !nextDraft[p.id]);
      if (nextPersona) {
        // 少し間隔を空けてから、次のお客さんの予想と選択肢を出す
        enqueueMessages([
          {
            message: {
              role: 'system',
              text: `\`${nextPersona.name}\` はどう思うか予想しよう`,
              predictionPrompt: { draft: nextDraft },
            },
            typingMs: PREDICTION_NEXT_DELAY_MS,
          },
        ]);
      } else {
        void handleEvaluate(nextDraft);
      }
    });
    pillRow.appendChild(btn);
  }

  const skipBtn = document.createElement('button');
  skipBtn.type = 'button';
  skipBtn.className = 'cohort-pill-btn is-prediction-skip';
  skipBtn.disabled = state.evaluateLoading;
  skipBtn.textContent = '予想せずすぐ見る';
  skipBtn.addEventListener('click', () => {
    void handleEvaluate();
  });
  pillRow.appendChild(skipBtn);

  return pillRow;
}

/** お助けAIが出す選択肢（返信候補・お客さんの反応確認・予想パネル）をシステムメッセージのピルボタンとして描画する */
function renderActionChoiceRow(msg: TimelineMessage): HTMLElement | null {
  if (msg.predictionPrompt && state.selectedCohortId) {
    return renderPredictionPrompt(msg);
  }
  const pillRow = document.createElement('div');
  pillRow.className = 'cohort-pill-row';
  if (msg.showEvaluateAction && !evaluationFor(getCurrentCheckpoint()) && state.selectedCohortId) {
    const evalBtn = document.createElement('button');
    evalBtn.type = 'button';
    evalBtn.className = 'cohort-pill-btn';
    evalBtn.textContent = 'お客さんの反応を見る';
    evalBtn.addEventListener('click', () => {
      handleRequestEvaluate();
    });
    pillRow.appendChild(evalBtn);

    const skipEvalBtn = document.createElement('button');
    skipEvalBtn.type = 'button';
    skipEvalBtn.className = 'cohort-pill-btn';
    skipEvalBtn.textContent = '続けて手直しする';
    skipEvalBtn.addEventListener('click', () => {
      clearTransientChoiceMessages();
      renderStudio();
      focusComposerInput();
    });
    pillRow.appendChild(skipEvalBtn);
  }
  const followUps = msg.followUps ?? [];
  for (const followUp of followUps) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'cohort-pill-btn';
    btn.textContent = followUp;
    btn.addEventListener('click', () => {
      void handleSendMessage(followUp, msg.targetComponentIds);
    });
    pillRow.appendChild(btn);
  }
  if (followUps.length > 0) {
    const noopBtn = document.createElement('button');
    noopBtn.type = 'button';
    noopBtn.className = 'cohort-pill-btn';
    noopBtn.textContent = '何もしない';
    noopBtn.addEventListener('click', () => {
      if (msg.targetComponentIds && msg.targetComponentIds.length > 0) {
        void recordDecision({
          stage: 'apply_change',
          targetId: msg.targetComponentIds.join(','),
          chosenLabel: '何もしない',
          reason: '',
        });
      }
      state.penPoints = [];
      state.circledComponentIds = [];
      state.topicTargetIds = [];
      clearTransientChoiceMessages();
      renderStudio();
    });
    pillRow.appendChild(noopBtn);
  }
  return pillRow.childNodes.length > 0 ? pillRow : null;
}

/**
 * システムメッセージ行のテキストを描画する。
 * バッククォート `...` で囲まれた固有名詞・変更前後の値はインラインコード要素（.chat-system-token）にする。
 */
function appendFormattedSystemText(container: HTMLElement, text: string): void {
  const parts = text.split(/`([^`]+)`/g);
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i];
    if (!part) continue;
    if (i % 2 === 1) {
      const code = document.createElement('code');
      code.className = 'chat-system-token';
      code.textContent = part;
      container.appendChild(code);
    } else {
      container.appendChild(document.createTextNode(part));
    }
  }
}

/** 改造の進捗・変更内容を、ボックスに入れず通常のシステムメッセージ行と同じ形式で描画する */
function appendWorkLogLines(row: HTMLElement, msg: TimelineMessage): void {
  const log = msg.workLog!;
  for (const step of log.steps) {
    const line = document.createElement('div');
    if (step.done) {
      const componentId = step.componentId;
      line.className = `chat-system-line${componentId ? ' is-interactive is-selectable' : ''}`;
      appendFormattedSystemText(line, step.text);
      if (componentId) {
        line.title = 'クリックするとスマホ画面の該当箇所を光らせます';
        line.addEventListener('click', (e) => {
          e.stopPropagation();
          highlightSandboxComponent(componentId);
        });
      }
    } else {
      line.className = 'chat-system-line is-progress';
      const dots = document.createElement('span');
      dots.className = 'typing-dots';
      dots.setAttribute('aria-hidden', 'true');
      dots.innerHTML = '<span></span><span></span><span></span>';
      const text = document.createElement('span');
      appendFormattedSystemText(text, step.text);
      line.append(dots, text);
    }
    row.appendChild(line);
  }
}

const AGENT_NAME = 'お助けAI';
const AGENT_ICON = '助';

/** 相手は入力欄の上の「〜に話しかける」で分かるので、ここでは何を書くかだけを示す。例は出さない */
function getRecipientPlaceholder(recipient: string): string {
  if (recipient === 'owner') {
    return `${OWNER.name}への質問や相談を入力`;
  }
  if (recipient === 'customer') {
    return 'お客さんへの質問を入力';
  }
  // 囲まないと手直しは受け付けないので、まだ囲んでいないときは先に囲むことを伝える
  return state.circledComponentIds.length > 0
    ? '手直しの指示や質問を入力'
    : '手直しは部品を赤ペンで囲んでから入力';
}

const COMPOSER_TEXTAREA_MIN_HEIGHT_PX = 40;
const COMPOSER_TEXTAREA_MAX_HEIGHT_PX = 112;
let composerInitialMinHeightPx = COMPOSER_TEXTAREA_MIN_HEIGHT_PX;

function syncComposerTextareaHeight(el: HTMLTextAreaElement): void {
  el.style.minHeight = `${COMPOSER_TEXTAREA_MIN_HEIGHT_PX}px`;
  el.style.height = 'auto';
  const measured = el.scrollHeight;
  if (!el.value && measured > composerInitialMinHeightPx) {
    composerInitialMinHeightPx = Math.min(measured, COMPOSER_TEXTAREA_MAX_HEIGHT_PX);
  }
  const minHeight = Math.max(COMPOSER_TEXTAREA_MIN_HEIGHT_PX, composerInitialMinHeightPx);
  const nextHeight = Math.min(Math.max(measured, minHeight), COMPOSER_TEXTAREA_MAX_HEIGHT_PX);
  el.style.minHeight = `${minHeight}px`;
  el.style.height = `${nextHeight}px`;
  el.style.overflowY = measured > COMPOSER_TEXTAREA_MAX_HEIGHT_PX ? 'auto' : 'hidden';

  const outerMinHeightPx = `${minHeight + 2}px`;
  const wrapEl = el.closest<HTMLElement>('.composer-input-wrap');
  if (wrapEl) {
    wrapEl.style.minHeight = outerMinHeightPx;
  }
  const sendBtnEl = el
    .closest<HTMLElement>('.composer-form-row')
    ?.querySelector<HTMLElement>('.btn-composer-send');
  if (sendBtnEl) {
    sendBtnEl.style.minHeight = outerMinHeightPx;
  }
}

function focusComposerInput(moveCursorToEnd = false): void {
  window.requestAnimationFrame(() => {
    const inputEl = shellEl?.querySelector<HTMLTextAreaElement>('.composer-input');
    if (!inputEl) return;
    syncComposerTextareaHeight(inputEl);
    inputEl.focus();
    if (moveCursorToEnd) {
      const len = inputEl.value.length;
      inputEl.setSelectionRange(len, len);
    }
  });
}

function selectRecipientAndFocus(recipient: string): void {
  if (!state.selectedCohortId) return;
  state.recipient = recipient;
  renderStudio();
  focusComposerInput();
}

function createBubble(className: string, text: string): HTMLElement {
  const bubble = document.createElement('div');
  bubble.className = className;
  const body = document.createElement('div');
  body.className = 'bubble-text';
  appendFormattedSystemText(body, text);
  bubble.appendChild(body);
  return bubble;
}

function createTypingBubble(className: string, label: string): HTMLElement {
  const bubble = document.createElement('div');
  bubble.className = className;
  bubble.setAttribute('aria-label', label);
  bubble.innerHTML =
    '<span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span>';
  return bubble;
}

/**
 * 発言者が気にしているスマホ上の部品を、システムメッセージ行として描画する。
 * 例: `時間のないお客さん` は `クーポン広告のポップアップ` を気にしています
 * 部品名のトークンをクリックすると、スマホ画面のその部品を表示し、発言者のペン色で丸く囲む。
 */
function createPointedSystemRow(
  speakerName: string,
  componentIds: string[] | undefined,
  role: 'customer' | 'assistant' | 'owner',
  isEntering: boolean,
): HTMLElement | null {
  const comp = findComponentsByIds(componentIds)[0];
  if (!comp) return null;

  const row = document.createElement('div');
  row.className = `chat-system-row${isEntering ? ' is-entering' : ''}`;

  const line = document.createElement('div');
  line.className = 'chat-system-line';

  const speakerToken = document.createElement('code');
  speakerToken.className = 'chat-system-token';
  speakerToken.textContent = speakerName;

  line.append(
    speakerToken,
    document.createTextNode(' は '),
    createComponentLinkToken(comp, role),
    document.createTextNode(' を気にしています'),
  );
  row.appendChild(line);
  return row;
}

/** 部品IDから、今の画面にある部品を重ならないように取り出す */
function findComponentsByIds(componentIds: string[] | undefined): AppSpec['components'] {
  if (!componentIds || componentIds.length === 0) return [];
  const spec = getCurrentSpec();
  return Array.from(new Set(componentIds))
    .map((id) => spec.components.find((c) => c.id === id))
    .filter((c): c is AppSpec['components'][number] => Boolean(c));
}

/** 部品の役割の名前を出し、押すとスマホ画面のその部品をペンで囲むトークン */
function createComponentLinkToken(
  comp: AppSpec['components'][number],
  role: 'customer' | 'assistant' | 'owner',
): HTMLButtonElement {
  const compBtn = document.createElement('button');
  compBtn.type = 'button';
  compBtn.className = 'chat-system-token is-comp-link is-selectable';
  compBtn.textContent = describeComponentFunctionName(comp);
  compBtn.title = 'クリックするとスマホ画面のこの部品を表示してペンで囲みます（手直しするときは赤ペンで囲んでください）';
  compBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    highlightSandboxComponent([comp.id], role);
  });
  return compBtn;
}

/**
 * 赤ペンで囲んだ部品を、システムメッセージ行として描画する。
 * 例: `クーポン広告のポップアップ` を囲みました
 * 画面の文字（「🎉会員限定🎉500円OFF!!」など）は出さず、読み上げもしない
 */
function createCircledSystemRow(componentIds: string[], isEntering: boolean): HTMLElement | null {
  const comps = findComponentsByIds(componentIds).slice(0, 3);
  if (comps.length === 0) return null;
  const row = document.createElement('div');
  row.className = `chat-system-row${isEntering ? ' is-entering' : ''}`;
  const line = document.createElement('div');
  line.className = 'chat-system-line';
  comps.forEach((comp, i) => {
    if (i > 0) line.append(document.createTextNode(' と '));
    line.append(createComponentLinkToken(comp, 'assistant'));
  });
  line.append(document.createTextNode(' を囲みました'));
  row.appendChild(line);
  return row;
}

/**
 * アイコンと名前つきの発言行（お客さん・お助けエージェント・店長共通）
 * アイコンや名前をクリックすると、その人をメッセージの宛先に選べる
 */
function createSpeakerRow(
  kind: 'customer' | 'agent' | 'owner',
  icon: string,
  name: string,
  bubble: HTMLElement,
  extras: HTMLElement[] = [],
  isContinuation = false,
  recipientKey?: string,
): HTMLElement {
  const row = document.createElement('div');
  row.className = `chat-bubble-row is-speaker is-${kind}${isContinuation ? ' is-continuation' : ''}`;

  const avatar = document.createElement('span');
  avatar.className = `speaker-avatar is-${kind}${isContinuation ? ' is-continuation' : ''}${recipientKey ? ' is-selectable' : ''}`;
  avatar.textContent = icon;
  if (recipientKey) {
    avatar.title = `${name}に話しかける`;
    avatar.addEventListener('click', () => selectRecipientAndFocus(recipientKey));
  }

  const body = document.createElement('div');
  body.className = 'speaker-body';
  if (!isContinuation) {
    const nameEl = document.createElement('div');
    nameEl.className = `speaker-name${recipientKey ? ' is-selectable' : ''}`;
    nameEl.textContent = name;
    if (recipientKey) {
      nameEl.title = `${name}に話しかける`;
      nameEl.addEventListener('click', () => selectRecipientAndFocus(recipientKey));
    }
    body.appendChild(nameEl);
  }

  body.append(bubble, ...extras);
  row.append(avatar, body);
  return row;
}

function speakerKeyOf(msg: Pick<TimelineMessage, 'role' | 'personaId'>): string {
  return msg.role === 'customer' ? `customer:${msg.personaId ?? ''}` : msg.role;
}

// 入力欄は一度だけ作って使い回す。描き直しのたびに作り直すと、日本語の変換中の文字が消えてしまう
let composerInputEl: HTMLTextAreaElement | null = null;
let composerIsComposing = false;
let onComposerKeydown: ((e: KeyboardEvent) => void) | null = null;
let onComposerInput: (() => void) | null = null;

function getComposerInput(): HTMLTextAreaElement {
  if (composerInputEl) return composerInputEl;
  const el = document.createElement('textarea');
  el.rows = 1;
  el.className = 'composer-input';
  el.addEventListener('compositionstart', () => {
    composerIsComposing = true;
  });
  el.addEventListener('compositionend', () => {
    composerIsComposing = false;
  });
  // 描き直すたびに最新の状態を見る処理に差し替えるので、ここでは呼び出すだけにする
  el.addEventListener('keydown', (e) => onComposerKeydown?.(e));
  el.addEventListener('input', () => onComposerInput?.());
  composerInputEl = el;
  return el;
}

function syncElementAttributes(target: Element, source: Element): void {
  for (const attr of [...target.attributes]) {
    if (!source.hasAttribute(attr.name)) target.removeAttribute(attr.name);
  }
  for (const attr of [...source.attributes]) {
    if (target.getAttribute(attr.name) !== attr.value) target.setAttribute(attr.name, attr.value);
  }
}

// 新しいチャット欄を画面に出す。入力欄が画面にあるときは、入力欄とその親を画面から外さずに、周りだけを新しいものに入れ替える
function mountChatPanel(slot: HTMLElement, panel: HTMLElement): void {
  const input = composerInputEl;
  const marker = panel.querySelector<HTMLElement>('[data-composer-input-slot]');
  if (!input || !marker) {
    slot.replaceChildren(panel);
    return;
  }
  const oldChain: Element[] = [input];
  const newChain: Element[] = [marker];
  if (input.isConnected && slot.contains(input) && slot.childNodes.length === 1) {
    while (oldChain[oldChain.length - 1].parentElement !== slot) {
      oldChain.push(oldChain[oldChain.length - 1].parentElement as Element);
    }
    while (newChain[newChain.length - 1] !== panel) {
      newChain.push(newChain[newChain.length - 1].parentElement as Element);
    }
  }
  const canGraft =
    oldChain.length > 1 &&
    oldChain.length === newChain.length &&
    oldChain.slice(1).every((el, i) => el.tagName === newChain[i + 1].tagName);
  if (!canGraft) {
    marker.replaceWith(input);
    slot.replaceChildren(panel);
    return;
  }
  for (let i = oldChain.length - 1; i >= 1; i -= 1) {
    const oldEl = oldChain[i];
    const newEl = newChain[i];
    const keepOld = oldChain[i - 1];
    const keepNew = newChain[i - 1];
    syncElementAttributes(oldEl, newEl);
    for (const child of [...oldEl.childNodes]) {
      if (child !== keepOld) child.remove();
    }
    const kids = [...newEl.childNodes];
    const idx = kids.indexOf(keepNew);
    for (const child of kids.slice(0, idx)) oldEl.insertBefore(child, keepOld);
    for (const child of kids.slice(idx + 1)) oldEl.appendChild(child);
  }
}

function renderChatPanel(): HTMLElement {
  const chatCard = document.createElement('section');
  chatCard.className = 'vibe-chat-card';

  const stream = document.createElement('div');
  const isPlayingQueue =
    isQueueRunning || queueTimers.size > 0 || activeQueuedItem !== null || messageQueue.length > 0;
  stream.className = `vibe-chat-stream ${isPlayingQueue ? 'is-skippable' : ''}`;
  if (isPlayingQueue) stream.title = 'クリックすると次の発言をすぐ表示します';
  // 順番に出している途中の発言は、チャットを押せば1つずつすぐ表示して次に進める（ボタンや入力欄を押したときは除く）
  stream.addEventListener('click', (e) => {
    if ((e.target as HTMLElement).closest('button, input, textarea, a, select, .is-selectable')) return;
    if (!isPlayingQueue) return;
    advanceOneQueuedMessage();
  });
  chatScrollEl = stream;

  const isBusy = state.chatLoading || state.evaluateLoading;
  const latestSelectorMessageId = [...state.messages].reverse().find((m) => m.showCohortSelector)?.id;
  const latestChallengeMessageId = [...state.messages].reverse().find((m) => m.showNextChallenge)?.id;
  const latestActionChoiceId = [...state.messages]
    .reverse()
    .find((m) => isTransientChoiceMessage(m))?.id;
  let prevSpeakerKey = '';

  for (const msg of state.messages) {
    // 進行役のアナウンス・AIの作業プログレス・選択肢：吹き出しなしで中央に小さく出す
    if (msg.role === 'system') {
      const isActionChoiceMsg = isTransientChoiceMessage(msg);
      if (isActionChoiceMsg) {
        if (msg.id !== latestActionChoiceId || isBusy || isPlayingQueue) continue;
        const actionRow = renderActionChoiceRow(msg);
        if (!actionRow) continue;
        const row = document.createElement('div');
        if (msg.text) {
          const line = document.createElement('div');
          line.className = 'chat-system-line';
          appendFormattedSystemText(line, msg.text);
          row.appendChild(line);
        }
        row.appendChild(actionRow);
        const isFirstSeen = !seenMessageIds.has(msg.id);
        seenMessageIds.add(msg.id);
        row.className = `chat-system-row${isFirstSeen ? ' is-entering' : ''}`;
        prevSpeakerKey = 'system';
        stream.appendChild(row);
        continue;
      }

      if (msg.circledNoticeIds) {
        const isFirstSeen = !seenMessageIds.has(msg.id);
        seenMessageIds.add(msg.id);
        const circledRow = createCircledSystemRow(msg.circledNoticeIds, isFirstSeen);
        if (circledRow) {
          stream.appendChild(circledRow);
          prevSpeakerKey = 'system';
        }
        continue;
      }

      const row = document.createElement('div');
      if (msg.workLog) {
        appendWorkLogLines(row, msg);
      } else if (msg.text) {
        const line = document.createElement('div');
        const canRewindPrediction =
          Boolean(msg.predictionStepDone) &&
          !isBusy &&
          (state.messages.some((m) => Boolean(m.predictionPrompt)) ||
            messageQueue.some((q) => Boolean(q.message.predictionPrompt)) ||
            Boolean(activeQueuedItem?.message.predictionPrompt));
        line.className = `chat-system-line ${msg.tone === 'warning' ? 'is-warning' : ''}${canRewindPrediction ? ' is-interactive is-selectable' : ''}`;
        appendFormattedSystemText(line, msg.text);
        if (canRewindPrediction && msg.predictionStepDone) {
          const stepInfo = msg.predictionStepDone;
          line.title = 'クリックするとこのお客さんの予想から選び直せます';
          line.addEventListener('click', () => {
            flushMessageQueue();
            state.messages = state.messages.filter(
              (m) =>
                !m.predictionPrompt &&
                !(m.predictionStepDone && m.predictionStepDone.personaIndex >= stepInfo.personaIndex),
            );
            const personas = getReviewersByCohort(state.selectedCohortId);
            const targetPersona = personas[stepInfo.personaIndex];
            if (targetPersona) {
              pushMessage({
                role: 'system',
                text: `\`${targetPersona.name}\` はどう思うか予想しよう`,
                predictionPrompt: { draft: { ...stepInfo.draftBefore } },
              });
            }
            renderStudio();
            scrollToLatestMessage();
          });
        }
        row.appendChild(line);
      }
      if (msg.showCohortSelector) {
        row.appendChild(renderCohortSelector(msg, msg.id === latestSelectorMessageId));
      }
      if (msg.showNextChallenge && state.selectedCohortId) {
        row.appendChild(renderNextChallengeSelector(msg, msg.id === latestChallengeMessageId));
      }
      if (row.childNodes.length === 0) continue;

      const isFirstSeen = !seenMessageIds.has(msg.id);
      seenMessageIds.add(msg.id);
      row.className = `chat-system-row${isFirstSeen ? ' is-entering' : ''}`;
      prevSpeakerKey = 'system';
      stream.appendChild(row);
      continue;
    }

    const isFirstSeen = !seenMessageIds.has(msg.id);
    seenMessageIds.add(msg.id);
    const enteringClass = isFirstSeen ? ' is-entering' : '';
    const currentSpeakerKey = speakerKeyOf(msg);
    const isContinuation = currentSpeakerKey === prevSpeakerKey;
    prevSpeakerKey = currentSpeakerKey;

    // お客さんの発言：アイコンと名前つきの吹き出し
    if (msg.role === 'customer') {
      const persona = findPersona(msg.personaId);
      const speakerName = persona?.name ?? 'お客さん';
      const outcomeClass = msg.review ? ` is-${msg.review.outcome}` : '';
      const bubble = createBubble(`chat-bubble is-customer${outcomeClass}${enteringClass}`, msg.text);
      const extras: HTMLElement[] = [];
      if (msg.review?.predictedOutcome) {
        const badgeInfo = formatPredictionBadgeText(msg.review.predictedOutcome, msg.review.outcome);
        const badge = document.createElement('div');
        badge.className = `prediction-result-badge ${badgeInfo.matched ? 'is-matched' : 'is-missed'}`;
        badge.textContent = badgeInfo.text;
        extras.push(badge);
      }
      stream.appendChild(
        createSpeakerRow(
          'customer',
          persona?.icon ?? '客',
          speakerName,
          bubble,
          extras,
          isContinuation,
          'customer',
        ),
      );
      const pointedSystemRow = createPointedSystemRow(
        speakerName,
        msg.pointedComponentIds,
        'customer',
        isFirstSeen,
      );
      if (pointedSystemRow) {
        stream.appendChild(pointedSystemRow);
        prevSpeakerKey = 'system';
      }
      continue;
    }

    // 店長の発言
    if (msg.role === 'owner') {
      const bubble = createBubble(`chat-bubble is-owner${enteringClass}`, msg.text);
      stream.appendChild(
        createSpeakerRow('owner', OWNER.icon, OWNER.name, bubble, [], isContinuation, 'owner'),
      );
      const pointedSystemRow = createPointedSystemRow(
        OWNER.name,
        msg.pointedComponentIds,
        'owner',
        isFirstSeen,
      );
      if (pointedSystemRow) {
        stream.appendChild(pointedSystemRow);
        prevSpeakerKey = 'system';
      }
      continue;
    }

    // お助けエージェントの発言：進行案内・質問回答・改造完了の報告
    if (msg.role === 'assistant') {
      const bubble = createBubble(`chat-bubble is-agent${enteringClass}`, msg.text);
      stream.appendChild(
        createSpeakerRow('agent', AGENT_ICON, AGENT_NAME, bubble, [], isContinuation, 'assistant'),
      );
      if (!msg.circleReaction) {
        const pointedSystemRow = createPointedSystemRow(
          AGENT_NAME,
          msg.pointedComponentIds,
          'assistant',
          isFirstSeen,
        );
        if (pointedSystemRow) {
          stream.appendChild(pointedSystemRow);
          prevSpeakerKey = 'system';
        }
      }
      continue;
    }

    // 学習者の発言
    const row = document.createElement('div');
    row.className = 'chat-bubble-row is-user';
    const stack = document.createElement('div');
    stack.className = 'user-bubble-stack';
    stack.appendChild(createBubble(`chat-bubble is-user${enteringClass}`, msg.text));
    if (canCancelOrUndoLastUserTurn() && lastUserTurn?.userMessageId === msg.id) {
      const undoBtn = document.createElement('button');
      undoBtn.type = 'button';
      undoBtn.className = 'user-message-undo-btn is-selectable';
      undoBtn.textContent = '送信を取り消し';
      undoBtn.title = '直前の発言を取り消して入力欄に戻す';
      undoBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        handleCancelLastUserMessage();
      });
      stack.appendChild(undoBtn);
    }
    row.appendChild(stack);
    stream.appendChild(row);
  }

  const cohortPersonas = state.selectedCohortId ? getReviewersByCohort(state.selectedCohortId) : [];
  const workLogRunning = state.messages.some((m) => m.workLog && !m.workLog.finished);
  const loadingPersona =
    state.chatLoading &&
    !workLogRunning &&
    state.recipient !== 'assistant' &&
    state.recipient !== 'owner'
      ? cohortPersonas[0]
      : undefined;

  // 店長の書き込み中の表示
  if (
    state.typingMessage?.role === 'owner' ||
    (state.chatLoading && !workLogRunning && state.recipient === 'owner')
  ) {
    stream.appendChild(
      createSpeakerRow(
        'owner',
        OWNER.icon,
        OWNER.name,
        createTypingBubble('chat-bubble is-owner is-loading', `${OWNER.name}が書き込み中`),
        [],
        prevSpeakerKey === 'owner',
      ),
    );
  }

  // お客さんの書き込み中の表示
  if (state.typingMessage?.role === 'customer' || loadingPersona) {
    const persona = loadingPersona ?? findPersona(state.typingMessage?.personaId);
    stream.appendChild(
      createSpeakerRow(
        'customer',
        persona?.icon ?? '',
        persona?.name ?? '',
        createTypingBubble('chat-bubble is-customer is-loading', `${persona?.name ?? ''}が書き込み中`),
        [],
        prevSpeakerKey === `customer:${persona?.id ?? ''}`,
      ),
    );
  }

  // お助けエージェントの書き込み中の表示（演出キューまたはAI応答待ち）
  if (
    state.typingMessage?.role === 'assistant' ||
    (state.chatLoading && !workLogRunning && state.recipient === 'assistant')
  ) {
    stream.appendChild(
      createSpeakerRow(
        'agent',
        AGENT_ICON,
        AGENT_NAME,
        createTypingBubble('chat-bubble is-agent is-loading', `${AGENT_NAME}が書き込み中`),
        [],
        prevSpeakerKey === 'assistant',
      ),
    );
  }

  // 反応確認中は状況を表示する
  if (state.evaluateLoading) {
    const row = document.createElement('div');
    row.className = 'chat-system-row';
    const line = document.createElement('div');
    line.className = 'chat-system-line is-progress';
    line.innerHTML = `
      <span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span>
      <span>お客さんが画面を操作しています</span>
    `;
    row.appendChild(line);
    stream.appendChild(row);
  }

  // 返事が遅いときは、経過した秒数を出す（以降は1秒ごとに tickWaitNote が書き換える）
  if (isWaitingForAi() && waitElapsedSec() >= WAIT_NOTE_AFTER_SEC) {
    stream.appendChild(buildWaitNoteRow());
  }

  chatCard.appendChild(stream);

  const composer = document.createElement('div');
  composer.className = 'vibe-composer';

  const recipientRow = document.createElement('div');
  recipientRow.className = 'composer-recipient-row';
  // 宛先のボタンを並べたあとに「に話しかける」と続け、ひと続きの文として読めるようにする
  const recipientLabel = document.createElement('span');
  recipientLabel.className = 'composer-recipient-label';
  recipientLabel.textContent = 'に話しかける';

  const latestEvalForRecipients = findLatestCohortEvaluation()?.evaluation;
  const allCustomersAchieved =
    cohortPersonas.length > 0 &&
    cohortPersonas.every(
      (p) =>
        latestEvalForRecipients !== undefined &&
        isReviewRevealed(latestEvalForRecipients, p.id) &&
        latestEvalForRecipients.results.find((r) => r.reviewerId === p.id)?.outcome === 'achieved',
    );
  const recipientOptions: Array<{
    key: string;
    label: string;
    isCustomer?: boolean;
    achieved?: boolean;
  }> = [
    { key: 'assistant', label: AGENT_NAME },
    { key: 'owner', label: OWNER.name },
    ...(cohortPersonas.length > 0
      ? [
          {
            key: 'customer',
            label: allCustomersAchieved ? '✓ お客さん' : 'お客さん',
            isCustomer: true,
            achieved: allCustomersAchieved,
          },
        ]
      : []),
  ];
  const isCohortLocked = !state.selectedCohortId;
  const now = performance.now();
  for (const opt of recipientOptions) {
    const pill = document.createElement('button');
    pill.type = 'button';
    pill.dataset.recipientKey = opt.key;
    pill.disabled = isCohortLocked;
    pill.className = `composer-recipient-pill${state.recipient === opt.key ? ' is-active' : ''}${opt.isCustomer ? ' is-customer-pill' : ''}${opt.achieved ? ' is-achieved-customer' : ''}`;
    if (opt.isCustomer) {
      pill.title = opt.achieved
        ? 'お客さん全員が満足して注文できました（クリックで話しかける）'
        : 'お客さんに質問する';
      let firstSeen = recipientFirstSeenAt.get(opt.key);
      if (firstSeen === undefined) {
        firstSeen = now;
        recipientFirstSeenAt.set(opt.key, firstSeen);
      }
      const elapsed = now - firstSeen;
      if (elapsed < RECIPIENT_ENTER_DURATION_MS) {
        pill.classList.add('is-entering');
        pill.style.animationDelay = `${Math.round(-elapsed)}ms`;
      }
    }
    pill.textContent = opt.label;
    pill.addEventListener('click', () => {
      if (isCohortLocked) return;
      selectRecipientAndFocus(opt.key);
    });
    recipientRow.appendChild(pill);
  }
  recipientRow.appendChild(recipientLabel);

  composer.appendChild(recipientRow);

  const formRow = document.createElement('form');
  formRow.className = 'composer-form-row';
  formRow.addEventListener('submit', (e) => {
    e.preventDefault();
    void handleSendMessage();
  });

  const inputWrap = document.createElement('div');
  inputWrap.className = `composer-input-wrap${state.isRecordingAudio ? ' is-recording' : ''}${state.isTranscribingAudio ? ' is-transcribing' : ''}${isCohortLocked ? ' is-locked' : ''}`;

  const sendBtn = document.createElement('button');
  sendBtn.type = 'submit';
  // 送信ボタンは送るだけにする。直前の発言の取り消しは、チャットの発言の下の「送信を取り消し」で行う
  sendBtn.className = 'btn-composer-send';
  sendBtn.textContent = '送信';
  const syncSendButtonState = (): void => {
    sendBtn.disabled =
      isCohortLocked ||
      !state.inputText.trim() ||
      state.isRecordingAudio ||
      state.isTranscribingAudio;
  };
  syncSendButtonState();

  const input = getComposerInput();
  input.disabled = isCohortLocked;
  input.placeholder = isCohortLocked
    ? cohortSelectionGuidance()
    : state.runFinished
      ? '挑戦は終わりました。やり直すとまた手直しできます（店長とお客さんには質問できます）'
      : state.isRecordingAudio
      ? '録音中… 話し終えたら右の停止ボタンを押してください'
      : state.isTranscribingAudio
        ? '音声から指示を整えています…'
        : getRecipientPlaceholder(state.recipient);
  // 日本語を変換している途中で値を書き換えると変換が消えるので、変換中は触らない
  if (!composerIsComposing && input.value !== state.inputText) {
    input.value = state.inputText;
  }
  onComposerKeydown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      if (e.isComposing || e.keyCode === 229) return;
      e.preventDefault();
      if (
        !isCohortLocked &&
        state.inputText.trim() &&
        !state.isRecordingAudio &&
        !state.isTranscribingAudio
      ) {
        void handleSendMessage();
      }
    }
  };
  onComposerInput = () => {
    state.inputText = input.value;
    syncComposerTextareaHeight(input);
    syncSendButtonState();
    const switched = detectExplicitRecipientMention(state.inputText);
    if (switched && switched !== state.recipient) {
      state.recipient = switched;
      for (const pillEl of recipientRow.querySelectorAll<HTMLButtonElement>(
        '.composer-recipient-pill',
      )) {
        const active = pillEl.dataset.recipientKey === state.recipient;
        pillEl.classList.toggle('is-active', active);
        if (active) {
          pillEl.classList.remove('is-auto-switched');
          void pillEl.offsetWidth;
          pillEl.classList.add('is-auto-switched');
        }
      }
      input.placeholder = getRecipientPlaceholder(state.recipient);
    }
  };

  const micBtn = document.createElement('button');
  micBtn.type = 'button';
  micBtn.className = `btn-composer-mic${state.isRecordingAudio ? ' is-recording' : ''}`;
  micBtn.disabled = isCohortLocked || state.chatLoading || state.isTranscribingAudio;
  micBtn.title = isCohortLocked
    ? cohortSelectionGuidance()
    : state.isRecordingAudio
      ? '録音を終えて入力欄に反映する'
      : state.isTranscribingAudio
        ? '音声を整えています…'
        : '声で入力する（雑音や言い淀みを省いて入力欄に整えます）';
  micBtn.setAttribute(
    'aria-label',
    state.isRecordingAudio ? '録音を終えて入力欄に反映' : '音声で入力',
  );
  micBtn.innerHTML = state.isRecordingAudio
    ? `${ICONS.micStop}<span>停止</span>`
    : state.isTranscribingAudio
      ? '<span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span>'
      : ICONS.mic;
  micBtn.addEventListener('click', () => {
    void toggleVoiceRecording();
  });

  // 入力欄そのものは画面を描き直しても作り直さない。ここには目印だけ置き、mountChatPanel で同じ入力欄をはめ込む
  const inputSlot = document.createElement('span');
  inputSlot.dataset.composerInputSlot = '1';
  inputWrap.append(inputSlot, micBtn);

  formRow.append(inputWrap, sendBtn);
  if (state.pendingSend) {
    const note = document.createElement('div');
    note.className = 'composer-pending-note';
    const noteText = document.createElement('span');
    noteText.textContent = '返事が届いたら送ります';
    const cancelPendingBtn = document.createElement('button');
    cancelPendingBtn.type = 'button';
    cancelPendingBtn.className = 'composer-pending-cancel-btn';
    cancelPendingBtn.textContent = '取消';
    cancelPendingBtn.addEventListener('click', () => {
      if (!state.pendingSend) return;
      state.pendingSend = false;
      renderStudio();
      focusComposerInput(true);
    });
    note.append(noteText, cancelPendingBtn);
    composer.appendChild(note);
  }
  composer.appendChild(formRow);

  chatCard.appendChild(composer);
  return chatCard;
}

function ensureStudioShellInitialized(): void {
  if (!appRoot || shellEl) return;

  shellEl = document.createElement('div');
  shellEl.className = 'studio-shell';

  headerSlotEl = document.createElement('div');
  headerSlotEl.className = 'studio-header-slot';
  shellEl.appendChild(headerSlotEl);

  const stage = document.createElement('main');
  stage.className = 'game-stage';

  const deviceColumn = document.createElement('section');
  deviceColumn.className = 'device-column';

  phoneFrameEl = document.createElement('div');
  phoneFrameEl.className = 'smartphone-bezel';

  sandboxStageEl = document.createElement('div');
  sandboxStageEl.className = 'sandbox-viewport';

  sandboxAppMountEl = document.createElement('div');
  sandboxAppMountEl.className = 'sandbox-app-mount';
  sandboxStageEl.appendChild(sandboxAppMountEl);

  penCanvasEl = document.createElement('canvas');
  penCanvasEl.className = 'pen-overlay-canvas';
  sandboxStageEl.appendChild(penCanvasEl);
  attachPenEvents(penCanvasEl, sandboxStageEl);

  phoneFrameEl.appendChild(sandboxStageEl);

  bottomBarEl = document.createElement('div');
  bottomBarEl.className = 'device-bottom-bar';

  deviceColumn.append(phoneFrameEl, bottomBarEl);

  chatSlotEl = document.createElement('div');
  chatSlotEl.className = 'chat-column-slot';

  stage.append(deviceColumn, chatSlotEl);
  shellEl.appendChild(stage);

  // 画面の下部：みんなの集計と、利用上の注意・集計への協力の読み直し
  const footer = document.createElement('footer');
  footer.className = 'studio-footer';

  // 同意された挑戦の集計を見る
  const statsBtn = document.createElement('button');
  statsBtn.type = 'button';
  statsBtn.className = 'footer-link stats-btn';
  statsBtn.textContent = 'みんなの集計';
  statsBtn.addEventListener('click', openAggregateStatsDialog);
  footer.appendChild(statsBtn);

  // 公開ホスト用の案内を有効にしたときだけ、利用上の注意と集計への協力を読み直せるリンクを出す
  if (PUBLIC_NOTICE_ENABLED) {
    const aboutBtn = document.createElement('button');
    aboutBtn.type = 'button';
    aboutBtn.className = 'footer-link about-btn';
    aboutBtn.textContent = 'このツールについて';
    aboutBtn.addEventListener('click', () => openTermsDialog('review'));
    footer.appendChild(aboutBtn);
  }

  shellEl.appendChild(footer);

  appRoot.innerHTML = '';
  appRoot.appendChild(shellEl);
}

function renderStudio(): void {
  if (!appRoot) return;
  ensureStudioShellInitialized();
  const spec = getCurrentSpec();

  const activeEl = document.activeElement;
  const focusedComposer =
    activeEl instanceof HTMLTextAreaElement && activeEl.classList.contains('composer-input')
      ? activeEl
      : null;
  const selectionStart = focusedComposer ? focusedComposer.selectionStart : null;
  const selectionEnd = focusedComposer ? focusedComposer.selectionEnd : null;
  const prevScrollTop = chatScrollEl ? chatScrollEl.scrollTop : 0;

  if (headerSlotEl) {
    headerSlotEl.innerHTML = '';
    const header = document.createElement('header');
    header.className = 'studio-header';

    const cohort = state.selectedCohortId
      ? TARGET_COHORTS.find((c) => c.id === state.selectedCohortId)
      : undefined;
    const cohortPersonas = state.selectedCohortId
      ? getReviewersByCohort(state.selectedCohortId)
      : [];
    const evalRes = evaluationFor(getCurrentCheckpoint());
    // まだ言い終えていないお客さんは、満足の欄に数えない
    const achievedOf = (e: EvaluationResponse) =>
      e.results.filter((r) => r.outcome === 'achieved' && isReviewRevealed(e, r.reviewerId))
        .length;
    const prevEval = !evalRes
      ? findLatestCohortEvaluation(state.activeCheckpointIndex - 1)?.evaluation
      : undefined;
    const previous = prevEval ? { evalRes: prevEval, achieved: achievedOf(prevEval) } : undefined;

    const goalGroup = document.createElement('div');
    goalGroup.className = 'goal-status-group';

    // まだ選んでいないときは押せるようにし、押すとチャットの選ぶボタンへ案内する
    const cohortBadge = document.createElement(cohort ? 'div' : 'button');
    cohortBadge.className = `goal-cohort-badge${cohort ? '' : ' is-empty is-clickable'}`;
    if (!cohort) {
      (cohortBadge as HTMLButtonElement).type = 'button';
      cohortBadge.title = cohortSelectionGuidance();
      cohortBadge.addEventListener('click', focusCohortSelector);
    }
    const cohortLabelEl = document.createElement('span');
    cohortLabelEl.className = 'goal-badge-prefix';
    cohortLabelEl.textContent = '対象';
    const cohortNameEl = document.createElement('span');
    cohortNameEl.className = 'goal-cohort-name';
    cohortNameEl.textContent = cohort ? cohort.label : '未選択';
    cohortBadge.append(cohortLabelEl, cohortNameEl);

    const progressBadge = document.createElement('div');
    const activeEval = evalRes ?? previous?.evalRes;
    const currentAchieved = evalRes ? achievedOf(evalRes) : (previous?.achieved ?? 0);
    const isCleared = Boolean(
      cohortPersonas.length > 0 && evalRes && currentAchieved === cohortPersonas.length,
    );
    const isUnverified = Boolean(state.selectedCohortId && !evalRes && state.checkpoints.length > 1);
    progressBadge.className = `goal-progress-badge${isCleared ? ' is-cleared' : ''}${isUnverified ? ' is-unverified' : ''}${cohort ? '' : ' is-empty'}`;

    const progressPrefixEl = document.createElement('span');
    progressPrefixEl.className = 'goal-badge-prefix';
    progressPrefixEl.textContent = '満足';

    const dotsWrap = document.createElement('span');
    dotsWrap.className = 'goal-progress-dots';
    const dotCount = cohortPersonas.length > 0 ? cohortPersonas.length : 3;
    for (let i = 0; i < dotCount; i++) {
      const persona = cohortPersonas[i];
      const dot = document.createElement('span');
      const personaAchieved =
        persona &&
        activeEval &&
        isReviewRevealed(activeEval, persona.id) &&
        activeEval.results.find((r) => r.reviewerId === persona.id)?.outcome === 'achieved';
      dot.className = `goal-dot${personaAchieved ? ' is-achieved' : ''}${!evalRes && previous ? ' is-pending' : ''}`;
      // 満足した人はサムズアップ、まだの人は同じ形を薄い灰色で出し、空いている枠に見せる
      dot.textContent = '👍';
      dot.setAttribute('aria-hidden', 'true');
      if (persona) {
        dot.title = `${persona.name}：${personaAchieved ? '満足して注文できた' : 'まだ困っている'}`;
      }
      dotsWrap.appendChild(dot);
    }

    if (isUnverified) {
      progressBadge.title = '画面を手直ししました。「お客さんの反応を見る」で確かめられます';
    } else if (cohort) {
      progressBadge.title = `満足して注文できた人数：${currentAchieved}/${dotCount}人`;
    }

    progressBadge.append(progressPrefixEl, dotsWrap);

    const voiceToggleBtn = document.createElement('button');
    voiceToggleBtn.type = 'button';
    voiceToggleBtn.className = `header-voice-toggle${state.voiceEnabled ? '' : ' is-off'}`;
    const voiceLabel = state.voiceEnabled
      ? '音声読み上げをオフにする'
      : '音声読み上げはオフになっています（クリックでオンに戻す）';
    voiceToggleBtn.title = voiceLabel;
    voiceToggleBtn.setAttribute('aria-label', voiceLabel);
    voiceToggleBtn.innerHTML = state.voiceEnabled ? ICONS.volumeOn : ICONS.volumeOff;
    voiceToggleBtn.addEventListener('click', () => {
      state.voiceEnabled = !state.voiceEnabled;
      if (!state.voiceEnabled) {
        stopAllTtsPlayback();
      }
      renderStudio();
    });

    const headerChatBar = document.createElement('div');
    headerChatBar.className = 'header-chat-bar';

    goalGroup.append(cohortBadge, progressBadge);
    headerChatBar.append(goalGroup, voiceToggleBtn);
    header.appendChild(headerChatBar);

    headerSlotEl.appendChild(header);
  }

  if (phoneFrameEl) {
    phoneFrameEl.className = `smartphone-bezel ${state.interactionMode === 'pen' ? 'is-pen-active' : ''}`;
  }

  if (bottomBarEl) {
    bottomBarEl.innerHTML = '';
    const modeSwitch = document.createElement('div');
    modeSwitch.className = 'bottom-mode-switch';

    const playModeBtn = document.createElement('button');
    playModeBtn.type = 'button';
    playModeBtn.className = `header-mode-btn ${state.interactionMode === 'play' ? 'is-active' : ''}`;
    playModeBtn.innerHTML = `${ICONS.cursor}<span>操作する</span>`;
    playModeBtn.addEventListener('click', () => {
      state.interactionMode = 'play';
      renderStudio();
    });

    const penModeBtn = document.createElement('button');
    penModeBtn.type = 'button';
    penModeBtn.className = `header-mode-btn is-pen ${state.interactionMode === 'pen' ? 'is-active' : ''}`;
    penModeBtn.disabled = !state.selectedCohortId;
    if (!state.selectedCohortId) {
      penModeBtn.title = cohortSelectionGuidance();
    }
    penModeBtn.innerHTML = `${ICONS.pen}<span>赤ペンで囲む</span>`;
    penModeBtn.addEventListener('click', () => {
      if (!state.selectedCohortId) return;
      state.interactionMode = 'pen';
      renderStudio();
    });

    modeSwitch.append(playModeBtn, penModeBtn);

    const hasUnevaluatedChanges =
      state.checkpoints.length > 1 && !evaluationFor(getCurrentCheckpoint());
    const evalBtn = document.createElement('button');
    evalBtn.type = 'button';
    evalBtn.className = `btn-bottom-evaluate ${hasUnevaluatedChanges ? 'is-ready' : ''}`;
    // 誰のために改造するかを選ぶまでは、お客さんが来ていないので反応を見られない
    evalBtn.disabled = state.evaluateLoading || !state.selectedCohortId || state.runFinished;
    // 押せない間は目立たない色にして、最初に目が行かないようにする
    evalBtn.classList.toggle('is-locked', !state.selectedCohortId);
    if (!state.selectedCohortId) evalBtn.title = cohortSelectionGuidance();
    evalBtn.innerHTML = `${ICONS.evaluate}<span>${
      state.evaluateLoading ? '反応を確認中...' : 'お客さんの反応を見る'
    }</span>`;
    evalBtn.addEventListener('click', () => {
      handleRequestEvaluate();
    });

    bottomBarEl.append(modeSwitch, evalBtn);
  }

  if (chatSlotEl) {
    const chatPanel = renderChatPanel();
    mountChatPanel(chatSlotEl, chatPanel);
    const composerTextarea = chatSlotEl.querySelector<HTMLTextAreaElement>('.composer-input');
    if (composerTextarea) {
      syncComposerTextareaHeight(composerTextarea);
    }
    if (chatScrollEl) {
      chatScrollEl.scrollTop = prevScrollTop;
    }
  }

  // スマホ画面の仕様・状態・選択部品が変わったときだけサンドボックスを再描画する（顧客層切替等では一切リロードしない）
  const nextSandboxKey = JSON.stringify({
    spec,
    liveAppState: state.liveAppState,
    circled: state.circledComponentIds,
    // 作業ログの行やシステムメッセージ内の部品名を押したときに部品を光らせるため、光らせる部品が変わったときも描き直す
    justChanged: state.justChangedIds,
  });
  if (sandboxAppMountEl && nextSandboxKey !== lastSandboxRenderKey) {
    lastSandboxRenderKey = nextSandboxKey;
    renderSandboxApp(
      sandboxAppMountEl,
      spec,
      state.liveAppState,
      new Set(state.circledComponentIds),
      {
        justChangedIds: new Set(state.justChangedIds),
        onStateChange: (nextState) => {
          state.liveAppState = nextState;
          renderStudio();
        },
        onStatePatch: (patch) => {
          state.liveAppState = withDerivedValues(getCurrentSpec(), {
            ...state.liveAppState,
            ...patch,
          });
          renderStudio();
        },
        onNotice: (msg, t) => postSandboxNotice(msg, t),
        onScroll: () => redrawPenCanvas(),
      },
    );
  }

  if (focusedComposer && shellEl) {
    const nextInput = shellEl.querySelector<HTMLTextAreaElement>('.composer-input');
    // 同じ入力欄が画面に残っていればフォーカスも変換もそのままなので、触らない
    if (nextInput && document.activeElement !== nextInput) {
      nextInput.focus();
      if (selectionStart !== null && selectionEnd !== null) {
        nextInput.setSelectionRange(selectionStart, selectionEnd);
      }
    }
  }

  window.requestAnimationFrame(() => {
    redrawPenCanvas();
  });
}

window.addEventListener('resize', () => {
  redrawPenCanvas();
  const composerTextarea = shellEl?.querySelector<HTMLTextAreaElement>('.composer-input');
  if (composerTextarea) {
    syncComposerTextareaHeight(composerTextarea);
  }
});

// 公開ホスト用の案内が有効なときだけ利用上の注意を先に出し、既定ではすぐ店長の紹介を始める
renderStudio();
if (PUBLIC_NOTICE_ENABLED) {
  openTermsDialog('first', playInitialWelcomeSequence);
} else {
  playInitialWelcomeSequence();
}

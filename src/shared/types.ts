// お題アプリのハイブリッド構造化スキーマ (判断1: 案C)
// UI部品ツリーと状態変数は型安全なJSONで管理し、単一HTMLとしてiframe内で隔離実行する。

import type { Rect2D } from './hitTest.ts';
import type { CustomComponentProfile } from './customComponentProfile.ts';
import type {
  EmbeddingSignal,
  NaturalLanguageSignal,
  PromptArchetype,
  PromptSessionMode,
  PromptSessionSummary,
  PromptTelemetryEvent,
  PublicAggregateStats,
} from './promptTelemetry.ts';

export type {
  EmbeddingSignal,
  NaturalLanguageSignal,
  PromptArchetype,
  PromptSessionMode,
  PromptSessionSummary,
  PromptTelemetryEvent,
  PublicAggregateStats,
};

export type StateValue = string | number | boolean;

export interface AppStateVar {
  key: string;
  label: string;
  type: 'string' | 'number' | 'boolean';
  initialValue: StateValue;
}

/**
 * - progress_steps: 手順の表示。options が手順名、label が今の手順
 * - product_card: 商品カード。label が商品名、badge・caption・priceLabel を添え、stateKey の数量を±ボタンで変える
 * - choice_cards: 選べるカード。options が選択肢、optionDetails が各選択肢の補足
 * - summary: 金額の内訳。rows に行を並べる
 */
export type ComponentType =
  | 'heading'
  | 'text'
  | 'product_hero'
  | 'product_card'
  | 'progress_steps'
  | 'choice_cards'
  | 'summary'
  | 'input'
  | 'select'
  | 'checkbox'
  | 'toggle'
  | 'slider'
  | 'button'
  | 'status_banner'
  | 'popup'
  | 'custom';

export type PopupClosePosition =
  | 'top-right'
  | 'top-left'
  | 'bottom-right'
  | 'bottom-left'
  | 'bottom-bar';

export type ComponentSize = 'xs' | 'sm' | 'md' | 'lg' | 'xl';
export type ComponentVariant = 'default' | 'primary' | 'danger' | 'warning' | 'success' | 'muted';

export interface ComponentStyle {
  size: ComponentSize;
  variant: ComponentVariant;
  align?: 'left' | 'center' | 'right';
  fullWidth?: boolean;
  noisyAnimation?: boolean;
  /** 部品を傾ける角度（度）。0 または未指定でまっすぐ */
  rotateDeg?: number;
  /** ラベル内の {状態キー} に差し込む数値の表記。kanji で漢数字にする */
  numberFormat?: 'arabic' | 'kanji';
  /** true なら画面下に固定したバーに置く（合計金額と購入ボタンなど） */
  sticky?: boolean;
  /**
   * 見た目の系統。native は元のデザインに揃える。
   * intrusive は後から付け足されたように、元のデザインの決まりを無視した見た目で描く
   */
  skin?: 'native' | 'intrusive';
}

export type BehaviorEffect = 'none' | 'escape_on_hover' | 'require_3_clicks' | 'confirm_dialog';

export interface ComponentBehavior {
  effect: BehaviorEffect;
  confirmText?: string;
}

export type ActionType =
  | 'set_state'
  | 'toggle_state'
  | 'increment_state'
  | 'reset_all'
  | 'show_toast'
  | 'open_page'
  | 'complete_task';

/**
 * - open_page: 別のページに飛ばす。message がそのページの見出しになり、少しすると最初の画面に戻る
 * - complete_task: タスク完了。message が完了ページの見出しになり、少しすると最初の画面に戻る
 */
export interface ComponentAction {
  type: ActionType;
  targetKey?: string;
  value?: StateValue;
  message?: string;
}

export interface VisibilityCondition {
  stateKey: string;
  operator: 'eq' | 'neq' | 'gt' | 'lt';
  value: StateValue;
}

/**
 * 部品が画面の中で果たす役割。操作シミュレーターは部品のIDではなく役割を見て判定する。
 * - primary_action: タスクを先へ進める本物のボタン（complete_task を持つもの、確認画面へ進むもの）
 * - decoy: 本物らしく見えるが押しても何も起きないおとり
 * - danger: 押すと入力が消えるなど、誤って押すと離脱につながるボタン
 * - required_input: タスク完了までに入力が必要な欄。値が空か placeholder と同じなら未入力
 * - distraction: タスクと関係ない項目や煽り表示
 * - trap: 気づかずに完了すると後悔につながる項目（最初からチェックの入った定期便など）
 * - total: 支払う合計金額などタスクの結果を示す表示
 * - popup: 時間差で画面を覆う広告など
 * - info: 上記以外の説明表示
 */
export type ComponentRole =
  | 'primary_action'
  | 'decoy'
  | 'danger'
  | 'required_input'
  | 'distraction'
  | 'trap'
  | 'total'
  | 'popup'
  | 'info';

export interface AppComponent {
  id: string;
  type: ComponentType;
  role: ComponentRole;
  label: string;
  /** product_hero・product_card に表示する絵。coffee_bag ならコーヒー袋のイラスト、それ以外は絵文字として描く */
  icon?: string;
  /** product_card の上に添える小さな札（例: 深煎り・ブラジル） */
  badge?: string;
  /** label の下に添える補足。{状態キー} を差し込める */
  caption?: string;
  /** product_card の値段。{状態キー} を差し込める（例: ¥{unitPrice}） */
  priceLabel?: string;
  /** choice_cards の各選択肢の補足（options と同じ順番） */
  optionDetails?: string[];
  /**
   * summary の行。label と value に {状態キー} を差し込める。
   * visibleWhen があれば、条件を満たすときだけ出す（手数料や割引が0円のときに行を隠すなど）
   */
  rows?: Array<{ label: string; value: string; visibleWhen?: VisibilityCondition }>;
  placeholder?: string;
  options?: string[];
  stateKey?: string;
  style: ComponentStyle;
  behavior?: ComponentBehavior;
  visibleWhen?: VisibilityCondition;
  /**
   * お客さんをだます仕掛けの種類。false_urgency は偽の残り時間などで急かすもの。
   * ラベルに {countdown} がある部品も急かす仕掛けとして扱う
   */
  darkPattern?: 'false_urgency';
  /** role が trap のとき、この条件が成り立ったまま完了すると後悔につながる */
  trapWhen?: VisibilityCondition;
  /** role が trap のとき、気づいた人が直す値。未指定なら気づいても直せない（承知のうえで進む） */
  trapSafeValue?: StateValue;
  /** popup が現れるまでの時間の範囲（ミリ秒）。この範囲でランダムに現れる */
  popupDelayMs?: { min: number; max: number };
  /** popup の閉じるボタンの位置（既定は top-right。bottom-left や bottom-bar 等に変更可能） */
  closePosition?: PopupClosePosition;
  /** popup の閉じるボタンの大きさ（未指定なら style.size） */
  closeSize?: ComponentSize;
  /** popup の閉じるボタンのラベル（既定は ×、bottom-bar のときは 閉じる） */
  closeLabel?: string;
  /**
   * サンドボックス内で自由に描画するカスタムHTML。
   * {状態キー} で状態値を表示でき、data-set-state="key:value" / data-toggle-state="key" /
   * data-increment-state="key" / data-complete-task="message" 属性で状態変更や完了イベントを発火できる。
   */
  customHtml?: string;
  /** この部品に適用する追加のカスタムCSS */
  customCss?: string;
  actions: ComponentAction[];
}

export interface AppTheme {
  bgColor: string;
  cardColor: string;
  textColor: string;
  accentColor: string;
  chaoticLayout: boolean;
}

/** 状態から計算して求める値（合計金額など）。式は四則演算・比較・三項演算子のみ */
export interface DerivedValue {
  key: string;
  expression: string;
}

export interface AppSpec {
  id: string;
  title: string;
  subtitle: string;
  taskDescription: string;
  theme: AppTheme;
  state: AppStateVar[];
  derived?: DerivedValue[];
  components: AppComponent[];
}

// 顧客層とペルソナ＋制約変数（題材ごとに定義する）
export type TargetCohortId = string;

export interface TargetCohortInfo {
  id: TargetCohortId;
  label: string;
  shortDesc: string;
}

export interface PersonaConstraints {
  /** 視力制約: スクリーンショットにかけるガウシアンぼかし半径(px) */
  blurPx: number;
  /** 視力制約: コントラスト倍率 (1.0が標準、低いほど見分けにくい) */
  contrast: number;
  /** 指の正確さ制約: クリック座標に加算される最大ズレ幅(px) */
  jitterPx: number;
  /** 手数の上限: この回数以内にタスク完了イベントが出なければ離脱 */
  maxSteps: number;
  /** 待てる時間(ms): 操作完了までに許容する最大時間 */
  patienceMs: number;
  /** 使用環境タグ */
  environment: 'normal' | 'one_hand' | 'moving_train';
  /** 関係ない項目・煽り表示・広告をいくつまで無視できるか */
  clutterTolerance: number;
  /** 細かい文字やチェック欄まで読む。読める大きさなら罠に気づける */
  readsFinePrint: boolean;
  /** 完了前に確認の機会がないと不安で押せない */
  wantsConfirmation: boolean;
  /** 合計金額がすぐ読めないと完了しない */
  needsReadableTotal: boolean;
  /** 残り時間などで急かされると嫌な気分になる（買えても不満が残る） */
  dislikesPressure?: boolean;
  /** まともな定期便があれば申し込みたい。定期便の欄自体が消されると買えても不満が残る */
  wantsSubscription?: boolean;
  /** 割引クーポンがあれば使いたい。クーポン自体が消されると買えても不満が残る */
  wantsCoupon?: boolean;
}

export interface ReviewerPersona {
  id: string;
  cohortId: TargetCohortId;
  traitLabel: string;
  name: string;
  age: number;
  role: string;
  icon: string;
  goal: string;
  /** 来店時にお客さん本人が話す一言（口語） */
  greeting: string;
  focusPoint: string;
  constraints: PersonaConstraints;
}

/** 達成・離脱に加え、完了はできたが罠に気づかず後悔したケースを区別する */
/**
 * achieved は嫌な思いをせず買えた、bothered は買えたが途中で嫌な思いをした（おとりを押した・罠に気づいて自分で外した・広告を閉じた など）、
 * abandoned は途中でやめた、regretted は買えたが罠に気づかず後悔した
 */
export type ReviewOutcome = 'achieved' | 'bothered' | 'abandoned' | 'regretted';

export interface ReviewerEvaluation {
  reviewerId: string;
  outcome: ReviewOutcome;
  frictionPoint: string;
  comment: string;
  /** Computer Use 実操作シミュレーションの軌跡ログ */
  actionTrace?: string[];
  /** 適用された制約変数の要約 */
  appliedConstraintNote?: string;
  /** 困りごとや後悔の原因になった画面上の部品ID */
  frictionComponentIds?: string[];
}

/** 店長が採点のたびに見る、お店の側の結果 */
export interface OwnerReport {
  /** 注文してくれた人数（後悔した人も注文は入っている） */
  orders: number;
  customers: number;
  metrics: Array<{ label: string; count: number; missing: boolean }>;
  /** 店長の一言 */
  comment: string;
  /** 選んだお客さん全員が満足したときの、AIが作った店長の締めのせりふ */
  closingLines?: string[];
}

export interface EvaluationResponse {
  evaluatedAt: string;
  cohortId?: TargetCohortId;
  results: ReviewerEvaluation[];
  owner?: OwnerReport;
  /** クリア時などの行動ベースのプロンプト到達効率サマリー */
  telemetrySummary?: PromptSessionSummary;
  /**
   * 判定に使った、独自の部品の読み取り結果（部品のID → 結果）。
   * 選ばなかった客層の判定も、画面側で同じ読み取り結果を使ってそろえる
   */
  customComponentProfiles?: Record<string, CustomComponentProfile>;
  /** 読み取れず、部品のHTMLに書かれた動きと文字だけで判定した独自の部品のID */
  customComponentFallbackIds?: string[];
}

// 人間の判断記録 (Own the Why / Function Calling で構造化記録)
export type DecisionStage = 'select_cohort' | 'apply_change' | 'review_disposition';

export interface HumanDecisionRecord {
  id: string;
  stage: DecisionStage;
  targetId: string;
  chosenLabel: string;
  reason: string;
  recordedVia: 'gemini_function_call' | 'local_function_call';
  createdAt: string;
}

export interface RecordDecisionRequest {
  stage: DecisionStage;
  targetId: string;
  chosenLabel: string;
  reason: string;
}

export interface RecordDecisionResponse {
  decision: HumanDecisionRecord;
  persistedToFirestore: boolean;
}

// サンドボックス検証結果
export interface SandboxValidationResult {
  valid: boolean;
  canRenderHtml: boolean;
  hasTaskCompletionEvent: boolean;
  errors: string[];
}

// 差分パッチ操作（案B: 変更箇所だけを適用し、触っていない部品を保護する）
export type SpecPatch =
  | { op: 'remove_component'; id: string }
  | { op: 'replace_component'; component: AppComponent }
  | { op: 'insert_component'; afterId?: string | null; component: AppComponent }
  | { op: 'move_component'; id: string; afterId?: string | null }
  | { op: 'upsert_state'; variable: AppStateVar }
  | { op: 'upsert_derived'; key: string; expression: string };

/** AIに渡す直近の会話（発言者名と本文） */
export interface ChatHistoryEntry {
  speaker: string;
  text: string;
}

// 改修反映 API の入出力
export interface ApplyRequest {
  currentSpec: AppSpec;
  targetComponentIds: string[];
  userInstruction: string;
  cohortId?: TargetCohortId;
  /** 1回の挑戦ごとの番号。振り返りの記録を、ほかの挑戦やほかの人の記録と混ぜないために使う */
  runId?: string;
  /** その挑戦の中で何番目の発言か。指示の本文の代わりに、振り返りの引用の目印にする */
  turnIndex?: number;
  /** 学習者が、この挑戦の記録をみんなの集計に加えることに同意したか */
  shareForAggregate?: boolean;
  annotatedImageBase64?: string;
  history?: ChatHistoryEntry[];
}

export interface ApplyResponse {
  commitMessage: string;
  updatedSpec: AppSpec;
  patches?: SpecPatch[];
  /** 改造の作業内容を短い言葉にしたもの。チャットで作業の過程として見せる */
  workSteps?: Array<{ text: string; componentId?: string }>;
  /** 店長の決まったせりふ（AIを使えないときの代わり） */
  ownerReaction?: string;
  /** 店長が気にする変化の事実。店長のせりふはこれをもとにAIが作る */
  ownerFacts?: string[];
  /** 店長が言及している部品のID */
  ownerComponentIds?: string[];
  validation?: SandboxValidationResult;
  rolledBack?: boolean;
}

// 店長のせりふ API の入出力
export interface OwnerLineRequest {
  /** change は改造の直後、closing は選んだお客さん全員が満足したとき、reply は店長に話しかけられたとき */
  situation: 'change' | 'closing' | 'reply';
  /** reply のときの、学習者が店長に話しかけた言葉 */
  userMessage?: string;
  facts: string[];
  /** AIが使えないときのせりふ */
  fallback: string;
  history?: ChatHistoryEntry[];
}

export interface OwnerLineResponse {
  text: string;
}

// レビュアー使用テスト API の入出力
export type ComponentBoundsInput = Rect2D;

export interface EvaluateRequest {
  currentSpec: AppSpec;
  cohortId?: TargetCohortId;
  /** 1回の挑戦ごとの番号。振り返りの記録を、ほかの挑戦やほかの人の記録と混ぜないために使う */
  runId?: string;
  /** 取り消されずに残っている発言の数。直前の発言を取り消したあとに反応を見たとき、取り消した発言を集計から外すために使う */
  turnCount?: number;
  screenshotBase64?: string;
  componentBounds?: ComponentBoundsInput[];
  /** 店長が同じ言い回しを繰り返さないための、これまでの会話 */
  history?: ChatHistoryEntry[];
}

// チェックポイント履歴 (Gitのコミット相当)
export interface Checkpoint {
  id: string;
  stepNumber: number;
  commitMessage: string;
  createdAt: string;
  spec: AppSpec;
  evaluation?: EvaluationResponse;
}

// 自由対話タイムライン型ハイブリッド壁打ち API
/** explain は質問への回答、clarify は曖昧な発言への聞き返し、apply は改修の実行、owner は店長の返事、customer はお客さん本人の返事 */
export type ChatIntent = 'explain' | 'clarify' | 'apply' | 'owner' | 'customer';

/** お助けエージェントに渡す、直近にお客さんが実際に操作した記録 */
export interface LatestReviewRecord {
  /** どの画面での記録か（最初・改造1など）と、それが今の画面かどうか */
  screenLabel: string;
  isCurrentScreen: boolean;
  reviews: Array<{
    name: string;
    outcome: ReviewOutcome;
    actionTrace: string[];
    frictionComponentIds?: string[];
  }>;
  /** そのとき店長が数えた数字（定期便の件数など）。店長に聞かれたときに答えられるようにする */
  ownerMetrics?: Array<{ label: string; count: number; missing: boolean }>;
}

export interface ChatRequest {
  currentSpec: AppSpec;
  latestReviews?: LatestReviewRecord;
  targetComponentIds: string[];
  userMessage: string;
  cohortId?: TargetCohortId;
  /** 1回の挑戦ごとの番号。振り返りの記録を、ほかの挑戦やほかの人の記録と混ぜないために使う */
  runId?: string;
  /** その挑戦の中で何番目の発言か。指示の本文の代わりに、振り返りの引用の目印にする */
  turnIndex?: number;
  /** 学習者が、この挑戦の記録をみんなの集計に加えることに同意したか */
  shareForAggregate?: boolean;
  annotatedImageBase64?: string;
  history?: ChatHistoryEntry[];
  /** circled は、学習者が赤ペンで囲んだだけのとき。受け止めて尋ねるだけで、画面は変えない */
  mode?: 'circled';
  /** 誰に向けて送った発言か（assistant / owner / customer / ペルソナID） */
  recipient?: string;
  /** 決まった条件で描画して測った部品の座標。お客さんへの質問の返事を、反応を見たときと同じ条件で判定するために使う */
  componentBounds?: ComponentBoundsInput[];
}

export interface ChatResponse {
  intent: ChatIntent;
  reply: string;
  suggestedFollowUps: string[];
  applied?: ApplyResponse;
  /** 返答の中で画面上の部品を指し示しているときの部品IDリスト */
  pointedComponentIds?: string[];
  /** intent が customer で選んだ客層のお客さんたちが順番に返答するときのリスト */
  customerReplies?: Array<{ personaId: string; text: string; pointedComponentIds?: string[] }>;
}

export interface TtsRequest {
  text: string;
  /** 声の話し手（assistant / owner / ペルソナID） */
  speaker: string;
}

export interface TtsResponse {
  audioContent?: string;
  fallbackBrowserTts?: boolean;
}

/** 音声入力（STT）をノイズ除去・赤ペン対象補足して入力欄用に整形するリクエスト */
export interface TranscribeRequest {
  audioBase64: string;
  audioMimeType?: string;
  currentSpec: AppSpec;
  targetComponentIds?: string[];
  annotatedImageBase64?: string;
}

export interface TranscribeResponse {
  /** 意味のある発話が含まれていたか（無音・咳払い・マイクテストのみなら false） */
  hasMeaningfulSpeech: boolean;
  /** フィラー・雑音を除き、赤ペン囲み対象（「これ」「ここ」）を補った入力欄用の1文 */
  cleanedText: string;
}

// お題アプリの実行ロジック。画面描画（ブラウザ）と操作シミュレーター（サーバー）で同じ規則を共有する。

import type {
  AppComponent,
  AppSpec,
  ComponentAction,
  StateValue,
  VisibilityCondition,
} from './types.ts';

export type AppState = Record<string, StateValue>;

const UNSAFE_KEYS = new Set(['__proto__', 'constructor', 'prototype', 'toString', 'valueOf']);

/** プロトタイプ汚染を防ぐため、危険なオブジェクトキー名を拒否する */
export function isSafeStateKey(key: string): boolean {
  return Boolean(key) && !UNSAFE_KEYS.has(key);
}

/** 部品IDとして安全か（空でなく長すぎず、プロトタイプ汚染キーでない） */
export function isSafeComponentId(id: unknown): id is string {
  return typeof id === 'string' && id.length > 0 && id.length <= 80 && isSafeStateKey(id);
}

function readStateValue(state: AppState, key: string): StateValue | undefined {
  return isSafeStateKey(key) && Object.hasOwn(state, key) ? state[key] : undefined;
}

// ============================================================================
// 計算式（合計金額など）
// 四則演算・比較・論理演算・三項演算子だけを扱う小さな式評価器。eval は使わない。
// ============================================================================

type Token =
  | { kind: 'num'; value: number }
  | { kind: 'str'; value: string }
  | { kind: 'id'; value: string }
  | { kind: 'op'; value: string };

const OPERATORS = ['>=', '<=', '==', '!=', '&&', '||', '+', '-', '*', '/', '>', '<', '?', ':', '(', ')', '!'];

function tokenize(expression: string): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < expression.length) {
    const ch = expression[i];
    if (/\s/.test(ch)) {
      i++;
      continue;
    }
    if (/[0-9.]/.test(ch)) {
      let j = i;
      while (j < expression.length && /[0-9.]/.test(expression[j])) j++;
      tokens.push({ kind: 'num', value: Number(expression.slice(i, j)) });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(ch)) {
      let j = i;
      while (j < expression.length && /[A-Za-z0-9_]/.test(expression[j])) j++;
      tokens.push({ kind: 'id', value: expression.slice(i, j) });
      i = j;
      continue;
    }
    if (ch === "'" || ch === '"') {
      const end = expression.indexOf(ch, i + 1);
      if (end === -1) throw new Error(`文字列が閉じていません: ${expression}`);
      tokens.push({ kind: 'str', value: expression.slice(i + 1, end) });
      i = end + 1;
      continue;
    }
    const op = OPERATORS.find((o) => expression.startsWith(o, i));
    if (!op) throw new Error(`計算式に使えない文字があります: ${ch}`);
    tokens.push({ kind: 'op', value: op });
    i += op.length;
  }
  return tokens;
}

function toNumber(v: StateValue): number {
  if (typeof v === 'boolean') return v ? 1 : 0;
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function toBool(v: StateValue): boolean {
  if (typeof v === 'string') return v !== '' && v !== 'false' && v !== '0';
  return Boolean(v);
}

const MAX_EXPR_TOKENS = 256;
const MAX_EXPR_DEPTH = 32;

/** 計算式を評価する。未知の状態キーは 0 として扱う */
export function evaluateExpression(expression: string, state: AppState): StateValue {
  if (expression.length > 1000) {
    throw new Error(`計算式が長すぎます: ${expression.slice(0, 40)}`);
  }
  const tokens = tokenize(expression);
  if (tokens.length > MAX_EXPR_TOKENS) {
    throw new Error(`計算式が複雑すぎます: ${expression.slice(0, 40)}`);
  }
  let pos = 0;

  const peek = (): Token | undefined => tokens[pos];
  const takeOp = (value: string): boolean => {
    const t = tokens[pos];
    if (t?.kind === 'op' && t.value === value) {
      pos++;
      return true;
    }
    return false;
  };

  const checkDepth = (depth: number): void => {
    if (depth > MAX_EXPR_DEPTH) {
      throw new Error(`計算式の入れ子が深すぎます: ${expression.slice(0, 40)}`);
    }
  };

  const parsePrimary = (depth: number): StateValue => {
    checkDepth(depth);
    const t = tokens[pos++];
    if (!t) throw new Error(`計算式が途中で終わっています: ${expression}`);
    if (t.kind === 'num' || t.kind === 'str') return t.value;
    if (t.kind === 'id') {
      if (t.value === 'true') return true;
      if (t.value === 'false') return false;
      return readStateValue(state, t.value) ?? 0;
    }
    if (t.value === '(') {
      const v = parseTernary(depth + 1);
      if (!takeOp(')')) throw new Error(`括弧が閉じていません: ${expression}`);
      return v;
    }
    if (t.value === '-') return -toNumber(parsePrimary(depth + 1));
    if (t.value === '!') return !toBool(parsePrimary(depth + 1));
    throw new Error(`計算式を読めません: ${expression}`);
  };

  const parseMul = (depth: number): StateValue => {
    let left = parsePrimary(depth);
    for (;;) {
      if (takeOp('*')) left = toNumber(left) * toNumber(parsePrimary(depth));
      else if (takeOp('/')) {
        const right = toNumber(parsePrimary(depth));
        left = right === 0 ? 0 : toNumber(left) / right;
      } else return left;
    }
  };

  const parseAdd = (depth: number): StateValue => {
    let left = parseMul(depth);
    for (;;) {
      if (takeOp('+')) left = toNumber(left) + toNumber(parseMul(depth));
      else if (takeOp('-')) left = toNumber(left) - toNumber(parseMul(depth));
      else return left;
    }
  };

  const parseCompare = (depth: number): StateValue => {
    const left = parseAdd(depth);
    const t = peek();
    if (t?.kind !== 'op') return left;
    if (['>=', '<=', '>', '<'].includes(t.value)) {
      pos++;
      const a = toNumber(left);
      const b = toNumber(parseAdd(depth));
      if (t.value === '>=') return a >= b;
      if (t.value === '<=') return a <= b;
      if (t.value === '>') return a > b;
      return a < b;
    }
    if (t.value === '==' || t.value === '!=') {
      pos++;
      const right = parseAdd(depth);
      const equal = String(left) === String(right);
      return t.value === '==' ? equal : !equal;
    }
    return left;
  };

  const parseAnd = (depth: number): StateValue => {
    let left = parseCompare(depth);
    while (takeOp('&&')) left = toBool(left) && toBool(parseCompare(depth));
    return left;
  };

  const parseOr = (depth: number): StateValue => {
    let left = parseAnd(depth);
    while (takeOp('||')) left = toBool(left) || toBool(parseAnd(depth));
    return left;
  };

  const parseTernary = (depth: number): StateValue => {
    checkDepth(depth);
    const cond = parseOr(depth);
    if (!takeOp('?')) return cond;
    const whenTrue = parseTernary(depth + 1);
    if (!takeOp(':')) throw new Error(`三項演算子の : がありません: ${expression}`);
    const whenFalse = parseTernary(depth + 1);
    return toBool(cond) ? whenTrue : whenFalse;
  };

  const result = parseTernary(0);
  if (pos !== tokens.length) throw new Error(`計算式の末尾を読めません: ${expression}`);
  return result;
}

/**
 * 状態に計算値（derived）を書き足した状態を返す。
 * 後から追加された計算値（codFee など）を先に定義された計算値（total など）が参照していても正しく求まるよう、
 * 値が変わらなくなるまで繰り返し計算する。式が壊れている値は 0 にする。
 */
export function withDerivedValues(spec: AppSpec, state: AppState): AppState {
  const next: AppState = { ...state };
  const derivedList = spec.derived ?? [];
  if (derivedList.length === 0) return next;

  const maxPasses = Math.min(8, derivedList.length + 1);
  for (let pass = 0; pass < maxPasses; pass++) {
    let changed = false;
    for (const d of derivedList) {
      if (!isSafeStateKey(d.key)) continue;
      let val: StateValue = 0;
      try {
        val = evaluateExpression(d.expression, next);
      } catch {
        val = 0;
      }
      if (next[d.key] !== val) {
        next[d.key] = val;
        changed = true;
      }
    }
    if (!changed) break;
  }
  return next;
}

/** 文字で書かれた値を、状態の型（数値・真偽・文字）に合わせて直す */
export function coerceStateValue(spec: AppSpec, key: string, raw: string): StateValue {
  const def = spec.state.find((s) => s.key === key);
  if (def?.type === 'number') {
    const n = Number(raw);
    return Number.isFinite(n) ? n : 0;
  }
  if (def?.type === 'boolean') return raw === 'true';
  return raw;
}

/** 独自の部品の data-set-state（`キー:値;キー:値`）を、set_state の動きの並びに直す。安全でないキーは捨てる */
export function parseSetStateAttribute(spec: AppSpec, raw: string): ComponentAction[] {
  const acts: ComponentAction[] = [];
  for (const pair of raw.split(';')) {
    const eqIdx = pair.indexOf(':');
    if (eqIdx <= 0) continue;
    const k = pair.slice(0, eqIdx).trim();
    const v = pair.slice(eqIdx + 1).trim();
    if (isSafeStateKey(k)) {
      acts.push({ type: 'set_state', targetKey: k, value: coerceStateValue(spec, k, v) });
    }
  }
  return acts;
}

/** 独自の部品の data-* 属性から、実行するアクションの並びを組み立てる（描画とシミュレーターで共通） */
export function buildActionsFromDataAttrs(
  spec: AppSpec,
  attrs: {
    setState?: string;
    toggleState?: string;
    incrementState?: string;
    toast?: string;
    completeTask?: string;
  },
): ComponentAction[] {
  const acts: ComponentAction[] = [];
  if (attrs.setState) {
    acts.push(...parseSetStateAttribute(spec, attrs.setState));
  }
  if (attrs.toggleState) {
    const k = attrs.toggleState.trim();
    if (isSafeStateKey(k)) {
      acts.push({ type: 'toggle_state', targetKey: k });
    }
  }
  if (attrs.incrementState) {
    const k = attrs.incrementState.trim();
    if (isSafeStateKey(k)) {
      acts.push({ type: 'increment_state', targetKey: k });
    }
  }
  if (attrs.toast) {
    acts.push({ type: 'show_toast', message: attrs.toast });
  }
  if (attrs.completeTask !== undefined) {
    acts.push({
      type: 'complete_task',
      message: attrs.completeTask || 'ご購入ありがとうございます',
    });
  }
  return acts;
}

export function createInitialState(spec: AppSpec): AppState {
  const state: AppState = {};
  for (const s of spec.state) {
    if (isSafeStateKey(s.key)) {
      state[s.key] = s.initialValue;
    }
  }
  return withDerivedValues(spec, state);
}

// ============================================================================
// 表示条件・入力判定
// ============================================================================

export function matchesCondition(cond: VisibilityCondition, state: AppState): boolean {
  const current = readStateValue(state, cond.stateKey);
  switch (cond.operator) {
    case 'eq':
      return String(current) === String(cond.value);
    case 'neq':
      return String(current) !== String(cond.value);
    case 'gt':
      return Number(current) > Number(cond.value);
    case 'lt':
      return Number(current) < Number(cond.value);
    default:
      return true;
  }
}

export function isComponentVisible(comp: AppComponent, state: AppState): boolean {
  return comp.visibleWhen ? matchesCondition(comp.visibleWhen, state) : true;
}

/** required_input の部品に値が入っているか。空か placeholder と同じなら未入力とみなす */
export function isInputFilled(comp: AppComponent, state: AppState): boolean {
  if (!comp.stateKey) return true;
  const v = readStateValue(state, comp.stateKey);
  if (v === undefined || v === '' || v === false) return false;
  return comp.placeholder === undefined || String(v) !== comp.placeholder;
}

// ============================================================================
// アクション
// ============================================================================

export interface ActionResult {
  state: AppState;
  completed: boolean;
  /** open_page で飛ばされた先のページの見出し */
  openedPage?: string;
  messages: Array<{ text: string; type: 'info' | 'success' | 'warning' }>;
}

export function applyActions(
  spec: AppSpec,
  state: AppState,
  actions: ComponentAction[],
): ActionResult {
  let next: AppState = { ...state };
  let completed = false;
  let openedPage: string | undefined;
  const messages: ActionResult['messages'] = [];

  for (const act of actions) {
    if (act.type === 'set_state' && act.targetKey && isSafeStateKey(act.targetKey)) {
      next[act.targetKey] = act.value ?? true;
    } else if (act.type === 'toggle_state' && act.targetKey && isSafeStateKey(act.targetKey)) {
      next[act.targetKey] = !toBool(readStateValue(next, act.targetKey) ?? false);
    } else if (act.type === 'increment_state' && act.targetKey && isSafeStateKey(act.targetKey)) {
      next[act.targetKey] = toNumber(readStateValue(next, act.targetKey) ?? 0) + 1;
    } else if (act.type === 'reset_all') {
      next = createInitialState(spec);
    } else if (act.type === 'show_toast') {
      messages.push({ text: act.message || '実行しました', type: 'warning' });
    } else if (act.type === 'open_page') {
      openedPage = act.message || '別のページ';
      const pageLabel = openedPage.endsWith('ページ') ? openedPage : `${openedPage}のページ`;
      messages.push({ text: `${pageLabel}に移動してしまいました`, type: 'warning' });
    } else if (act.type === 'complete_task') {
      completed = true;
      messages.push({ text: act.message || '完了しました', type: 'success' });
    }
  }

  return { state: withDerivedValues(spec, next), completed, openedPage, messages };
}

export function hasCompletionAction(comp: AppComponent): boolean {
  return comp.actions.some((a) => a.type === 'complete_task');
}

/**
 * 定期便などのチェック欄が「最初からOFF・読める大きさ・不当な解約縛り文言なし」の正直な任意選択欄に整っているか
 */
export function isHonestOptInComponent(
  comp: AppComponent | undefined,
  initialValue: StateValue | undefined,
): boolean {
  return Boolean(
    comp &&
      initialValue === false &&
      comp.style.size !== 'xs' &&
      !comp.label.includes('解約は電話のみ') &&
      !comp.label.includes('解約不可'),
  );
}

// ============================================================================
// 表示用の文字
// ============================================================================

const KANJI_DIGITS = ['', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
const KANJI_SMALL_UNITS = ['', '十', '百', '千'];
const KANJI_LARGE_UNITS = ['', '万', '億', '兆'];

/** 整数を漢数字にする（例: 3960 → 三千九百六十、12000 → 一万二千） */
export function toKanjiNumber(value: number): string {
  let n = Math.round(Math.abs(value));
  if (n === 0) return '零';
  const prefix = value < 0 ? 'マイナス' : '';
  let result = '';
  let groupIndex = 0;
  while (n > 0 && groupIndex < KANJI_LARGE_UNITS.length) {
    const group = n % 10000;
    if (group > 0) {
      let groupText = '';
      let g = group;
      for (let unit = 0; unit < 4 && g > 0; unit++) {
        const digit = g % 10;
        if (digit > 0) {
          // 十・百・千の位の「一」は省く（一千ではなく千）
          const digitText = digit === 1 && unit > 0 ? '' : KANJI_DIGITS[digit];
          groupText = digitText + KANJI_SMALL_UNITS[unit] + groupText;
        }
        g = Math.floor(g / 10);
      }
      result = groupText + KANJI_LARGE_UNITS[groupIndex] + result;
    }
    n = Math.floor(n / 10000);
    groupIndex++;
  }
  return prefix + result;
}

function formatNumber(value: number, format: 'arabic' | 'kanji' | undefined): string {
  if (format === 'kanji') return toKanjiNumber(value);
  return Math.round(value).toLocaleString('ja-JP');
}

/**
 * ラベルの {countdown} は、画面では1秒ずつ減っては戻る偽のカウントダウンになる。
 * 画面以外（操作シミュレーターやAIに渡す文字、スクリーンショット）では最初の値で表す
 */
export const COUNTDOWN_INITIAL_TEXT = '2:59';

/** 文字列内の {状態キー} を現在の値に置き換える */
export function renderTemplate(
  text: string,
  state: AppState,
  numberFormat?: 'arabic' | 'kanji',
): string {
  return text.replace(/\{([A-Za-z0-9_]+)\}/g, (_, key: string) => {
    const v = readStateValue(state, key);
    if (typeof v === 'number') return formatNumber(v, numberFormat);
    if (typeof v === 'string' && v !== '' && Number.isFinite(Number(v))) {
      return formatNumber(Number(v), numberFormat);
    }
    if (v === undefined) return key === 'countdown' ? COUNTDOWN_INITIAL_TEXT : '';
    return String(v);
  });
}

/** 部品のラベル内の {状態キー} を現在の値に置き換える */
export function renderLabel(comp: AppComponent, state: AppState): string {
  return renderTemplate(comp.label, state, comp.style.numberFormat);
}

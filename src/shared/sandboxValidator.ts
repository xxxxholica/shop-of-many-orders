import {
  applyActions,
  createInitialState,
  evaluateExpression,
  hasCompletionAction,
  isComponentVisible,
  isHonestOptInComponent,
  isInputFilled,
  matchesCondition,
  renderLabel,
  withDerivedValues,
  type AppState,
} from './appRuntime.ts';
import {
  parseCustomHtmlActions,
  TAP_HEIGHT_BY_SIZE as HEIGHT_BY_SIZE,
} from './customComponentProfile.ts';
import type {
  AppComponent,
  AppSpec,
  ComponentAction,
  ComponentBoundsInput,
  ComponentSize,
  ReviewerEvaluation,
  ReviewerPersona,
  ReviewOutcome,
  SandboxValidationResult,
} from './types.ts';

export function escapeHtml(raw: string): string {
  return raw
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

const SAFE_CSS_COLOR_RE =
  /^(#[0-9a-fA-F]{3,8}|[a-zA-Z]+|(?:rgb|hsl|oklch)a?\([0-9.,%\s/deg+-]+\))$/;

export function sanitizeCssColor(raw: string | undefined, fallback: string): string {
  const trimmed = raw?.trim() ?? '';
  return SAFE_CSS_COLOR_RE.test(trimmed) ? trimmed : fallback;
}

export const FORBIDDEN_CUSTOM_HTML_TAGS = [
  'script',
  'style',
  'iframe',
  'frame',
  'frameset',
  'object',
  'embed',
  'applet',
  'meta',
  'link',
  'base',
  'form',
  'noscript',
  'template',
  'foreignObject',
  'animate',
  'animateMotion',
  'animateTransform',
  'set',
  'use',
  'feImage',
  'math',
  'portal',
  'audio',
  'video',
  'source',
  'track',
  'marquee',
  'dialog',
] as const;

const FORBIDDEN_TAG_PAIR_RE = new RegExp(
  `<(${FORBIDDEN_CUSTOM_HTML_TAGS.join('|')})\\b[\\s\\S]*?<\\/\\1\\s*>`,
  'gi',
);
const FORBIDDEN_TAG_SINGLE_RE = new RegExp(
  `<\\/?(${FORBIDDEN_CUSTOM_HTML_TAGS.join('|')})\\b[^>]*\\/?>`,
  'gi',
);
const FORBIDDEN_TAG_UNCLOSED_RE = new RegExp(
  `<\\/?(${FORBIDDEN_CUSTOM_HTML_TAGS.join('|')})\\b[\\s\\S]*$`,
  'gi',
);

const FORBIDDEN_CUSTOM_HTML_ATTRS = new Set([
  'srcdoc',
  'ping',
  'srcset',
  'xlink:href',
  'action',
  'formaction',
  'form',
  'is',
  'popover',
  'popovertarget',
  'command',
  'commandfor',
  'autofocus',
  'poster',
  'background',
  'data-component-id',
  'data-spec-id',
]);

function decodeHtmlEntitiesForAttrCheck(raw: string): string {
  return raw
    .replace(/&#x([0-9a-f]+);?/gi, (_m, hex: string) => {
      const code = Number.parseInt(hex, 16);
      return Number.isFinite(code) && code >= 0 && code <= 0x10ffff
        ? String.fromCodePoint(code)
        : '';
    })
    .replace(/&#([0-9]+);?/g, (_m, dec: string) => {
      const code = Number.parseInt(dec, 10);
      return Number.isFinite(code) && code >= 0 && code <= 0x10ffff
        ? String.fromCodePoint(code)
        : '';
    })
    .replace(/&colon;?/gi, ':')
    .replace(/&(tab|newline);?/gi, '')
    .replace(/[\s\u0000-\u001f]+/g, '');
}

export function isSafeDataImageUrl(raw: string): boolean {
  const cleaned = decodeHtmlEntitiesForAttrCheck(raw);
  return /^data:image\/(?:png|jpeg|jpg|gif|webp);base64,[a-z0-9+/=]+$/i.test(cleaned);
}

export function isForbiddenCustomHtmlAttr(attrName: string, attrValue: string): boolean {
  const lower = attrName.toLowerCase();
  if (lower.startsWith('on') || FORBIDDEN_CUSTOM_HTML_ATTRS.has(lower)) {
    return true;
  }
  if (lower === 'src') {
    return !isSafeDataImageUrl(attrValue);
  }
  if (lower === 'href') {
    const cleaned = decodeHtmlEntitiesForAttrCheck(attrValue);
    return !cleaned.startsWith('#');
  }
  return false;
}

/**
 * カスタムCSS文字列から外部通信やスクリプト実行につながる危険な記述を取り除く。
 */
export function sanitizeCustomCss(raw: string | undefined): string {
  if (!raw) return '';
  return raw
    .slice(0, 4000)
    .replace(/\/\*[\s\S]*?(?:\*\/|$)/g, '')
    .replace(/\\/g, '')
    .replace(/@(?:import|charset|namespace)\b[^;{}]*;?/gi, '')
    .replace(/url\s*\([^)]*\)?/gi, 'none')
    .replace(/(?:-[a-z]+-)?image-set\s*\([^)]*\)?/gi, 'none')
    .replace(/expression\s*\([^)]*\)?/gi, 'none')
    .replace(/(?:behavior|-moz-binding)\s*:[^;{}]+;?/gi, '')
    .replace(/position\s*:\s*fixed\b/gi, 'position: relative')
    .replace(/(?:javascript|vbscript|data)\s*:/gi, '')
    .replace(/[<>]/g, '')
    .trim();
}

const HTML_OPEN_TAG_RE = /<([a-zA-Z][a-zA-Z0-9:-]*)((?:"[^"]*"|'[^']*'|[^>"'])*?)(\/?>)/g;
const HTML_ATTR_TOKEN_RE =
  /(?:\/+\s*)?([a-zA-Z_:][a-zA-Z0-9_.:-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?/g;

/**
 * カスタムHTML文字列からスクリプトや危険な埋め込みタグ・イベント属性を取り除く。
 */
export function sanitizeCustomHtml(raw: string): string {
  let html = raw
    .slice(0, 8000)
    .replace(/<!--[\s\S]*?(?:-->|$)/g, '')
    .replace(/<[a-zA-Z/!?][^>]*$/, '');

  // 入れ子になった偽装タグ（例: <scr<script></script>ipt>）が残らないよう収束するまで除去する
  for (let pass = 0; pass < 16; pass++) {
    const next = html
      .replace(FORBIDDEN_TAG_PAIR_RE, '')
      .replace(FORBIDDEN_TAG_SINGLE_RE, '')
      .replace(FORBIDDEN_TAG_UNCLOSED_RE, '');
    if (next === html) break;
    html = next;
  }

  return html.replace(
    HTML_OPEN_TAG_RE,
    (_full, tagName: string, attrChunk: string, closeToken: string) => {
      if (!attrChunk) return `<${tagName}${closeToken}`;
      const cleanedAttrs = attrChunk.replace(
        HTML_ATTR_TOKEN_RE,
        (
          attrMatch: string,
          attrName: string,
          dqVal: string | undefined,
          sqVal: string | undefined,
          uqVal: string | undefined,
        ) => {
          const val = dqVal ?? sqVal ?? uqVal ?? '';
          if (attrName.toLowerCase() === 'style') {
            const cleanedCss = sanitizeCustomCss(val);
            return cleanedCss ? ` style="${escapeHtml(cleanedCss)}"` : '';
          }
          if (isForbiddenCustomHtmlAttr(attrName, val)) {
            return '';
          }
          return attrMatch;
        },
      );
      return `<${tagName}${cleanedAttrs}${closeToken}`;
    },
  );
}

/**
 * お題アプリ仕様 (AppSpec) から、iframe 内で単一HTMLとして独立動作する HTML 文字列を生成する。
 */
export function buildStandaloneSandboxHtml(spec: AppSpec): string {
  const safeTitle = escapeHtml(spec.title || 'お題アプリ');
  const bg = sanitizeCssColor(spec.theme?.bgColor, '#ffffff');
  const fg = sanitizeCssColor(spec.theme?.textColor, '#1c1917');

  return `<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${safeTitle}</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    /* スマホ画面の幅に合わせて文字の基準を伸び縮みさせ、端末が大きく表示されたら中身も大きくする */
    html { font-size: 4.2vw; }
    html, body {
      width: 100%;
      height: 100%;
      font-family: "Zen Maru Gothic", "Hiragino Maru Gothic ProN", "BIZ UDPGothic", "Hiragino Sans", "Noto Sans JP", sans-serif;
      font-weight: 700;
      background: ${bg};
      color: ${fg};
      overflow: hidden;
    }
    #sandbox-mount {
      width: 100%;
      height: 100%;
    }
  </style>
</head>
<body>
  <div id="sandbox-mount" data-spec-id="${escapeHtml(spec.id || 'app')}"></div>
</body>
</html>`;
}

export interface ValidateSandboxOptions {
  requireTaskCompletionEvent?: boolean;
  previousSpec?: AppSpec;
  touchedComponentIds?: ReadonlySet<string>;
}

/**
 * 改修後の AppSpec が単一HTMLとして描画可能か、タスク完了イベントが残っているか、
 * またパッチ対象外の部品が勝手に書き換わっていないかを検証する。
 */
export function validateSandboxSpec(
  spec: AppSpec,
  options: ValidateSandboxOptions = {},
): SandboxValidationResult {
  const errors: string[] = [];

  if (!spec || typeof spec !== 'object') {
    return {
      valid: false,
      canRenderHtml: false,
      hasTaskCompletionEvent: false,
      errors: ['アプリ仕様データが空または不正です'],
    };
  }

  if (!Array.isArray(spec.components) || spec.components.length === 0) {
    errors.push('画面に表示する部品が1つもありません');
  }

  if (!Array.isArray(spec.state) || spec.state.length === 0) {
    errors.push('状態データ定義が失われています');
  }

  // 単一HTML生成テスト
  let canRenderHtml = false;
  try {
    const html = buildStandaloneSandboxHtml(spec);
    canRenderHtml = html.includes('<!DOCTYPE html>') && html.includes('sandbox-mount');
  } catch {
    canRenderHtml = false;
    errors.push('単一HTMLの組み立てに失敗しました');
  }

  // タスク完了イベント（complete_task を持つボタン、または data-complete-task を持つカスタム部品）が存在するかチェック
  const hasTaskCompletionEvent = Array.isArray(spec.components)
    ? spec.components.some(
        (c) =>
          (Array.isArray(c.actions) && hasCompletionAction(c)) ||
          Boolean(c.customHtml?.includes('data-complete-task')),
      )
    : false;

  if (options.requireTaskCompletionEvent && !hasTaskCompletionEvent) {
    errors.push('タスク完了イベント complete_task を持つボタンが見つかりません');
  }

  // 計算式が読めるか・部品が参照している状態キーが存在するか
  if (Array.isArray(spec.state)) {
    const state = Object.fromEntries(spec.state.map((v) => [v.key, v.initialValue]));
    const knownKeys = new Set<string>(spec.state.map((v) => v.key));
    for (const d of spec.derived ?? []) {
      knownKeys.add(d.key);
      try {
        evaluateExpression(d.expression, state);
      } catch {
        errors.push(`計算式 ${d.key} を読めません`);
      }
    }
    for (const comp of spec.components ?? []) {
      if (comp.stateKey && !knownKeys.has(comp.stateKey)) {
        errors.push(`部品 ${comp.id} が参照する状態キー ${comp.stateKey} が定義されていません`);
      }
      if (comp.visibleWhen?.stateKey && !knownKeys.has(comp.visibleWhen.stateKey)) {
        errors.push(`部品 ${comp.id} の表示条件キー ${comp.visibleWhen.stateKey} が定義されていません`);
      }
      for (const row of comp.rows ?? []) {
        if (row.visibleWhen?.stateKey && !knownKeys.has(row.visibleWhen.stateKey)) {
          errors.push(`部品 ${comp.id} の内訳の行の表示条件キー ${row.visibleWhen.stateKey} が定義されていません`);
        }
      }
    }
  }

  // パッチ対象外の部品が変わっていないかチェック
  if (options.previousSpec && options.touchedComponentIds && Array.isArray(spec.components)) {
    for (const prevComp of options.previousSpec.components) {
      if (options.touchedComponentIds.has(prevComp.id)) continue;
      const nextComp = spec.components.find((c) => c.id === prevComp.id);
      if (!nextComp || JSON.stringify(nextComp) !== JSON.stringify(prevComp)) {
        errors.push(`変更対象外の部品 ${prevComp.id} が書き換わっています`);
      }
    }
  }

  return {
    valid: errors.length === 0 && canRenderHtml,
    canRenderHtml,
    hasTaskCompletionEvent,
    errors,
  };
}


// ============================================================================
// 操作シミュレーター
// ============================================================================

export interface ComputerUseSimulationResult {
  outcome: ReviewOutcome;
  frictionPoint: string;
  simulatedComment: string;
  actionTrace: string[];
  appliedConstraintNote: string;
  /** つまずき・不満・後悔の原因になった部品のID一覧 */
  frictionComponentIds: string[];
  /** 完了したときの状態（定期便に入ったままか等、お店の側の集計に使う） */
  finalState?: AppState;
}

interface SimulatedBox {
  comp: AppComponent;
  left: number;
  top: number;
  right: number;
  bottom: number;
  centerX: number;
  centerY: number;
  width: number;
  height: number;
}

const SCREEN_WIDTH = 320;
const SIZE_RANK: Record<ComponentSize, number> = { xs: 0, sm: 1, md: 2, lg: 3, xl: 4 };
/** 片手の親指が届く横位置の上限 */
const ONE_HAND_REACH_X = 230;
/** 1手あたりの操作時間・邪魔な表示1つあたり・漢数字を読む時間・ポップアップを閉じる時間（ミリ秒） */
const MS_PER_STEP = 1200;
const MS_PER_DISTRACTION = 700;
const MS_KANJI_READING = 2000;
const MS_POPUP = 1500;
const MAX_SCREENS = 6;

/** 部品の種類ごとの高さの見積もり（px）。種類で決まらないものは文字サイズで決める */
function estimateHeight(comp: AppComponent): number {
  switch (comp.type) {
    case 'product_card':
      return 128;
    case 'choice_cards':
      return 16 + 60 * Math.max(1, comp.options?.length ?? 1);
    case 'summary':
      return 24 + 26 * Math.max(1, comp.rows?.length ?? 1);
    case 'progress_steps':
      return 28;
    default:
      return HEIGHT_BY_SIZE[comp.style.size] ?? 42;
  }
}

/**
 * 画面上の各部品の擬似バウンディングボックスを算出する（実DOM座標があればそれを使用）。
 * ポップアップは画面に重なるため並びには含めない。sticky の部品は画面下のバーに横並びで置く。
 */
function layoutComponentBoxes(
  components: AppComponent[],
  customBounds?: ComponentBoundsInput[],
): SimulatedBox[] {
  const customMap = new Map((customBounds ?? []).map((b) => [b.id, b]));
  const boxes: SimulatedBox[] = [];
  const toBox = (comp: AppComponent, left: number, top: number, w: number, h: number): SimulatedBox => ({
    comp,
    left,
    top,
    right: left + w,
    bottom: top + h,
    centerX: left + w / 2,
    centerY: top + h / 2,
    width: w,
    height: h,
  });
  let cursorY = 58;

  const flow = components.filter((c) => c.type !== 'popup' && !c.style.sticky);
  const sticky = components.filter((c) => c.type !== 'popup' && c.style.sticky);

  for (const comp of [...flow, ...sticky]) {
    const existing = customMap.get(comp.id);
    if (existing && existing.right > existing.left && existing.bottom > existing.top) {
      boxes.push(
        toBox(comp, existing.left, existing.top, existing.right - existing.left, existing.bottom - existing.top),
      );
      continue;
    }
    if (comp.style.sticky) continue;

    const h = estimateHeight(comp);
    const isFull =
      comp.style.fullWidth ||
      comp.style.size === 'lg' ||
      comp.style.size === 'xl' ||
      ['product_card', 'choice_cards', 'summary', 'progress_steps'].includes(comp.type);
    const w = isFull ? SCREEN_WIDTH - 32 : comp.style.size === 'xs' ? 92 : 160;
    let left = 16;
    if (!isFull) {
      if (comp.style.align === 'right') {
        left = SCREEN_WIDTH - 16 - w;
      } else if (comp.style.align === 'center') {
        left = (SCREEN_WIDTH - w) / 2;
      }
    }
    boxes.push(toBox(comp, left, cursorY, w, h));
    cursorY += h + 8;
  }

  // 画面下に固定したバー：合計などの表示を左に、ボタンを右に並べる
  const stickyWithoutBounds = sticky.filter((c) => !customMap.has(c.id));
  if (stickyWithoutBounds.length > 0) {
    const barTop = cursorY + 8;
    const buttons = stickyWithoutBounds.filter((c) => c.type === 'button');
    const others = stickyWithoutBounds.filter((c) => c.type !== 'button');
    const othersWidth = others.length > 0 ? 110 : 0;
    others.forEach((comp, i) => {
      const w = othersWidth / others.length;
      boxes.push(toBox(comp, 16 + i * w, barTop, w, 56));
    });
    const buttonsLeft = 16 + othersWidth + (others.length > 0 ? 12 : 0);
    const buttonsWidth = SCREEN_WIDTH - 16 - buttonsLeft;
    buttons.forEach((comp, i) => {
      const w = buttonsWidth / buttons.length;
      boxes.push(toBox(comp, buttonsLeft + i * w, barTop, w, 54));
    });
  }

  return boxes;
}

/**
 * ペルソナIDと手数から決定論的なズレ量（-jitterPx 〜 +jitterPx）を算出する。
 * テストの再現性を保ちつつ、指の震え・片手操作・電車の揺れによる誤タップを再現する。
 */
function computeDeterministicJitter(
  personaId: string,
  step: number,
  jitterPx: number,
): { dx: number; dy: number } {
  let hash = step * 31;
  for (let i = 0; i < personaId.length; i++) {
    hash = (hash * 33 + personaId.charCodeAt(i)) % 997;
  }
  const normX = ((hash % 200) - 100) / 100;
  const normY = (((hash * 7) % 200) - 100) / 100;
  // 指のズレが大きいペルソナほど端方向にズレが出るように最低65%の振幅を持たせる
  const magX = Math.max(0.65, Math.abs(normX)) * jitterPx * (normX >= 0 ? 1 : -1);
  const magY = Math.max(0.65, Math.abs(normY)) * jitterPx * (normY >= 0 ? 1 : -1);
  return { dx: Math.round(magX), dy: Math.round(magY) };
}

/** 視力制約から、読み取れる最小の文字サイズを決める */
function minReadableRank(persona: ReviewerPersona): number {
  const blur = persona.constraints.blurPx;
  if (blur >= 3) return SIZE_RANK.lg;
  if (blur >= 2) return SIZE_RANK.md;
  if (blur >= 1.2) return SIZE_RANK.sm;
  return SIZE_RANK.xs;
}

function canRead(comp: AppComponent, persona: ReviewerPersona): boolean {
  let rank = SIZE_RANK[comp.style.size] ?? SIZE_RANK.md;
  // 薄い色の文字は、コントラストが落ちている人には一段小さく見える
  if (comp.style.variant === 'muted' && persona.constraints.contrast < 0.8) rank -= 1;
  return rank >= minReadableRank(persona);
}

/** 目立ち具合。読めないときや細かく読まない人は、一番目立つボタンを押す */
function prominence(comp: AppComponent): number {
  const s = comp.style;
  return (
    (SIZE_RANK[s.size] ?? 2) * 10 +
    (['primary', 'danger', 'warning'].includes(s.variant) ? 4 : 0) +
    (s.fullWidth ? 2 : 0) +
    (s.noisyAnimation ? 3 : 0)
  );
}

type FailureReason =
  | 'popup_unclosable'
  | 'popup_tapped'
  | 'clutter'
  | 'unreadable_input'
  | 'unreachable'
  | 'hit_danger'
  | 'tap_target_too_small'
  | 'left_page'
  | 'decoy_only'
  | 'no_response'
  | 'no_target'
  | 'too_many_steps'
  | 'timeout'
  | 'no_confirmation'
  | 'total_unreadable';

const FAILURE_TEXT: Record<FailureReason, { friction: string; comment: string }> = {
  popup_unclosable: {
    friction: '広告の閉じるボタンが見えない',
    comment: '広告の閉じるボタンが見えなくて、先に進めなかった',
  },
  popup_tapped: {
    friction: '広告を閉じようとして押してしまった',
    comment: '広告を閉じようとしたら、広告のほうを押してしまった',
  },
  clutter: {
    friction: '関係ない表示が多くて迷った',
    comment: '関係ない表示が多すぎて、どこを見ればいいのか分からなかった',
  },
  unreadable_input: {
    friction: '入力欄の文字が読めない',
    comment: '入力欄の文字が小さくて、何を選べばいいのか読めなかった',
  },
  unreachable: {
    friction: 'ボタンに指が届かない',
    comment: '片手だとボタンに指が届かなかった',
  },
  hit_danger: {
    friction: '押し間違えて入力が消えた',
    comment: '押し間違えて、入力した内容が全部消えてしまった',
  },
  tap_target_too_small: {
    friction: 'ボタンや選択欄が小さくて押せない',
    comment: '指がズレてうまく押せなくて、あきらめた',
  },
  left_page: {
    friction: '別のページに飛ばされた',
    comment: 'ボタンを押したら別のページに飛ばされて、注文が途中で終わった',
  },
  decoy_only: {
    friction: 'それらしいボタンが反応しない',
    comment: '目立つボタンを押しても何も起きなくて、あきらめた',
  },
  no_response: {
    friction: '注文ボタンが反応しない',
    comment: '注文ボタンを押しても何も起きなかった',
  },
  no_target: {
    friction: '注文するボタンが見つからない',
    comment: '注文するボタンがどこにあるのか分からなかった',
  },
  too_many_steps: {
    friction: '手順が多すぎる',
    comment: 'やることが多すぎて、途中で面倒になった',
  },
  timeout: {
    friction: '時間がかかりすぎる',
    comment: '時間がかかりすぎて、途中でやめてしまった',
  },
  no_confirmation: {
    friction: '注文を確定する前に内容を確かめる画面がなかった',
    comment: '確かめられないまま注文するのが怖くて、押せなかった',
  },
  total_unreadable: {
    friction: '合計金額がすぐ分からない',
    comment: '結局いくら払うのかすぐ分からなくて、買うのをやめた',
  },
};

function describeConstraints(persona: ReviewerPersona): string {
  const c = persona.constraints;
  return [
    c.blurPx >= 2.0 ? `視力ぼかし${c.blurPx}px` : '',
    c.contrast <= 0.7 ? `低コントラスト${Math.round(c.contrast * 100)}%` : '',
    c.jitterPx >= 15 ? `指のズレ±${c.jitterPx}px` : '',
    `手数上限${c.maxSteps}手`,
    c.environment === 'one_hand' ? '片手親指操作' : c.environment === 'moving_train' ? '揺れる車内' : '',
    c.readsFinePrint ? '細かい字も読む' : '細かい字は読み飛ばす',
    c.wantsConfirmation ? '確認したい' : '',
    c.needsReadableTotal ? '合計金額を確かめたい' : '',
    c.dislikesPressure ? '急かされるのが嫌い' : '',
    c.wantsSubscription ? '安心できる定期便があれば申し込みたい' : '',
    c.wantsCoupon ? '使えるクーポンがあれば使いたい' : '',
  ]
    .filter(Boolean)
    .join('・');
}

export function isSubscriptionComponent(comp: AppComponent): boolean {
  return (
    comp.stateKey === 'subscription' ||
    comp.stateKey === 'subscribe' ||
    comp.id === 'check-subscription' ||
    comp.id === 'sub-trap' ||
    comp.label.includes('定期') ||
    Boolean(comp.customHtml && (comp.customHtml.includes('定期') || comp.customHtml.includes('subscription')))
  );
}

/** 店に最初からクーポンの仕掛け・状態変数があるか */
export function hasCouponFeatureInShop(spec: AppSpec): boolean {
  return spec.state.some((s) => s.key === 'couponPopup' || s.key.toLowerCase().includes('coupon'));
}

/** クーポンを使いたかった人の不満。使えなかったか、使っても安くならなかったか、無くなっていたか */
export const COUPON_FRICTIONS = {
  missing: 'クーポンがなくなっていた',
  unusable: 'クーポンがあったのに使えなかった',
  noDiscount: 'クーポンを使っても安くならなかった',
} as const;

/**
 * クーポンの部品を押したときの動き。独自の部品（customHtml）で動きが部品の外に書かれていないときは、
 * HTMLの data-set-state から読む
 */
function couponActionsOf(spec: AppSpec, comp: AppComponent): ComponentAction[] {
  if (comp.actions.length > 0 || !comp.customHtml) return comp.actions;
  return parseCustomHtmlActions(spec, comp.customHtml);
}

export function isCouponComponent(comp: AppComponent): boolean {
  const text = `${comp.label} ${comp.caption ?? ''} ${comp.customHtml ?? ''}`;
  return (
    comp.id === 'popup-coupon' ||
    comp.id.toLowerCase().includes('coupon') ||
    comp.stateKey === 'couponPopup' ||
    Boolean(comp.stateKey?.toLowerCase().includes('coupon')) ||
    text.includes('クーポン') ||
    text.includes('OFF') ||
    text.includes('割引') ||
    comp.actions.some(
      (a) =>
        Boolean(a.targetKey?.toLowerCase().includes('coupon')) ||
        Boolean(a.targetKey?.toLowerCase().includes('discount')),
    )
  );
}

/**
 * 操作シミュレーター。部品の役割（role）だけを見て、ペルソナの制約のもとで画面を最後まで操作してみる。
 *
 * - 視力制約: 小さい文字・薄い文字を読めず、読めないときは一番目立つボタンを押す
 * - 指の正確さ: 押す位置にズレを加え、外れた先が危険なボタンなら離脱
 * - 手数と時間: 手数上限・待てる時間を超えたら離脱
 * - 罠: 細かい字を読む人だけが気づいて直す。気づかずに完了したら後悔
 * - 完了判定: complete_task が実行されたかどうかをコードで判定する
 */
export function simulatePersonaComputerUse(
  spec: AppSpec,
  persona: ReviewerPersona,
  customBounds?: ComponentBoundsInput[],
): ComputerUseSimulationResult {
  const c = persona.constraints;
  const actionTrace: string[] = [];
  const appliedConstraintNote = describeConstraints(persona);

  let state: AppState = createInitialState(spec);
  let steps = 0;
  let elapsedMs = 0;
  let confirmedOnEarlierScreen = false;
  let kanjiPenaltyApplied = false;
  const closedPopups = new Set<string>();
  const noticedTrapIds = new Set<string>();
  const triedDecoyIds = new Set<string>();
  const appliedInlineCouponIds = new Set<string>();
  let pressureNoticed = false;
  let sawSubscriptionOption = false;
  let sawHonestSubscriptionOption = false;
  const seenSubscriptionIds: string[] = [];
  // クーポンは実際に目にしたときだけ「見た」とする（画面内の表示を見たか、クーポン広告が出たとき）
  let sawCouponFeature = false;
  const seenCouponIds: string[] = [];
  // クーポンを押してみたか、押して実際に合計が下がったか。下がらなければ、使いたかった人は満足しない
  let triedCoupon = false;
  let couponLoweredTotal = false;
  let lastFailureComponentIds: string[] = [];
  /** クーポンの動きを適用し、合計が下がったかを記録する。合計の計算がない題材では、適用できたことをもって使えたとする */
  const useCoupon = (actions: ComponentAction[]): void => {
    triedCoupon = true;
    const before = withDerivedValues(spec, state).total;
    state = applyActions(spec, state, actions).state;
    const after = state.total;
    if (typeof before !== 'number' || typeof after !== 'number' || after < before) {
      couponLoweredTotal = true;
    }
  };
  /** 買えたとしても嫌な思いをしたこと。1つでもあれば「買えたけど不満」になる */
  const bothers: Array<{ friction: string; comment: string; componentIds: string[] }> = [];

  const fail = (
    reason: FailureReason,
    componentIds: string[] = lastFailureComponentIds,
  ): ComputerUseSimulationResult => {
    // 1人が同時に複数箇所を指摘すると分かりにくいため、その離脱の直接の原因になった1箇所だけに絞る
    const primaryCandidates =
      reason === 'too_many_steps' || reason === 'timeout'
        ? [
            ...Array.from(triedDecoyIds),
            ...bothers.flatMap((b) => b.componentIds),
            ...componentIds,
          ]
        : componentIds;
    const singleId = primaryCandidates.find(Boolean);
    return {
      outcome: 'abandoned',
      frictionPoint: FAILURE_TEXT[reason].friction,
      simulatedComment: FAILURE_TEXT[reason].comment,
      actionTrace,
      appliedConstraintNote,
      frictionComponentIds: singleId ? [singleId] : [],
    };
  };

  const overBudget = (): FailureReason | null => {
    if (steps > c.maxSteps) return 'too_many_steps';
    if (elapsedMs > c.patienceMs) return 'timeout';
    return null;
  };

  /**
   * 部品を1回押す。外れた先も判定する。
   * 戻り値: hit=狙いどおり / miss=空振り / danger=危険な部品に当たった / unreachable=届かない
   */
  const tap = (
    box: SimulatedBox,
    boxes: SimulatedBox[],
  ): 'hit' | 'miss' | 'danger' | 'unreachable' => {
    steps++;
    elapsedMs += MS_PER_STEP;
    if (c.environment === 'one_hand' && !box.comp.style.fullWidth && box.centerX > ONE_HAND_REACH_X) {
      actionTrace.push(`${steps}手目: 右端の${box.comp.label}に親指が届かない`);
      lastFailureComponentIds = [box.comp.id];
      return 'unreachable';
    }
    const { dx, dy } = computeDeterministicJitter(persona.id, steps, c.jitterPx);
    // 傾いた部品は当たり判定が狭くなる
    const tilt = Math.abs(box.comp.style.rotateDeg ?? 0) > 5 ? 0.6 : 1;
    const hit = Math.abs(dx) <= (box.width / 2) * tilt && Math.abs(dy) <= (box.height / 2) * tilt;
    if (hit) return 'hit';

    const x = box.centerX + dx;
    const y = box.centerY + dy;
    const other = boxes.find(
      (b) => b !== box && x >= b.left && x <= b.right && y >= b.top && y <= b.bottom,
    );
    if (other?.comp.role === 'danger' && other.comp.type === 'button') {
      actionTrace.push(`${steps}手目: 指がズレて${other.comp.label}を押してしまった`);
      lastFailureComponentIds = [other.comp.id, box.comp.id];
      return 'danger';
    }
    actionTrace.push(`${steps}手目: ${box.comp.label}を狙ったが指がズレて外れた`);
    return 'miss';
  };

  /**
   * 出現時間を過ぎたポップアップを閉じる。ポップアップは操作の途中で時間差に現れるため、
   * 手早く終える人は出る前に完了できることがある。
   * また、閉じるボタンの位置・大きさ・ラベルを工夫したり、別ページ遷移ではなくその場での特典適用に変えた場合は、
   * ストレスなく閉じられる（または特典を活用できる）ものとして扱う。
   */
  const closeDuePopups = (): FailureReason | null => {
    // 次に指が画面に触れる時点（1手ぶん先）で出ているかを見る。
    // 画面を開いてから最初のタップまでに出る広告を、すぐに押した人だけがすり抜けないようにする
    const touchAtMs = elapsedMs + MS_PER_STEP;
    const due = spec.components.filter(
      (comp) =>
        comp.type === 'popup' &&
        !closedPopups.has(comp.id) &&
        (!comp.stateKey || state[comp.stateKey] !== 'closed') &&
        touchAtMs >= (comp.popupDelayMs?.min ?? 0),
    );
    for (const popup of due) {
      closedPopups.add(popup.id);
      if (isCouponComponent(popup)) {
        sawCouponFeature = true;
        if (!seenCouponIds.includes(popup.id)) seenCouponIds.push(popup.id);
      }
      const closePos = popup.closePosition ?? 'top-right';
      const effectiveCloseSize: ComponentSize =
        popup.closeSize ?? (closePos === 'bottom-bar' ? 'lg' : popup.style.size);
      const hasCustomCloseLabel = Boolean(popup.closeLabel && popup.closeLabel.trim() !== '×');
      const hasFriendlyClose =
        closePos !== 'top-right' ||
        (SIZE_RANK[effectiveCloseSize] ?? 0) >= SIZE_RANK.md ||
        hasCustomCloseLabel ||
        Boolean(popup.customHtml);
      const popupActions = isCouponComponent(popup) ? couponActionsOf(spec, popup) : popup.actions;
      const hasHelpfulAction =
        popupActions.length > 0 && !popupActions.some((a) => a.type === 'open_page');

      const closeCompForRead: AppComponent = {
        ...popup,
        style: {
          ...popup.style,
          size: effectiveCloseSize,
          variant: hasFriendlyClose ? 'default' : popup.style.variant,
        },
      };

      if (!canRead(closeCompForRead, persona)) {
        steps++;
        elapsedMs += MS_POPUP;
        actionTrace.push(`${steps}手目: ${popup.label}が画面を覆い、閉じるボタンが見えない`);
        lastFailureComponentIds = [popup.id];
        return 'popup_unclosable';
      }

      const baseCloseHalf = HEIGHT_BY_SIZE[effectiveCloseSize] / 2;
      const closeHalf =
        closePos === 'bottom-bar'
          ? 90
          : hasFriendlyClose
            ? Math.max(baseCloseHalf * 1.8, 22)
            : baseCloseHalf;

      // 押しやすく工夫されていない初期状態のポップアップのみ、手数消費・誤タップ・不満が発生する
      if (!hasFriendlyClose && !hasHelpfulAction) {
        steps++;
        elapsedMs += MS_POPUP;
        const { dx, dy } = computeDeterministicJitter(persona.id, steps, c.jitterPx);
        if (Math.abs(dx) > closeHalf || Math.abs(dy) > closeHalf) {
          actionTrace.push(`${steps}手目: 広告の小さな閉じるボタンを外し、広告を押してしまった`);
          lastFailureComponentIds = [popup.id];
          return 'popup_tapped';
        }
        actionTrace.push(`${steps}手目: ${popup.label}を閉じた`);
        bothers.push({
          friction: '途中で広告を閉じさせられた',
          comment: '途中で広告が出てきて、閉じるのが面倒だった',
          componentIds: [popup.id],
        });
      } else if (hasHelpfulAction) {
        if (isCouponComponent(popup)) {
          useCoupon(popupActions);
        } else {
          state = applyActions(spec, state, popupActions).state;
        }
        actionTrace.push(`画面確認: ${popup.label}の特典を適用して閉じた`);
      } else {
        actionTrace.push(`画面確認: 押しやすい閉じるボタンで${popup.label}をさっと閉じた`);
      }
    }
    return null;
  };

  /** 外れたら手数の許す限り押し直す */
  const tapUntilHit = (
    box: SimulatedBox,
    boxes: SimulatedBox[],
  ): 'hit' | FailureReason => {
    let missCount = 0;
    for (;;) {
      const popupFailure = closeDuePopups();
      if (popupFailure) return popupFailure;
      const result = tap(box, boxes);
      if (result === 'hit') return 'hit';
      if (result === 'danger') return 'hit_danger';
      if (result === 'unreachable') return 'unreachable';
      missCount++;
      const budget = overBudget();
      if (budget) {
        lastFailureComponentIds = [box.comp.id];
        if (steps - missCount < c.maxSteps) {
          return 'tap_target_too_small';
        }
        return budget;
      }
    }
  };

  for (let screen = 0; screen < MAX_SCREENS; screen++) {
    const visible = spec.components.filter((comp) => isComponentVisible(comp, state));
    const boxes = layoutComponentBoxes(visible, customBounds);
    const boxOf = (comp: AppComponent) => boxes.find((b) => b.comp === comp);

    // 1. すでに出ているポップアップを閉じる
    const popupFailure = closeDuePopups();
    if (popupFailure) return fail(popupFailure);

    // 2. 関係ない表示が多すぎると迷う
    const distractionComps = visible.filter((comp) => comp.role === 'distraction');
    const distractions = distractionComps.length;
    if (distractions > c.clutterTolerance) {
      actionTrace.push(`画面確認: 関係ない表示が${distractions}個あり、どこを見ればいいか迷って離脱`);
      return fail(
        'clutter',
        distractionComps.map((d) => d.id),
      );
    }
    elapsedMs += distractions * MS_PER_DISTRACTION;

    // 偽の残り時間などで急かされると、急かされるのが嫌いな人は買えても不満が残る
    const pressure = visible.find(
      (comp) => comp.darkPattern === 'false_urgency' || comp.label.includes('{countdown}'),
    );
    if (pressure && c.dislikesPressure && !pressureNoticed) {
      pressureNoticed = true;
      actionTrace.push(`画面確認: ${renderLabel(pressure, state)}で急かされて嫌な気分になった`);
      bothers.push({
        friction: '残り時間で急かされた',
        comment: '残り時間で急かされて、落ち着いて選べなかった',
        componentIds: [pressure.id],
      });
    }

    // 漢数字の値段は読むのに時間がかかる
    const kanjiComps = visible.filter((comp) => comp.style.numberFormat === 'kanji');
    if (!kanjiPenaltyApplied && kanjiComps.length > 0) {
      kanjiPenaltyApplied = true;
      elapsedMs += MS_KANJI_READING;
      actionTrace.push('画面確認: 漢数字の値段を読むのに時間がかかった');
      bothers.push({
        friction: '値段が読みにくかった',
        comment: '値段が漢数字で、いくらなのか読むのに手間取った',
        componentIds: kanjiComps.map((k) => k.id),
      });
    }

    // 3a. 必須入力を一括で埋めるショートカット（「前回と同じ設定」等の＋α機能）があれば活用する
    const requiredKeys = new Set(
      visible
        .filter((v) => v.role === 'required_input' && !isInputFilled(v, state) && v.stateKey)
        .map((v) => v.stateKey!),
    );
    if (requiredKeys.size > 0) {
      const shortcut = visible.find(
        (v) =>
          v.role !== 'danger' &&
          !hasCompletionAction(v) &&
          !v.actions.some((a) => a.type === 'open_page' || a.type === 'reset_all') &&
          v.actions.some((a) => a.type === 'set_state' && a.targetKey && requiredKeys.has(a.targetKey)) &&
          canRead(v, persona),
      );
      if (shortcut) {
        const box = boxOf(shortcut);
        if (box) {
          state = applyActions(spec, state, shortcut.actions).state;
          actionTrace.push(`画面確認: ${shortcut.label}で入力項目をまとめて設定した`);
        }
      }
    }

    // 3. 必須の入力欄を埋める
    for (const comp of visible.filter((v) => v.role === 'required_input')) {
      if (isInputFilled(comp, state)) continue;
      if (!canRead(comp, persona)) {
        actionTrace.push(`画面確認: ${comp.label}の文字が小さくて読めず離脱`);
        return fail('unreadable_input', [comp.id]);
      }
      const box = boxOf(comp);
      if (!box) continue;
      const result = tapUntilHit(box, boxes);
      if (result !== 'hit') return fail(result);
      if (comp.stateKey) {
        const option = comp.options?.find((o) => o !== comp.placeholder);
        state = { ...state, [comp.stateKey]: option ?? '入力済み' };
      }
      actionTrace.push(`${steps}手目: ${comp.label}を入力した`);
      const budget = overBudget();
      if (budget) return fail(budget, [comp.id]);
    }

    // 4. 罠に気づけば直す（細かい字を読む人が、読める大きさの場合だけ）
    for (const comp of visible.filter((v) => v.role === 'trap' && v.trapWhen)) {
      if (!matchesCondition(comp.trapWhen!, state)) continue;
      if (!c.readsFinePrint || !canRead(comp, persona)) continue;
      noticedTrapIds.add(comp.id);
      if (comp.trapSafeValue === undefined || !comp.stateKey) {
        actionTrace.push(`画面確認: ${comp.label}の不親切な条件に気づいたうえで進む`);
        bothers.push({
          friction: '分かりにくい条件が付いていた',
          comment: `${comp.label}に気づいたけど、なんだか気分が悪い`,
          componentIds: [comp.id],
        });
        continue;
      }
      const box = boxOf(comp);
      if (!box) continue;
      const result = tapUntilHit(box, boxes);
      if (result !== 'hit') return fail(result);
      state = { ...state, [comp.stateKey]: comp.trapSafeValue };
      actionTrace.push(`${steps}手目: 最初から付いていた${comp.label}に気づいて直した`);
      bothers.push({
        friction: '勝手に付いていた申し込みを外した',
        comment: `危うく${comp.label}のまま買うところだった`,
        componentIds: [comp.id],
      });
      const budget = overBudget();
      if (budget) return fail(budget, [comp.id]);
    }

    // 4b. 正直な定期便（最初からOFF・読める文字サイズ・不当な縛り文言なし）なら、定期便を望む人やよく読む人は自分の意思で申し込む
    for (const comp of visible.filter((v) => isSubscriptionComponent(v))) {
      sawSubscriptionOption = true;
      if (!seenSubscriptionIds.includes(comp.id)) seenSubscriptionIds.push(comp.id);
      const initialSub = spec.state.find((s) => s.key === 'subscription')?.initialValue;
      const isHonest =
        isHonestOptInComponent(comp, initialSub) &&
        state.subscription === false &&
        canRead(comp, persona);
      if (isHonest) {
        sawHonestSubscriptionOption = true;
      }
      if (isHonest && (c.readsFinePrint || c.wantsSubscription) && !noticedTrapIds.has(comp.id)) {
        noticedTrapIds.add(comp.id);
        state = { ...state, subscription: true };
        actionTrace.push(`${steps}手目: 条件が分かりやすいので${comp.label}を自分で選んだ`);
      }
    }

    // 4c. 画面内に残されたクーポンバナーや特典ボタンがあれば確認・適用する
    for (const comp of visible.filter((v) => v.type !== 'popup' && isCouponComponent(v))) {
      sawCouponFeature = true;
      if (!seenCouponIds.includes(comp.id)) seenCouponIds.push(comp.id);
      const actions = couponActionsOf(spec, comp);
      if (
        !appliedInlineCouponIds.has(comp.id) &&
        actions.length > 0 &&
        !hasCompletionAction(comp) &&
        !actions.some((a) => a.type === 'open_page' || a.type === 'reset_all') &&
        canRead(comp, persona)
      ) {
        appliedInlineCouponIds.add(comp.id);
        useCoupon(actions);
        actionTrace.push(`画面確認: ${comp.label}の特典を適用した`);
      }
    }

    // 確認画面（2画面目以降）で内容表示の文字が小さすぎて読めないと不満が残る
    // （ステップ番号や見出しではなく、注文内容や合計金額などの確認項目を対象にする）
    if (screen > 0) {
      const unreadableConfirmInfo = visible.find(
        (v) =>
          (v.role === 'info' || v.role === 'total') &&
          v.type !== 'progress_steps' &&
          v.type !== 'heading' &&
          !canRead(v, persona),
      );
      if (unreadableConfirmInfo) {
        actionTrace.push(`画面確認: 確認画面の${unreadableConfirmInfo.label}の文字が小さくて確かめづらかった`);
        bothers.push({
          friction: '確認画面の文字が小さかった',
          comment: '確認画面の文字が小さくて、内容を確かめづらかった',
          componentIds: [unreadableConfirmInfo.id],
        });
      }
    }

    // 5. 押すボタンを決める
    const buttons = boxes.filter((b) => b.comp.type === 'button');
    const candidates = buttons.filter(
      (b) =>
        (b.comp.role === 'primary_action' || b.comp.role === 'decoy') &&
        !triedDecoyIds.has(b.comp.id),
    );
    const readable = candidates.filter((b) => canRead(b.comp, persona));
    const pool = readable.length > 0 ? readable : buttons;
    if (pool.length === 0) {
      actionTrace.push('画面確認: 進むためのボタンが見つからず離脱');
      return fail('no_target', []);
    }
    const preferred =
      c.readsFinePrint && readable.length > 0
        ? readable.filter((b) => b.comp.role === 'primary_action')
        : [];
    const target = (preferred.length > 0 ? preferred : pool).reduce((best, b) =>
      prominence(b.comp) > prominence(best.comp) ? b : best,
    );

    // 完了ボタンを押す前の確認
    const isCompletion = hasCompletionAction(target.comp);
    if (isCompletion && c.needsReadableTotal) {
      const totals = visible.filter((comp) => comp.role === 'total');
      const readableTotal = totals.some(
        (comp) => canRead(comp, persona) && comp.style.numberFormat !== 'kanji',
      );
      if (!readableTotal) {
        actionTrace.push('画面確認: 合計金額がすぐに読み取れず、買うのをやめた');
        return fail(
          'total_unreadable',
          totals.map((t) => t.id),
        );
      }
    }
    const confirmsOnTap = target.comp.behavior?.effect === 'confirm_dialog';
    if (isCompletion && c.wantsConfirmation && !confirmedOnEarlierScreen && !confirmsOnTap) {
      actionTrace.push(`画面確認: 確認なしで${target.comp.label}を押すのが怖くて離脱`);
      return fail('no_confirmation', [target.comp.id]);
    }

    // 6. 押す
    const tapResult = tapUntilHit(target, boxes);
    if (tapResult !== 'hit') {
      if (tapResult === 'hit_danger' && target.comp.role === 'danger') {
        return fail('hit_danger', [target.comp.id]);
      }
      return fail(tapResult);
    }
    if (target.comp.role === 'danger') {
      actionTrace.push(`${steps}手目: 一番目立つ${target.comp.label}を押してしまい、入力が消えた`);
      return fail('hit_danger', [target.comp.id]);
    }

    const effect = target.comp.behavior?.effect ?? 'none';
    if (effect === 'require_3_clicks') {
      steps += 2;
      elapsedMs += MS_PER_STEP * 2;
    } else if (effect === 'escape_on_hover') {
      steps += 1;
      elapsedMs += MS_PER_STEP;
    } else if (confirmsOnTap) {
      // 確認の1手の前にも、出てきた広告に当たる
      const popupBeforeConfirm = closeDuePopups();
      if (popupBeforeConfirm) return fail(popupBeforeConfirm);
      steps += 1;
      elapsedMs += MS_PER_STEP;
    }
    const budget = overBudget();
    if (budget) {
      actionTrace.push(
        `${steps}手目: そこまでの手順が多すぎて、${target.comp.label}で完了する前に離脱`,
      );
      return fail(budget, [target.comp.id]);
    }

    if (target.comp.actions.length === 0) {
      if (target.comp.role === 'decoy') {
        actionTrace.push(`${steps}手目: 目立つ${target.comp.label}を押したが何も起きない`);
        triedDecoyIds.add(target.comp.id);
        bothers.push({
          friction: '目立つボタンが効かなかった',
          comment: `${target.comp.label}を押しても何も起きなくて焦った`,
          componentIds: [target.comp.id],
        });
        screen--;
        if (triedDecoyIds.size >= 3) return fail('decoy_only', Array.from(triedDecoyIds));
        continue;
      }
      actionTrace.push(`${steps}手目: ${target.comp.label}を押したが何も起きず離脱`);
      return fail('no_response', [target.comp.id]);
    }

    const result = applyActions(spec, state, target.comp.actions);
    state = result.state;

    if (result.openedPage && !result.completed) {
      actionTrace.push(`${steps}手目: ${target.comp.label}を押したら${result.openedPage}のページに飛ばされた`);
      return fail('left_page', [target.comp.id]);
    }

    if (result.completed) {
      // 7. 完了。気づかなかった罠が残っていれば後悔
      const activeTraps = spec.components.filter(
        (comp) =>
          comp.role === 'trap' &&
          comp.trapWhen &&
          matchesCondition(comp.trapWhen, state) &&
          !noticedTrapIds.has(comp.id),
      );
      if (activeTraps.length > 0) {
        actionTrace.push(
          `${steps}手目: ${target.comp.label}で完了したが、${activeTraps.map((t) => t.label).join('と')}に気づかなかった`,
        );
        return {
          outcome: 'regretted',
          frictionPoint: '気づかないまま余計な契約をした',
          simulatedComment: `注文はできたけど、あとで${activeTraps[0].label}になっていたと気づいた`,
          actionTrace,
          appliedConstraintNote,
          frictionComponentIds: [activeTraps[0].id],
          finalState: state,
        };
      }
      actionTrace.push(`${steps}手目: ${target.comp.label}を押してタスク完了イベントを検知`);
      // 定期便もクーポンも、お客さんの希望は「あれば使いたい」。店に最初からないものには不満を言わず、
      // 店にあったのに消されたときだけ不満にする。部品を消しても状態の変数は残るので、そこから店にあったかを見る
      const hadSubscriptionInShop = spec.state.some((s) => s.key.toLowerCase().includes('subscription'));
      if (c.wantsSubscription && hadSubscriptionInShop && !sawSubscriptionOption) {
        actionTrace.push('画面確認: 毎月届く定期便を選びたかったが設定が見当たらなかった');
        bothers.push({
          friction: '定期便の設定がなかった',
          comment: '毎月届けてほしいのに設定がなくて残念',
          componentIds: [],
        });
      } else if (c.wantsSubscription && hadSubscriptionInShop && !sawHonestSubscriptionOption && !state.subscription) {
        actionTrace.push('画面確認: 定期便の設定はあったが、文字が小さいか条件が不安で選べなかった');
        bothers.push({
          friction: '定期便の条件が分かりづらくて選べなかった',
          comment: '定期便にしたかったけど、字が小さいか条件が不安で選べなかった',
          componentIds: seenSubscriptionIds,
        });
      }
      const hadCouponInShop = hasCouponFeatureInShop(spec);
      if (c.wantsCoupon && hadCouponInShop && !sawCouponFeature) {
        actionTrace.push('画面確認: クーポンを使いたかったが見当たらなかった');
        bothers.push({
          friction: COUPON_FRICTIONS.missing,
          comment: 'クーポンは無くなってしまったのか、使いたかったのに残念',
          componentIds: [],
        });
      } else if (c.wantsCoupon && hadCouponInShop && !couponLoweredTotal) {
        if (triedCoupon) {
          actionTrace.push('画面確認: クーポンを使ったが、合計金額が変わっていなかった');
          bothers.push({
            friction: COUPON_FRICTIONS.noDiscount,
            comment: 'クーポンを使ったのに、合計が安くなっていなかった',
            componentIds: seenCouponIds,
          });
        } else {
          actionTrace.push('画面確認: クーポンの表示はあったが、この画面で使う方法がなかった');
          bothers.push({
            friction: COUPON_FRICTIONS.unusable,
            comment: 'クーポンがあるのに、ここで使えなくて残念',
            componentIds: seenCouponIds,
          });
        }
      }
      if (bothers.length > 0) {
        const primaryBotherId = bothers[0].componentIds.find(Boolean);
        return {
          outcome: 'bothered',
          frictionPoint: bothers[0].friction,
          simulatedComment: `買えたけど、${bothers[0].comment}`,
          actionTrace,
          appliedConstraintNote,
          frictionComponentIds: primaryBotherId ? [primaryBotherId] : [],
          finalState: state,
        };
      }
      return {
        outcome: 'achieved',
        frictionPoint: `${steps}手で迷わず完了`,
        simulatedComment: '迷わずに最後まで注文できた',
        actionTrace,
        appliedConstraintNote,
        frictionComponentIds: [],
        finalState: state,
      };
    }

    // 完了しなかった場合は次の画面へ（確認画面など）
    actionTrace.push(`${steps}手目: ${target.comp.label}を押して次の画面へ`);
    confirmedOnEarlierScreen = true;
  }

  actionTrace.push('画面を行き来するだけで完了にたどり着けず離脱');
  return fail('too_many_steps', []);
}

/**
 * 手直し後にお客さんの満足人数が増えなかったとき、
 * 「隠れていた次の困りごとが見えてきます」と案内を出すべきか判定する。
 * 前回の反応確認があり、満足人数が減っておらず、前回も今回も未達成のお客さんの誰かで
 * 気にしている部品（frictionComponentIds）が別の部品に変わったときだけ true を返す
 */
export function shouldShowNextFrictionNotice(params: {
  changedSinceLastReview: boolean;
  previousResults?: ReadonlyArray<Pick<ReviewerEvaluation, 'reviewerId' | 'outcome' | 'frictionComponentIds'>>;
  currentResults: ReadonlyArray<Pick<ReviewerEvaluation, 'reviewerId' | 'outcome' | 'frictionComponentIds'>>;
  activeReviewerIds: ReadonlyArray<string>;
}): boolean {
  const { changedSinceLastReview, previousResults, currentResults, activeReviewerIds } = params;
  if (!changedSinceLastReview || !previousResults || activeReviewerIds.length === 0) {
    return false;
  }
  const countAchieved = (
    list: ReadonlyArray<Pick<ReviewerEvaluation, 'reviewerId' | 'outcome'>>,
  ) =>
    activeReviewerIds.filter(
      (id) => list.find((r) => r.reviewerId === id)?.outcome === 'achieved',
    ).length;

  const previousAchieved = countAchieved(previousResults);
  const currentAchieved = countAchieved(currentResults);
  if (currentAchieved >= activeReviewerIds.length || currentAchieved !== previousAchieved) {
    return false;
  }

  const normalizeIds = (ids: ReadonlyArray<string> | undefined): string[] =>
    [...new Set((ids ?? []).filter(Boolean))].sort();

  for (const reviewerId of activeReviewerIds) {
    const prev = previousResults.find((r) => r.reviewerId === reviewerId);
    const curr = currentResults.find((r) => r.reviewerId === reviewerId);
    if (!prev || !curr) continue;
    if (prev.outcome === 'achieved' || curr.outcome === 'achieved') continue;
    const prevIds = normalizeIds(prev.frictionComponentIds);
    const currIds = normalizeIds(curr.frictionComponentIds);
    if (prevIds.length === 0 || currIds.length === 0) continue;
    if (currIds.length !== prevIds.length || currIds.some((id, i) => id !== prevIds[i])) {
      return true;
    }
  }
  return false;
}

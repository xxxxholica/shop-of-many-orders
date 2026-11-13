import {
  applyActions,
  buildActionsFromDataAttrs,
  createInitialState,
  hasCompletionAction,
  isComponentVisible,
  matchesCondition,
  isInputFilled,
  coerceStateValue,
  isSafeStateKey,
  renderLabel,
  renderTemplate,
  withDerivedValues,
  type AppState,
} from '../shared/appRuntime.ts';
import {
  buildStandaloneSandboxHtml,
  escapeHtml,
  FORBIDDEN_CUSTOM_HTML_TAGS,
  isForbiddenCustomHtmlAttr,
  sanitizeCssColor,
  sanitizeCustomCss,
  sanitizeCustomHtml,
} from '../shared/sandboxValidator.ts';
import type { AppComponent, AppSpec, ComponentAction } from '../shared/types.ts';

export interface SandboxCallbacks {
  /** 状態をまるごと置き換える */
  onStateChange: (newState: AppState) => void;
  /** 状態の一部だけを書き換える（時間差で動くポップアップなど、描画後に起きる変化用） */
  onStatePatch: (patch: AppState) => void;
  /** スマホの中で起きたことのお知らせ。画面を邪魔しないよう、画面の外（チャット）に出す */
  onNotice: (message: string, type: 'info' | 'success' | 'warning') => void;
  /** 改造の作業ログやシステムメッセージ内の部品名クリックで光らせる部品 */
  justChangedIds?: Set<string>;
  /** スマホ画面の本文がスクロールしたときの通知（色ペンの丸囲み位置を追従させる用） */
  onScroll?: () => void;
}

/**
 * iframe 隔離されたサンドボックス内（またはフォールバック時は直下）から data-component-id 要素一覧を取得する。
 */
export function querySandboxComponentElements(container: HTMLElement): HTMLElement[] {
  const iframe = container.querySelector<HTMLIFrameElement>('iframe.sandbox-iframe');
  if (iframe?.contentDocument) {
    return Array.from(iframe.contentDocument.querySelectorAll<HTMLElement>('[data-component-id]'));
  }
  return Array.from(container.querySelectorAll<HTMLElement>('[data-component-id]'));
}

export function findSandboxComponentElement(
  container: HTMLElement,
  componentId: string,
): HTMLElement | undefined {
  return querySandboxComponentElements(container).find(
    (node) => node.dataset.componentId === componentId,
  );
}

/**
 * サンドボックス内の要素の矩形を、ステージ基準の座標にそろえる。
 * iframe 内の要素は iframe がステージ全面に重なっているためそのまま使い、直下フォールバック時はステージ左上を引く
 */
export function getStageRelativeRect(
  el: HTMLElement,
  stageRect: DOMRect,
): { left: number; top: number; right: number; bottom: number; width: number; height: number } {
  const r = el.getBoundingClientRect();
  const isInsideIframe = el.ownerDocument !== document;
  const dx = isInsideIframe ? 0 : stageRect.left;
  const dy = isInsideIframe ? 0 : stageRect.top;
  return {
    left: r.left - dx,
    top: r.top - dy,
    right: r.right - dx,
    bottom: r.bottom - dy,
    width: r.width,
    height: r.height,
  };
}

function syncStylesIntoIframe(iframeDoc: Document): void {
  const parentStyles = document.querySelectorAll('style, link[rel="stylesheet"]');
  iframeDoc.head.querySelectorAll('[data-synced-style]').forEach((n) => n.remove());
  for (const node of parentStyles) {
    const clone = node.cloneNode(true) as HTMLElement;
    clone.setAttribute('data-synced-style', '1');
    iframeDoc.head.appendChild(clone);
  }
}

/**
 * 文字の読み込みが終わると、本文の高さが変わる。読み込み後の文字で測り直す。
 * 本文の枠の大きさの変化は scheduleFit で見張っている
 */
function refitWhenFontsLoad(iframeDoc: Document): void {
  const fonts = iframeDoc.fonts;
  if (!fonts) return;
  fonts.addEventListener('loadingdone', () => refitVisibleSandbox(iframeDoc));
  void fonts.ready.then(() => refitVisibleSandbox(iframeDoc));
}

/**
 * 親ページのスタイルが変わったら（開発中のCSS差し替えなど）、iframe にもコピーし直す。
 * iframe を作ったときに1回コピーするだけだと、再読み込みするまで古い見た目のままになるため。
 */
let styleObserver: MutationObserver | null = null;

function watchParentStyles(iframe: HTMLIFrameElement): void {
  styleObserver?.disconnect();
  styleObserver = new MutationObserver(() => {
    if (iframe.contentDocument) syncStylesIntoIframe(iframe.contentDocument);
  });
  styleObserver.observe(document.head, { childList: true, subtree: true, characterData: true });
}

/** 状態変数の型に合わせて入力値を変換する */
/**
 * 購入完了・別ページへの移動の演出用の内部状態。
 * お題アプリの状態とは別に、キー名の先頭を __ にして持つ（初期状態に戻すと消える）。
 */
const PAGE_KEY = '__page';
const PAGE_TITLE_KEY = '__pageTitle';
const PAGE_PHASE_KEY = '__pagePhase';
/** 完了ページ・移動先ページを見せる時間と、読み込み中のスピナーを見せる時間（ミリ秒） */
const PAGE_SHOW_MS = 2400;
const PAGE_LOADING_MS = 1300;

type PagePhase = 'show' | 'loading';
let pageTimer: { phase: PagePhase; timer: number } | null = null;

/** 完了ページ → 読み込み中 → 最初の画面、の順に進める */
function schedulePageTransition(
  spec: AppSpec,
  liveState: AppState,
  callbacks: SandboxCallbacks,
): void {
  const phase = liveState[PAGE_KEY] ? (liveState[PAGE_PHASE_KEY] as PagePhase) : undefined;
  if (!phase) {
    if (pageTimer) {
      window.clearTimeout(pageTimer.timer);
      pageTimer = null;
    }
    return;
  }
  if (pageTimer?.phase === phase) return;
  if (pageTimer) window.clearTimeout(pageTimer.timer);
  pageTimer = {
    phase,
    timer: window.setTimeout(
      () => {
        pageTimer = null;
        if (phase === 'show') {
          callbacks.onStatePatch({ [PAGE_PHASE_KEY]: 'loading' });
        } else {
          callbacks.onStateChange(createInitialState(spec));
        }
      },
      phase === 'show' ? PAGE_SHOW_MS : PAGE_LOADING_MS,
    ),
  };
}

/** ポップアップを出すタイマー。描画し直しても二重に予約しないよう部品IDごとに持つ */
const popupTimers = new Map<string, number>();

function clearPopupTimers(): void {
  for (const timer of popupTimers.values()) window.clearTimeout(timer);
  popupTimers.clear();
}

function schedulePopups(
  spec: AppSpec,
  liveState: AppState,
  selectedComponentIds: Set<string>,
  callbacks: SandboxCallbacks,
): void {
  // 完了ページや移動先ページを見せている間は広告を出さない
  if (liveState[PAGE_KEY]) {
    clearPopupTimers();
    return;
  }
  // ポップアップ以外の部品を赤ペンで囲んでいる最中や、発言者が裏の部品を指し示している最中は、閉じた広告が勝手に被さらないようにする
  const activeFocusIds = [
    ...Array.from(selectedComponentIds),
    ...Array.from(callbacks.justChangedIds ?? []),
  ];
  const hasNonPopupCircled = activeFocusIds.some((id) => {
    const comp = spec.components.find((c) => c.id === id);
    return comp && comp.type !== 'popup';
  });
  const popups = spec.components.filter((c) => c.type === 'popup' && c.stateKey);
  const popupIds = new Set(popups.map((p) => p.id));
  for (const [id, timer] of popupTimers) {
    if (!popupIds.has(id)) {
      window.clearTimeout(timer);
      popupTimers.delete(id);
    }
  }
  for (const popup of popups) {
    const key = popup.stateKey!;
    if (liveState[key] === 'open' || (liveState[key] === 'closed' && hasNonPopupCircled)) {
      const timer = popupTimers.get(popup.id);
      if (timer !== undefined) {
        window.clearTimeout(timer);
        popupTimers.delete(popup.id);
      }
      continue;
    }
    if (liveState[key] !== 'waiting' && liveState[key] !== 'closed') {
      continue;
    }
    if (popupTimers.has(popup.id)) continue;
    const range = popup.popupDelayMs ?? { min: 3000, max: 6000 };
    // ×ボタンで閉じた（closed）後も、数秒後にまた自動で出てくるようにする（裏の画面を少し確認できる余裕を持たせる）
    const minDelay = liveState[key] === 'closed' ? Math.max(3500, range.min) : range.min;
    const maxDelay = liveState[key] === 'closed' ? Math.max(5500, range.max) : range.max;
    const delay = minDelay + Math.random() * Math.max(0, maxDelay - minDelay);
    popupTimers.set(
      popup.id,
      window.setTimeout(() => {
        popupTimers.delete(popup.id);
        callbacks.onStatePatch({ [key]: 'open' });
      }, delay),
    );
  }
}

/** 購入完了ページ・移動先ページ・読み込み中のスピナー */
function renderTransientPage(
  doc: Document,
  kind: 'complete' | 'away',
  title: string,
  loading: boolean,
): HTMLElement {
  const page = doc.createElement('div');
  page.className = `transient-page is-${kind}`;

  if (loading) {
    page.classList.add('is-loading');
    page.innerHTML = `
      <span class="page-spinner" role="status" aria-label="読み込み中"></span>
      <span class="page-loading-text">読み込み中</span>
    `;
    return page;
  }

  if (kind === 'complete') {
    page.innerHTML = `
      <span class="page-check" aria-hidden="true">
        <svg viewBox="0 0 256 256" width="34" height="34" fill="currentColor"><path d="M229.66,77.66l-128,128a8,8,0,0,1-11.32,0l-56-56a8,8,0,0,1,11.32-11.32L96,188.69,218.34,66.34a8,8,0,0,1,11.32,11.32Z"/></svg>
      </span>
    `;
    const heading = doc.createElement('div');
    heading.className = 'page-title';
    heading.textContent = title;
    const sub = doc.createElement('div');
    sub.className = 'page-sub';
    sub.textContent = 'ご注文の内容をメールでお送りしました';
    page.append(heading, sub);
    return page;
  }

  // 別のページに飛ばされたときは、それらしいページの骨組みを見せる
  const heading = doc.createElement('div');
  heading.className = 'page-title';
  heading.textContent = title;
  const sub = doc.createElement('div');
  sub.className = 'page-sub';
  sub.textContent = '別のページに移動しました';
  const skeleton = doc.createElement('div');
  skeleton.className = 'page-skeleton';
  skeleton.innerHTML = '<i></i><i></i><i></i><i></i>';
  page.append(heading, sub, skeleton);
  return page;
}

const CHEVRON_SVG =
  '<svg viewBox="0 0 24 24" width="1em" height="1em"><path d="M15 5l-7 7 7 7" stroke="currentColor" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const LOCK_SVG =
  '<svg viewBox="0 0 24 24" width="1em" height="1em"><rect x="5" y="11" width="14" height="9" rx="2" fill="currentColor"/><path d="M8 11V8a4 4 0 0 1 8 0v3" stroke="currentColor" stroke-width="2" fill="none"/></svg>';
/** クラフト紙のコーヒー袋のイラスト */
const COFFEE_BAG_SVG =
  '<svg viewBox="0 0 64 80" width="100%" height="100%" aria-hidden="true"><path d="M10 14 L14 6 L50 6 L54 14 Z" fill="#a9733f"/><path d="M10 14 h44 l3 58 a4 4 0 0 1 -4 4 H11 a4 4 0 0 1 -4 -4 Z" fill="#c8915a"/><path d="M10 14 h44 l1 10 H9 Z" fill="#b57d47"/><rect x="17" y="32" width="30" height="30" rx="3" fill="#fffaf2"/><g transform="rotate(-25 32 44)"><ellipse cx="32" cy="44" rx="6" ry="8.5" fill="#5b3a26"/><path d="M29 37 q5 7 0 14" stroke="#c8915a" stroke-width="1.6" fill="none"/></g><rect x="21" y="55" width="22" height="2.4" rx="1.2" fill="#c8915a"/></svg>';

type SetValue = (key: string, raw: string | boolean) => void;

/** 商品カード：絵・札・商品名・補足・値段と、数量の±ボタン */
function renderProductCard(
  doc: Document,
  comp: AppComponent,
  state: AppState,
  label: string,
  caption: string,
  setValue: SetValue,
): HTMLElement {
  const card = doc.createElement('div');
  card.className = 'comp-product-card';

  const thumb = doc.createElement('div');
  thumb.className = 'product-thumb';
  if (comp.icon === 'coffee_bag') thumb.innerHTML = COFFEE_BAG_SVG;
  else thumb.textContent = comp.icon ?? '🛍️';

  const info = doc.createElement('div');
  info.className = 'product-info';
  if (comp.badge) {
    const badge = doc.createElement('span');
    badge.className = 'product-badge';
    badge.textContent = comp.badge;
    info.appendChild(badge);
  }
  const name = doc.createElement('div');
  name.className = 'product-name';
  name.textContent = label;
  info.appendChild(name);
  if (caption) {
    const meta = doc.createElement('div');
    meta.className = 'product-meta';
    meta.textContent = caption;
    info.appendChild(meta);
  }

  const row = doc.createElement('div');
  row.className = 'product-row';
  const price = doc.createElement('span');
  price.className = 'product-price';
  price.textContent = comp.priceLabel
    ? renderTemplate(comp.priceLabel, state, comp.style.numberFormat)
    : '';
  row.appendChild(price);

  if (comp.stateKey) {
    const key = comp.stateKey;
    const values = (comp.options ?? ['1', '2', '3', '4', '5'])
      .map(Number)
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    const current = Number(state[key] ?? values[0]);
    const idx = values.indexOf(current);
    const stepper = doc.createElement('span');
    stepper.className = 'product-stepper';
    const minus = doc.createElement('button');
    minus.type = 'button';
    minus.textContent = '−';
    minus.setAttribute('aria-label', '数量を減らす');
    minus.disabled = idx <= 0;
    minus.addEventListener('click', () => {
      if (idx > 0) setValue(key, String(values[idx - 1]));
    });
    const count = doc.createElement('span');
    count.className = 'product-count';
    count.textContent = String(current);
    const plus = doc.createElement('button');
    plus.type = 'button';
    plus.textContent = '＋';
    plus.setAttribute('aria-label', '数量を増やす');
    plus.disabled = idx === -1 || idx >= values.length - 1;
    plus.addEventListener('click', () => {
      if (idx >= 0 && idx < values.length - 1) setValue(key, String(values[idx + 1]));
    });
    stepper.append(minus, count, plus);
    row.appendChild(stepper);
  }
  info.appendChild(row);

  card.append(thumb, info);
  return card;
}

/** 状態 step が初期値から次の画面（確認画面など）に進んでいれば、手順表示も最後の手順へ進める */
function resolveProgressCurrentStep(
  spec: AppSpec,
  comp: AppComponent,
  label: string,
  liveState: AppState,
): string {
  const steps = comp.options ?? [];
  if (steps.length === 0) return label;
  const stepDef = spec.state.find((s) => s.key === 'step');
  const currentStepVal = liveState.step;
  if (
    stepDef !== undefined &&
    currentStepVal !== undefined &&
    String(currentStepVal) !== String(stepDef.initialValue) &&
    currentStepVal !== '' &&
    currentStepVal !== false &&
    currentStepVal !== 0
  ) {
    const matched = steps.find((s) => s === String(currentStepVal));
    return matched ?? steps[steps.length - 1];
  }
  return label;
}

/** 手順の表示：今の手順より前は済み、今の手順を強調する */
function renderProgressSteps(doc: Document, comp: AppComponent, current: string): HTMLElement {
  const wrap = doc.createElement('div');
  wrap.className = 'comp-progress';
  const steps = comp.options ?? [];
  const currentIdx = steps.indexOf(current);
  steps.forEach((name, i) => {
    if (i > 0) {
      const rule = doc.createElement('span');
      rule.className = 'progress-rule';
      wrap.appendChild(rule);
    }
    const step = doc.createElement('span');
    step.className = `progress-step ${i < currentIdx ? 'is-done' : i === currentIdx ? 'is-now' : ''}`;
    const dot = doc.createElement('span');
    dot.className = 'progress-dot';
    dot.textContent = i < currentIdx ? '✓' : String(i + 1);
    const text = doc.createElement('span');
    text.textContent = name;
    step.append(dot, text);
    wrap.appendChild(step);
  });
  return wrap;
}

/** 選べるカード：選択肢ごとに丸印と補足を並べる */
function renderChoiceCards(
  doc: Document,
  comp: AppComponent,
  state: AppState,
  setValue: SetValue,
): HTMLElement {
  const list = doc.createElement('div');
  list.className = 'comp-choice-list';
  list.setAttribute('role', 'radiogroup');
  list.setAttribute('aria-label', comp.label);
  const current = comp.stateKey ? String(state[comp.stateKey] ?? '') : '';
  (comp.options ?? []).forEach((opt, i) => {
    const item = doc.createElement('button');
    item.type = 'button';
    item.className = `choice-item ${opt === current ? 'is-selected' : ''}`;
    item.setAttribute('role', 'radio');
    item.setAttribute('aria-checked', String(opt === current));
    const radio = doc.createElement('span');
    radio.className = 'choice-radio';
    const body = doc.createElement('span');
    body.className = 'choice-body';
    const main = doc.createElement('span');
    main.className = 'choice-main';
    main.textContent = opt;
    body.appendChild(main);
    const detail = comp.optionDetails?.[i];
    if (detail) {
      const sub = doc.createElement('span');
      sub.className = 'choice-sub';
      sub.textContent = renderTemplate(detail, state, comp.style.numberFormat);
      body.appendChild(sub);
    }
    item.append(radio, body);
    item.addEventListener('click', () => {
      if (comp.stateKey) setValue(comp.stateKey, opt);
    });
    list.appendChild(item);
  });
  return list;
}

/**
 * 煽り用の偽のカウントダウン。ラベルに {countdown} と書くと「2:59」から1秒ずつ減り、0になるとまた戻る。
 * 画面を作り直しても続きから数えるよう、開始時刻はページを開いたときの1つだけにする。
 * 状態として持つと毎秒画面を作り直すことになるため、数字の部分だけを直接書き換える。
 */
const COUNTDOWN_SECONDS = 179; // 2:59。appRuntime の COUNTDOWN_INITIAL_TEXT と揃える
const countdownStartedAt = Date.now();
let countdownDoc: Document | null = null;
let countdownTimer: number | null = null;

function currentCountdownText(): string {
  const elapsed = Math.floor((Date.now() - countdownStartedAt) / 1000);
  const left = COUNTDOWN_SECONDS - (elapsed % (COUNTDOWN_SECONDS + 1));
  return `${Math.floor(left / 60)}:${String(left % 60).padStart(2, '0')}`;
}

function startCountdownTicker(doc: Document): void {
  countdownDoc = doc;
  if (countdownTimer !== null) return;
  countdownTimer = window.setInterval(() => {
    const text = currentCountdownText();
    countdownDoc?.querySelectorAll<HTMLElement>('.js-countdown').forEach((el) => {
      el.textContent = text;
    });
  }, 1000);
}

/** ラベルを書き込む。{countdown} の部分だけは毎秒書き換えられる要素にする */
function writeLabel(el: HTMLElement, comp: AppComponent, state: AppState): void {
  // 判定用に裏で描画するときは、残り時間を最初の値で止め、見えている画面の時計も動かさない
  if (measuringRender) {
    el.textContent = renderLabel(comp, state);
    return;
  }
  if (!comp.label.includes('{countdown}')) {
    el.textContent = renderLabel(comp, state);
    return;
  }
  el.textContent = '';
  const doc = el.ownerDocument;
  comp.label.split('{countdown}').forEach((part, i) => {
    if (i > 0) {
      const span = doc.createElement('span');
      span.className = 'js-countdown';
      span.textContent = currentCountdownText();
      el.appendChild(span);
    }
    el.appendChild(doc.createTextNode(renderTemplate(part, state, comp.style.numberFormat)));
  });
  startCountdownTicker(doc);
}

/**
 * 文字の基準（スマホ画面の幅に対する割合）。中身が収まらないときは下限まで少しずつ小さくする。
 * 合計と購入ボタンは画面下に固定されているため、下限まで縮めても収まらない分はスクロールにする。
 */
const ROOT_FONT_VW_MAX = 4.2;
const ROOT_FONT_VW_MIN = 3.7;

/** 下にまだ続きがあるときは本文の下端をぼかし、スクロールできることを示す */
function updateScrollHint(scroller: HTMLElement): void {
  const hasMore = scroller.scrollTop + scroller.clientHeight < scroller.scrollHeight - 2;
  scroller.classList.toggle('has-more', hasMore);
}

function fitContentToScreen(doc: Document, scroller: HTMLElement): void {
  if (!scroller.isConnected) return;
  const root = doc.documentElement;
  let vw = ROOT_FONT_VW_MAX;
  root.style.fontSize = `${vw}vw`;
  while (scroller.scrollHeight > scroller.clientHeight + 1 && vw > ROOT_FONT_VW_MIN) {
    vw = Math.max(ROOT_FONT_VW_MIN, vw - 0.1);
    root.style.fontSize = `${vw}vw`;
  }
  updateScrollHint(scroller);
}

/**
 * 描画した直後はレイアウトが確定しておらず正しく測れないことがあるため、
 * 次のフレームと、スマホの大きさや本文の枠の大きさが変わったときにも測り直す。
 */
let fitTarget: { doc: Document; scroller: HTMLElement } | null = null;
let fitObserver: ResizeObserver | null = null;
/**
 * 本文の枠を見張る。スタイルが iframe に当たる前は、枠が中身の高さまで伸びていて「収まっている」と測ってしまう。
 * Firefox ではそのまま文字が大きく残り、広告が出て描き直したときに急に小さくなっていた。
 * スタイルが当たって枠が縮んだところで測り直す。iframe の中の要素なので、iframe 側の ResizeObserver を使う
 */
let scrollerObserver: { doc: Document; observer: ResizeObserver } | null = null;

function observeScroller(doc: Document, scroller: HTMLElement): void {
  const IframeResizeObserver = doc.defaultView?.ResizeObserver;
  if (!IframeResizeObserver) return;
  if (scrollerObserver?.doc !== doc) {
    scrollerObserver?.observer.disconnect();
    scrollerObserver = {
      doc,
      observer: new IframeResizeObserver(() => refitVisibleSandbox(doc)),
    };
  } else {
    scrollerObserver.observer.disconnect();
  }
  scrollerObserver.observer.observe(scroller);
}

/** 見えているスマホ画面を測り直す。判定用に裏で描画した画面は対象にしない */
function refitVisibleSandbox(doc: Document): void {
  if (fitTarget?.doc === doc) fitContentToScreen(fitTarget.doc, fitTarget.scroller);
}

function scheduleFit(
  iframe: HTMLIFrameElement | null,
  doc: Document,
  scroller: HTMLElement,
  restoreScrollTop = 0,
): void {
  fitTarget = { doc, scroller };
  fitContentToScreen(doc, scroller);
  scroller.scrollTop = restoreScrollTop;
  window.requestAnimationFrame(() => {
    if (fitTarget?.scroller !== scroller) return;
    fitContentToScreen(doc, scroller);
    scroller.scrollTop = restoreScrollTop;
    updateScrollHint(scroller);
  });
  if (iframe && !fitObserver) {
    fitObserver = new ResizeObserver(() => {
      if (fitTarget) fitContentToScreen(fitTarget.doc, fitTarget.scroller);
    });
    fitObserver.observe(iframe);
  }
  if (iframe) observeScroller(doc, scroller);
}

/** 判定用に裏で描画している最中か。見えている画面のタイマーや監視を、裏の描画で書き換えないために使う */
let measuringRender = false;

interface RenderSandboxOptions {
  /**
   * 判定用の座標を測るための、画面外での描画。広告や完了ページのタイマー、残り時間の時計、
   * スタイルの監視、大きさの測り直しの予約をせず、見えている画面の状態に触れない
   */
  measureOnly?: boolean;
}

export function renderSandboxApp(
  container: HTMLElement,
  spec: AppSpec,
  liveState: AppState,
  selectedComponentIds: Set<string>,
  callbacks: SandboxCallbacks,
  options: RenderSandboxOptions = {},
): void {
  if (options.measureOnly) {
    measuringRender = true;
    try {
      renderSandboxAppInner(container, spec, liveState, selectedComponentIds, callbacks, true);
    } finally {
      measuringRender = false;
    }
    return;
  }
  renderSandboxAppInner(container, spec, liveState, selectedComponentIds, callbacks, false);
}

function renderSandboxAppInner(
  container: HTMLElement,
  spec: AppSpec,
  liveState: AppState,
  selectedComponentIds: Set<string>,
  callbacks: SandboxCallbacks,
  measureOnly: boolean,
): void {
  let previousScrollTop = 0;
  // 単一HTMLとしてiframe内で隔離実行する
  let iframe = container.querySelector<HTMLIFrameElement>('iframe.sandbox-iframe');
  if (!iframe) {
    container.innerHTML = '';
    iframe = document.createElement('iframe');
    iframe.className = 'sandbox-iframe';
    iframe.setAttribute('sandbox', 'allow-same-origin');
    iframe.title = spec.title || 'お題アプリ';
    container.appendChild(iframe);
  }

  const iframeDoc = iframe.contentDocument;
  const targetDoc = iframeDoc ?? document;
  let mountRoot: HTMLElement = container;

  if (iframeDoc) {
    if (!iframeDoc.getElementById('sandbox-mount')) {
      iframeDoc.open();
      iframeDoc.write(buildStandaloneSandboxHtml(spec));
      iframeDoc.close();
      syncStylesIntoIframe(iframeDoc);
      if (!measureOnly) {
        watchParentStyles(iframe);
        refitWhenFontsLoad(iframeDoc);
      }
    }
    mountRoot = iframeDoc.getElementById('sandbox-mount') ?? iframeDoc.body;
    // 状態が変わるたびに中身を作り直すため、スクロール位置を覚えておき、作り直したあとに戻す
    previousScrollTop = mountRoot.querySelector<HTMLElement>('.sandbox-app-body')?.scrollTop ?? 0;
    mountRoot.innerHTML = '';
  } else {
    container.innerHTML = '';
  }

  if (!measureOnly) {
    schedulePopups(spec, liveState, selectedComponentIds, callbacks);
    schedulePageTransition(spec, liveState, callbacks);
  }

  const phoneScreen = targetDoc.createElement('div');
  phoneScreen.className = `phone-app-screen ${spec.theme.chaoticLayout ? 'is-chaotic' : 'is-clean'}`;
  phoneScreen.style.backgroundColor = sanitizeCssColor(spec.theme?.bgColor, '#ffffff');
  phoneScreen.style.color = sanitizeCssColor(spec.theme?.textColor, '#1c1917');
  if (spec.theme?.accentColor) {
    phoneScreen.style.setProperty(
      '--app-accent',
      sanitizeCssColor(spec.theme.accentColor, '#5b3a26'),
    );
  }

  const statusBar = targetDoc.createElement('div');
  statusBar.className = 'phone-status-bar';
  statusBar.innerHTML = `
    <span class="phone-status-time">12:30</span>
    <span class="phone-status-island" aria-hidden="true"></span>
    <span class="phone-status-icons" aria-hidden="true">
      <span class="phone-signal"><i></i><i></i><i></i><i></i></span>
      <span class="phone-battery"><i></i></span>
    </span>
  `;
  phoneScreen.appendChild(statusBar);

  // 購入完了・別ページへの移動中は、そのページだけを見せる
  const page = liveState[PAGE_KEY];
  if (page) {
    targetDoc.documentElement.style.fontSize = `${ROOT_FONT_VW_MAX}vw`;
    phoneScreen.appendChild(
      renderTransientPage(
        targetDoc,
        page === 'complete' ? 'complete' : 'away',
        String(liveState[PAGE_TITLE_KEY] ?? ''),
        liveState[PAGE_PHASE_KEY] === 'loading',
      ),
    );
    mountRoot.appendChild(phoneScreen);
    return;
  }

  // 上部のナビゲーション（戻る・見出し・安全な接続の印）。見た目だけで操作はできない
  const header = targetDoc.createElement('div');
  header.className = 'sandbox-app-header';
  header.innerHTML = `
    <span class="app-nav-back" aria-hidden="true">${CHEVRON_SVG}</span>
    <h3 class="sandbox-app-title"></h3>
    <span class="app-nav-lock" aria-hidden="true">${LOCK_SVG}</span>
  `;
  header.querySelector('.sandbox-app-title')!.textContent = spec.title;
  phoneScreen.appendChild(header);

  // 画面下に固定するバー（合計金額と購入ボタンなど）
  const stickyBar = targetDoc.createElement('div');
  stickyBar.className = 'sandbox-sticky-bar';

  const overlaySlot = targetDoc.createElement('div');
  overlaySlot.className = 'phone-inapp-overlay-slot';

  const body = targetDoc.createElement('div');
  body.className = 'sandbox-app-body';

  const executeActions = (actions: ComponentAction[]) => {
    const result = applyActions(spec, liveState, actions);
    // 購入完了・別ページへの移動はページとして見せ、少しすると最初の画面に戻る
    if (result.completed || result.openedPage) {
      if (result.openedPage) {
        for (const m of result.messages) {
          callbacks.onNotice(m.text, m.type);
        }
      }
      const completion = actions.find((a) => a.type === 'complete_task');
      callbacks.onStateChange({
        ...result.state,
        [PAGE_KEY]: result.completed ? 'complete' : 'away',
        [PAGE_TITLE_KEY]: result.completed
          ? completion?.message || '完了しました'
          : (result.openedPage ?? ''),
        [PAGE_PHASE_KEY]: 'show',
      });
      return;
    }
    for (const m of result.messages) {
      callbacks.onNotice(m.text, m.type);
    }
    callbacks.onStateChange(result.state);
  };

  const setValue = (key: string, raw: string | boolean) => {
    if (!isSafeStateKey(key)) return;
    const value = typeof raw === 'boolean' ? raw : coerceStateValue(spec, key, raw);
    callbacks.onStateChange(withDerivedValues(spec, { ...liveState, [key]: value }));
  };

  /**
   * サンドボックス内でカスタムHTMLを安全に描画し、data-* 属性による状態変更・タスク完了をバインドする。
   * 不活性な template 要素で構文解析してから属性・タグを無害化することで、解析時のイベント発火を防ぐ。
   */
  const renderCustomContent = (comp: AppComponent, rawHtml: string): HTMLElement => {
    const wrapper = targetDoc.createElement('div');
    wrapper.className = 'comp-custom-sandbox';
    const templated = sanitizeCustomHtml(renderTemplate(rawHtml, liveState, comp.style.numberFormat));
    const tpl = targetDoc.createElement('template');
    tpl.innerHTML = templated;
    const frag = tpl.content;
    frag.querySelectorAll(FORBIDDEN_CUSTOM_HTML_TAGS.join(', ')).forEach((node) => node.remove());
    frag.querySelectorAll('*').forEach((node) => {
      for (const attr of Array.from(node.attributes)) {
        const attrName = attr.name.toLowerCase();
        if (attrName === 'style') {
          const cleanedCss = sanitizeCustomCss(attr.value);
          if (cleanedCss) {
            node.setAttribute('style', cleanedCss);
          } else {
            node.removeAttribute(attr.name);
          }
          continue;
        }
        if (isForbiddenCustomHtmlAttr(attrName, attr.value)) {
          node.removeAttribute(attr.name);
        }
      }
    });
    wrapper.appendChild(frag);

    // 状態変数と双方向バインドされた input / select / textarea
    wrapper.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>('[data-state-key]').forEach((el) => {
      const key = el.dataset.stateKey?.trim();
      if (!key || !isSafeStateKey(key)) return;
      if (el instanceof HTMLInputElement && (el.type === 'checkbox' || el.type === 'radio')) {
        const expected = el.value && el.value !== 'on' ? el.value : 'true';
        el.checked = String(liveState[key]) === expected;
        el.addEventListener('change', () => {
          setValue(key, el.value && el.value !== 'on' ? el.value : el.checked);
        });
      } else {
        if (liveState[key] !== undefined) el.value = String(liveState[key]);
        el.addEventListener('change', () => {
          setValue(key, el.value);
        });
      }
    });

    // クリックによる状態更新・アクション実行バインド
    wrapper
      .querySelectorAll<HTMLElement>(
        '[data-set-state], [data-toggle-state], [data-increment-state], [data-complete-task], [data-close-popup]',
      )
      .forEach((el) => {
        el.addEventListener('click', (e) => {
          e.stopPropagation();
          if (el.hasAttribute('data-close-popup') && comp.stateKey && isSafeStateKey(comp.stateKey)) {
            callbacks.onStatePatch({ [comp.stateKey]: 'closed' });
            return;
          }
          const acts = buildActionsFromDataAttrs(spec, el.dataset);
          if (acts.length > 0) {
            executeActions(acts);
          }
        });
      });

    return wrapper;
  };

  const openConfirmOverlay = (comp: AppComponent, onConfirm: () => void) => {
    overlaySlot.innerHTML = '';
    const sheet = targetDoc.createElement('div');
    sheet.className = 'inapp-sheet';
    const titleDiv = targetDoc.createElement('div');
    titleDiv.className = 'inapp-sheet-title';
    titleDiv.textContent = comp.behavior?.confirmText || 'この内容でよろしいですか？';
    sheet.appendChild(titleDiv);
    const row = targetDoc.createElement('div');
    row.className = 'inapp-confirm-row';

    const noBtn = targetDoc.createElement('button');
    noBtn.type = 'button';
    noBtn.className = 'inapp-sheet-cancel';
    noBtn.textContent = 'もどる';
    noBtn.addEventListener('click', () => {
      overlaySlot.innerHTML = '';
    });

    const yesBtn = targetDoc.createElement('button');
    yesBtn.type = 'button';
    yesBtn.className = 'inapp-confirm-yes';
    yesBtn.textContent = 'はい';
    yesBtn.addEventListener('click', () => {
      overlaySlot.innerHTML = '';
      onConfirm();
    });

    row.append(noBtn, yesBtn);
    sheet.appendChild(row);
    overlaySlot.appendChild(sheet);
  };

  const renderPopup = (comp: AppComponent) => {
    const backdrop = targetDoc.createElement('div');
    backdrop.className = 'inapp-popup-backdrop';

    const closePos = comp.closePosition ?? 'top-right';
    const closeSize = comp.closeSize ?? comp.style.size;
    const defaultCloseLabel = closePos === 'bottom-bar' ? '閉じる' : '×';
    const closeLabel = (comp.closeLabel ?? defaultCloseLabel).trim() || defaultCloseLabel;

    const card = targetDoc.createElement('div');
    card.className = [
      'inapp-popup-card',
      `variant-${comp.style.variant}`,
      `close-pos-${closePos}`,
      comp.style.skin === 'intrusive' ? 'is-intrusive' : '',
      selectedComponentIds.has(comp.id) ? 'is-circled-target' : '',
      callbacks.justChangedIds?.has(comp.id) ? 'is-just-changed' : '',
    ]
      .filter(Boolean)
      .join(' ');
    card.dataset.componentId = comp.id;
    card.dataset.role = comp.role;
    if (comp.customCss) {
      const safeCss = sanitizeCustomCss(comp.customCss);
      if (safeCss) card.style.cssText += `;${safeCss}`;
    }

    const closeBtn = targetDoc.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = [
      'inapp-popup-close',
      `size-${closeSize}`,
      `pos-${closePos}`,
      closeLabel.length > 1 ? 'has-label' : '',
    ]
      .filter(Boolean)
      .join(' ');
    closeBtn.setAttribute('aria-label', '閉じる');
    closeBtn.textContent = closeLabel;
    closeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (comp.stateKey) callbacks.onStatePatch({ [comp.stateKey]: 'closed' });
    });

    if (comp.customHtml) {
      card.append(closeBtn, renderCustomContent(comp, comp.customHtml));
    } else {
      const text = targetDoc.createElement('div');
      text.className = 'inapp-popup-text';
      text.textContent = renderLabel(comp, liveState);
      card.append(closeBtn, text);
      if (comp.caption) {
        const sub = targetDoc.createElement('div');
        sub.className = 'inapp-popup-sub';
        sub.textContent = renderTemplate(comp.caption, liveState, comp.style.numberFormat);
        card.appendChild(sub);
      }
      if (comp.actions.some((a) => a.type === 'open_page')) {
        const go = targetDoc.createElement('div');
        go.className = 'inapp-popup-go';
        go.textContent = '今すぐチェック ＞＞';
        card.appendChild(go);
      } else if (comp.actions.length > 0) {
        const actionBtn = targetDoc.createElement('button');
        actionBtn.type = 'button';
        actionBtn.className = 'inapp-popup-action-btn';
        actionBtn.textContent = 'この特典を使う';
        actionBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          executeActions(comp.actions);
          if (comp.stateKey) callbacks.onStatePatch({ [comp.stateKey]: 'closed' });
        });
        card.appendChild(actionBtn);
      }
    }
    card.addEventListener('click', () => {
      if (comp.actions.length > 0) {
        executeActions(comp.actions);
        if (!comp.actions.some((a) => a.type === 'open_page') && comp.stateKey) {
          callbacks.onStatePatch({ [comp.stateKey]: 'closed' });
        }
      }
    });
    backdrop.appendChild(card);
    overlaySlot.appendChild(backdrop);
  };

  for (const comp of spec.components) {
    if (comp.type === 'popup') {
      if (comp.stateKey && liveState[comp.stateKey] === 'open') renderPopup(comp);
      continue;
    }
    if (!isComponentVisible(comp, liveState)) {
      continue;
    }

    const itemWrap = targetDoc.createElement('div');
    itemWrap.className = [
      'sandbox-comp-item',
      `type-${comp.type}`,
      `size-${comp.style.size}`,
      `variant-${comp.style.variant}`,
      `align-${comp.style.align || 'left'}`,
      comp.style.fullWidth ? 'is-full' : '',
      comp.style.noisyAnimation ? 'is-noisy' : '',
      comp.style.skin === 'intrusive' ? 'is-intrusive' : '',
      comp.style.sticky ? 'is-sticky' : '',
      selectedComponentIds.has(comp.id) ? 'is-circled-target' : '',
      callbacks.justChangedIds?.has(comp.id) ? 'is-just-changed' : '',
    ]
      .filter(Boolean)
      .join(' ');
    itemWrap.dataset.componentId = comp.id;
    itemWrap.dataset.role = comp.role;
    if (comp.style.rotateDeg) {
      itemWrap.style.transform = `rotate(${comp.style.rotateDeg}deg)`;
    }
    if (comp.customCss) {
      const safeCss = sanitizeCustomCss(comp.customCss);
      if (safeCss) itemWrap.style.cssText += `;${safeCss}`;
    }
    const label = renderLabel(comp, liveState);

    const caption = comp.caption
      ? renderTemplate(comp.caption, liveState, comp.style.numberFormat)
      : '';

    if (comp.customHtml || comp.type === 'custom') {
      itemWrap.appendChild(
        renderCustomContent(
          comp,
          comp.customHtml || `<div class="comp-custom-fallback">${escapeHtml(label)}</div>`,
        ),
      );
      if (comp.actions.length > 0 && !comp.customHtml?.includes('data-')) {
        itemWrap.style.cursor = 'pointer';
        itemWrap.addEventListener('click', () => executeActions(comp.actions));
      }
    } else if (comp.type === 'product_card') {
      itemWrap.appendChild(renderProductCard(targetDoc, comp, liveState, label, caption, setValue));
    } else if (comp.type === 'progress_steps') {
      itemWrap.appendChild(
        renderProgressSteps(
          targetDoc,
          comp,
          resolveProgressCurrentStep(spec, comp, label, liveState),
        ),
      );
    } else if (comp.type === 'choice_cards') {
      itemWrap.appendChild(renderChoiceCards(targetDoc, comp, liveState, setValue));
    } else if (comp.type === 'summary') {
      const list = targetDoc.createElement('div');
      list.className = 'comp-summary';
      for (const row of comp.rows ?? []) {
        if (row.visibleWhen && !matchesCondition(row.visibleWhen, liveState)) continue;
        const line = targetDoc.createElement('div');
        line.className = 'comp-summary-row';
        const k = targetDoc.createElement('span');
        k.textContent = renderTemplate(row.label, liveState, comp.style.numberFormat);
        const v = targetDoc.createElement('span');
        v.textContent = renderTemplate(row.value, liveState, comp.style.numberFormat);
        line.append(k, v);
        list.appendChild(line);
      }
      itemWrap.appendChild(list);
    } else if (comp.type === 'product_hero') {
      const hero = targetDoc.createElement('div');
      hero.className = 'comp-product-hero';
      const icon = targetDoc.createElement('div');
      icon.className = 'comp-product-icon';
      icon.textContent = comp.icon ?? '🛍️';
      const name = targetDoc.createElement('div');
      name.className = 'comp-product-name';
      name.textContent = label;
      hero.append(icon, name);
      itemWrap.appendChild(hero);
    } else if (comp.type === 'select') {
      const field = targetDoc.createElement('label');
      field.className = 'comp-field';
      const labelEl = targetDoc.createElement('span');
      labelEl.className = 'comp-field-label';
      labelEl.textContent = label;
      const select = targetDoc.createElement('select');
      select.className = 'comp-select';
      const current = comp.stateKey ? String(liveState[comp.stateKey] ?? '') : '';
      for (const opt of comp.options ?? []) {
        const option = targetDoc.createElement('option');
        option.value = opt;
        option.textContent = opt;
        option.selected = opt === current;
        select.appendChild(option);
      }
      select.addEventListener('change', () => {
        if (comp.stateKey) setValue(comp.stateKey, select.value);
      });
      field.append(labelEl, select);
      itemWrap.appendChild(field);
    } else if (comp.type === 'input') {
      const field = targetDoc.createElement('label');
      field.className = 'comp-field';
      const labelEl = targetDoc.createElement('span');
      labelEl.className = 'comp-field-label';
      labelEl.textContent = label;
      const input = targetDoc.createElement('input');
      input.type = 'text';
      input.className = 'comp-input';
      input.placeholder = comp.placeholder ?? '';
      input.value = comp.stateKey ? String(liveState[comp.stateKey] ?? '') : '';
      // 入力のたびに画面を描き直すと入力中の文字が消えるため、確定時（フォーカスが外れたとき）に反映する
      input.addEventListener('change', () => {
        if (comp.stateKey) setValue(comp.stateKey, input.value);
      });
      field.append(labelEl, input);
      if (caption) {
        const note = targetDoc.createElement('span');
        note.className = 'comp-field-note';
        note.textContent = caption;
        field.appendChild(note);
      }
      itemWrap.appendChild(field);
    } else if (comp.type === 'checkbox' || comp.type === 'toggle') {
      const isChecked = comp.stateKey ? Boolean(liveState[comp.stateKey]) : false;
      const row = targetDoc.createElement('label');
      row.className = comp.type === 'checkbox' ? 'comp-checkbox-row' : 'ios-toggle-row';
      if (comp.type === 'checkbox') {
        const box = targetDoc.createElement('input');
        box.type = 'checkbox';
        box.className = 'comp-checkbox';
        box.checked = isChecked;
        box.addEventListener('change', () => {
          if (comp.stateKey) setValue(comp.stateKey, box.checked);
        });
        const text = targetDoc.createElement('span');
        text.textContent = label;
        row.append(box, text);
      } else {
        const text = targetDoc.createElement('span');
        text.className = 'ios-toggle-label';
        text.textContent = label;
        const pill = targetDoc.createElement('button');
        pill.type = 'button';
        pill.className = `ios-switch ${isChecked ? 'is-on' : ''}`;
        pill.innerHTML = '<span class="ios-switch-knob"></span>';
        pill.addEventListener('click', () => {
          if (comp.stateKey) setValue(comp.stateKey, !isChecked);
        });
        row.append(text, pill);
      }
      itemWrap.appendChild(row);
    } else if (comp.type === 'slider') {
      const field = targetDoc.createElement('label');
      field.className = 'comp-field';
      const current = Number(comp.stateKey ? liveState[comp.stateKey] : 0) || 0;
      const head = targetDoc.createElement('span');
      head.className = 'comp-field-label';
      head.textContent = `${label} ${current}`;
      const range = targetDoc.createElement('input');
      range.type = 'range';
      range.min = '0';
      range.max = '100';
      range.value = String(current);
      range.className = 'comp-range-slider';
      range.addEventListener('change', () => {
        if (comp.stateKey) setValue(comp.stateKey, range.value);
      });
      field.append(head, range);
      itemWrap.appendChild(field);
    } else if (comp.type === 'status_banner') {
      const banner = targetDoc.createElement('div');
      banner.className = 'comp-status-banner';
      writeLabel(banner, comp, liveState);
      if (caption) {
        const sub = targetDoc.createElement('div');
        sub.className = 'comp-caption';
        sub.textContent = caption;
        banner.appendChild(sub);
      }
      if (comp.actions.length > 0) {
        banner.style.cursor = 'pointer';
        banner.addEventListener('click', () => executeActions(comp.actions));
      }
      itemWrap.appendChild(banner);
    } else if (comp.type === 'button') {
      const btn = targetDoc.createElement('button');
      btn.type = 'button';
      btn.className = 'comp-button';
      btn.textContent = label;

      let clickCount = 0;
      let escapedOnce = false;

      btn.addEventListener('click', () => {
        const effect = comp.behavior?.effect ?? 'none';

        if (effect === 'escape_on_hover' && !escapedOnce) {
          escapedOnce = true;
          // スマホの大きさに合わせて逃げる距離も変わるよう rem で動かす
          const offsetX = (Math.random() > 0.5 ? 1 : -1) * 2.6;
          btn.style.transform = `translate(${offsetX}rem, -0.6rem)`;
          callbacks.onNotice('ボタンが逃げました。もう一度押してください', 'info');
          return;
        }
        btn.style.transform = '';

        if (effect === 'require_3_clicks') {
          clickCount++;
          if (clickCount < 3) {
            btn.textContent = `${label} あと${3 - clickCount}回`;
            return;
          }
          clickCount = 0;
          btn.textContent = label;
        }

        if (comp.actions.length === 0) {
          callbacks.onNotice(`${label}を押しても反応がありません`, 'warning');
          return;
        }

        // 本物の購入画面と同じく、必須の項目が空のまま先へは進ませない
        if (comp.role === 'primary_action' || hasCompletionAction(comp)) {
          const missing = spec.components.find(
            (c) =>
              c.role === 'required_input' &&
              isComponentVisible(c, liveState) &&
              !isInputFilled(c, liveState),
          );
          if (missing) {
            const verb = missing.type === 'input' ? '入力' : '選択';
            callbacks.onNotice(`${renderLabel(missing, liveState)}を${verb}してください`, 'warning');
            const missingEl = findSandboxComponentElement(body, missing.id);
            if (missingEl) {
              missingEl.classList.remove('is-missing');
              void missingEl.offsetWidth;
              missingEl.classList.add('is-missing');
              missingEl.scrollIntoView({ block: 'center', behavior: 'smooth' });
            }
            return;
          }
        }

        if (effect === 'confirm_dialog') {
          openConfirmOverlay(comp, () => executeActions(comp.actions));
          return;
        }

        executeActions(comp.actions);
      });

      itemWrap.appendChild(btn);
    } else {
      if (caption) {
        const cap = targetDoc.createElement('div');
        cap.className = 'comp-caption';
        cap.textContent = caption;
        itemWrap.appendChild(cap);
      }
      const txt = targetDoc.createElement('div');
      txt.className = comp.type === 'heading' ? 'comp-heading' : 'comp-text';
      writeLabel(txt, comp, liveState);
      itemWrap.appendChild(txt);
    }

    (comp.style.sticky ? stickyBar : body).appendChild(itemWrap);
  }

  phoneScreen.append(body);
  if (stickyBar.childElementCount > 0) phoneScreen.append(stickyBar);
  phoneScreen.append(overlaySlot);
  mountRoot.appendChild(phoneScreen);
  if (measureOnly) {
    // 判定用の描画は、決まった大きさに合わせて文字の基準を決め、スクロールは先頭にする
    fitContentToScreen(targetDoc, body);
    body.scrollTop = 0;
    return;
  }
  body.addEventListener(
    'scroll',
    () => {
      updateScrollHint(body);
      callbacks.onScroll?.();
    },
    { passive: true },
  );
  scheduleFit(iframeDoc ? iframe : null, targetDoc, body, previousScrollTop);
}

/** 判定用の描画で、文字の基準を測り直す。スタイルや文字の読み込みを待ったあとに呼ぶ */
export function refitMeasuredSandbox(container: HTMLElement): void {
  const iframe = container.querySelector<HTMLIFrameElement>('iframe.sandbox-iframe');
  const doc = iframe?.contentDocument;
  const body = doc?.querySelector<HTMLElement>('.sandbox-app-body');
  if (!doc || !body) return;
  fitContentToScreen(doc, body);
  body.scrollTop = 0;
}

/**
 * スマホ画面の本文を縦にスクロールする。赤ペンで囲むときは画面の上に透明な描画面が重なり、
 * ホイールが本文に届かないため、描画面で受けたホイールをこちらに渡す。
 */
export function scrollSandboxBy(container: HTMLElement, deltaY: number): boolean {
  const iframe = container.querySelector<HTMLIFrameElement>('iframe.sandbox-iframe');
  const doc = iframe?.contentDocument ?? document;
  const body = doc.querySelector<HTMLElement>('.sandbox-app-body');
  if (!body) return false;
  const before = body.scrollTop;
  body.scrollTop += deltaY;
  return body.scrollTop !== before;
}

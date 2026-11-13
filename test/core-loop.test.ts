import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  AiNotConfiguredError,
  CIRCLE_INQUIRY_FOLLOW_UPS,
  applyOptionToSpec,
  applyPatchesToSpecWithValidation,
  canApplyWithoutRedPen,
  chatWithStudio,
  describeOwnerContext,
  evaluateAppWithPersonas,
  getAiConnectionStatus,
  normalizeUpdatedSpec,
  recordHumanDecisionWithFunctionCalling,
  relaxUnrequestedRequiredChoices,
  simulateReviewers,
} from '../src/server/gemini.ts';
import {
  applyActions,
  createInitialState,
  evaluateExpression,
  matchesCondition,
  renderLabel,
  renderTemplate,
  toKanjiNumber,
  withDerivedValues,
} from '../src/shared/appRuntime.ts';
import { clipToVisibleRects, detectHitComponents, type Rect2D } from '../src/shared/hitTest.ts';
import {
  isValidAppSpec,
  sanitizeBase64,
  sanitizeComponentBounds,
  sanitizeComponentIds,
} from '../src/server/api.ts';
import {
  ConcurrencyGate,
  DailyCounter,
  DailyLimitError,
  LARGE_BODY_BYTES,
  RateLimiter,
  SMALL_BODY_BYTES,
  clientKeyOf,
  isAllowedOrigin,
  isBotUserAgent,
  maxBodyBytesFor,
  rateLimitGroupOf,
} from '../src/server/rateLimit.ts';
import { PromptRejectedError, assertPromptAllowed, findPersonalInfoByRules, reasonFromModelArmorResult, stripThemeSamplePhrases } from '../src/server/modelArmor.ts';
import {
  ACTIVE_THEME,
  ALL_REVIEWER_PERSONAS,
  TARGET_COHORTS,
  getReviewersByCohort,
} from '../src/shared/initialTheme.ts';
import {
  sanitizeCssColor,
  sanitizeCustomCss,
  sanitizeCustomHtml,
  COUPON_FRICTIONS,
  shouldShowNextFrictionNotice,
  simulatePersonaComputerUse,
  validateSandboxSpec,
} from '../src/shared/sandboxValidator.ts';
import {
  classifyUserMessageIntent,
  detectCheckpointRevertCommand,
  detectExplicitRecipientMention,
} from '../src/shared/intent.ts';
import {
  formatPredictionBadgeText,
  formatPredictionReportLine,
  scorePredictionRound,
  summarizePredictions,
} from '../src/shared/prediction.ts';
import { describeOtherCohortImpact, describeOtherCohortReport } from '../src/shared/otherCohorts.ts';
import { splitIntoChatChunks } from '../src/shared/chatText.ts';
import {
  buildOwnerChangeReaction,
  buildOwnerClosingLines,
  buildOwnerReport,
} from '../src/shared/owner.ts';
import { applySpecPatches } from '../src/shared/specPatch.ts';
import {
  buildClearFeedbackFacts,
  classifyDeterministicArchetype,
  cosineSimilarity,
  extractDeterministicNlSignal,
  fillTurnRefs,
  formatPublicAggregateStats,
  inspectAppliedChangeForTelemetry,
  limitEntitiesToVocabulary,
  summarizePromptTelemetry,
} from '../src/shared/promptTelemetry.ts';
import {
  classifyPromptWithAgentPlatformEmbeddings,
  clearPromptTelemetryEvents,
  finalizeTelemetryOnEvaluate,
  mergeTelemetryEvents,
  parseFirestoreTelemetryDoc,
  recordPromptTelemetryEvent,
  toFirestoreTelemetryFields,
} from '../src/server/store.ts';
import { buildInitialSpec, buildScenarioSpec, findComponent } from '../src/shared/theme.ts';
import { describeComponentFunctionName } from '../src/shared/workSteps.ts';
import { readFileSync } from 'node:fs';
import {
  clearCustomProfileMemoryCache,
  resolveCustomComponentProfiles,
} from '../src/server/customComponentProfiles.ts';
import {
  parseCustomHtmlActions,
  prepareSimulationInput,
  type CustomComponentProfile,
} from '../src/shared/customComponentProfile.ts';
import {
  FIXED_LAYOUT_VIEWPORT,
  canonicalizeMeasuredBounds,
  fixedLayoutStateVariants,
  mergeBoundsVariants,
  type MeasuredComponentRect,
} from '../src/shared/fixedLayout.ts';
import type {
  AppComponent,
  AppSpec,
  ComponentBoundsInput,
  ReviewOutcome,
  SpecPatch,
} from '../src/shared/types.ts';

// テストは AI に接続しない前提で実行する（package.json の test スクリプトで接続設定を空にしている）

function persona(id: string) {
  const p = ALL_REVIEWER_PERSONAS.find((x) => x.id === id);
  if (!p) throw new Error(`persona not found: ${id}`);
  return p;
}

function outcomeOf(spec: AppSpec, personaId: string): ReviewOutcome {
  return simulatePersonaComputerUse(spec, persona(personaId)).outcome;
}

/** 妨害要素を入れる前の、まともに動く購入画面 */
function cleanSpec(): AppSpec {
  return structuredClone(ACTIVE_THEME.baseSpec);
}

function withAnnoyance(spec: AppSpec, id: string): AppSpec {
  const annoyance = ACTIVE_THEME.annoyances.find((a) => a.id === id);
  if (!annoyance) throw new Error(`annoyance not found: ${id}`);
  annoyance.apply(spec);
  return spec;
}

describe('core-loop', () => {
  it('ペン囲みの軌跡座標から重なるUI部品IDを特定できる', () => {
    const rects: Rect2D[] = [
      { id: 'product-hero', left: 20, top: 20, right: 300, bottom: 70 },
      { id: 'btn-clear-cart', left: 20, top: 180, right: 300, bottom: 250 },
      { id: 'btn-order', left: 20, top: 270, right: 180, bottom: 310 },
    ];
    const stroke = [
      { x: 15, y: 175 },
      { x: 305, y: 175 },
      { x: 305, y: 255 },
      { x: 15, y: 255 },
      { x: 15, y: 175 },
    ];
    assert.deepEqual(detectHitComponents(stroke, rects), ['btn-clear-cart']);

    // 線が隣の部品に少しはみ出しても、中心まで囲んだ部品だけを選ぶ
    const sloppy = [
      { x: 15, y: 175 },
      { x: 305, y: 175 },
      { x: 305, y: 275 },
      { x: 15, y: 275 },
      { x: 15, y: 175 },
    ];
    assert.deepEqual(detectHitComponents(sloppy, rects), ['btn-clear-cart']);

    // 大きな部品の一部だけを囲んだときは、その部品を選ぶ
    const partial = [
      { x: 30, y: 25 },
      { x: 90, y: 25 },
      { x: 90, y: 50 },
      { x: 30, y: 50 },
    ];
    assert.deepEqual(detectHitComponents(partial, rects), ['product-hero']);
  });

  it('1日の上限・同時に処理する数・本文の大きさ・呼び出し元で、予期しないトラフィックを止める', () => {
    // 1日の上限：日本時間で日付が変わったら数え直す
    let t = Date.parse('2026-11-10T14:00:00Z'); // 日本時間 23:00
    const daily = new DailyCounter(2, () => t);
    daily.consume();
    daily.consume();
    assert.equal(daily.isExhausted(), true);
    assert.throws(() => daily.consume(), DailyLimitError);
    t = Date.parse('2026-11-10T15:00:00Z'); // 日本時間 0:00、翌日
    assert.equal(daily.isExhausted(), false);
    daily.consume();

    // 同時に処理する数：接続元ごとに数え、終わったら空く
    const gate = new ConcurrencyGate(2);
    assert.equal(gate.tryEnter('a'), true);
    assert.equal(gate.tryEnter('a'), true);
    assert.equal(gate.tryEnter('a'), false);
    assert.equal(gate.tryEnter('b'), true);
    gate.leave('a');
    assert.equal(gate.tryEnter('a'), true);

    // 本文の大きさ：画像や音声を送るAPIだけ大きめ
    assert.equal(maxBodyBytesFor('/api/chat'), LARGE_BODY_BYTES);
    assert.equal(maxBodyBytesFor('/api/evaluate'), LARGE_BODY_BYTES);
    assert.equal(maxBodyBytesFor('/api/tts'), SMALL_BODY_BYTES);
    assert.equal(maxBodyBytesFor('/api/run-event'), SMALL_BODY_BYTES);

    // 呼び出し元：このサイト自身か、前に立つ配信が付けた名前と一致するときだけ通す。Origin がない直接通信は Sec-Fetch-Site: same-origin がなければ断る
    const req = (headers: Record<string, string>) => ({ headers }) as never;
    assert.equal(isAllowedOrigin(req({ host: 'app.example.run.app' })), false);
    assert.equal(
      isAllowedOrigin(req({ host: 'app.example.run.app', 'sec-fetch-site': 'same-origin' })),
      true,
    );
    assert.equal(isAllowedOrigin(req({ host: 'app.example.run.app', origin: 'https://app.example.run.app' })), true);
    assert.equal(isAllowedOrigin(req({ host: 'app.example.run.app', origin: 'https://evil.example.com' })), false);
    assert.equal(
      isAllowedOrigin(req({ host: 'app.example.run.app', 'x-forwarded-host': 'studio.web.app', origin: 'https://studio.web.app' })),
      true,
    );
    // 先頭に偽装ホストを付け足した X-Forwarded-Host や、HTTPS 配信への HTTP ダウングレード呼び出しは断る
    assert.equal(
      isAllowedOrigin(
        req({
          host: 'app.example.run.app',
          'x-forwarded-host': 'evil.example.com, studio.web.app',
          origin: 'https://evil.example.com',
        }),
      ),
      false,
    );
    assert.equal(
      isAllowedOrigin(
        req({
          host: 'app.example.run.app',
          'x-forwarded-proto': 'https',
          origin: 'http://app.example.run.app',
        }),
      ),
      false,
    );
    assert.equal(isAllowedOrigin(req({ host: 'localhost:5173', origin: 'http://localhost:5173' })), true);
    assert.equal(isAllowedOrigin(req({ host: 'localhost:5173', origin: 'null' })), false);

    // ボットやスクリプトの User-Agent、または空の User-Agent は断る
    assert.equal(isBotUserAgent(req({})), true);
    assert.equal(isBotUserAgent(req({ 'user-agent': 'curl/8.7.1' })), true);
    assert.equal(isBotUserAgent(req({ 'user-agent': 'python-requests/2.31.0' })), true);
    assert.equal(isBotUserAgent(req({ 'user-agent': 'Mozilla/5.0 (compatible; Googlebot/2.1)' })), true);
    assert.equal(
      isBotUserAgent(
        req({
          'user-agent':
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }),
      ),
      false,
    );
  });

  it('Gemini に送る前に、個人情報らしき内容やAIへの指示の乗っ取りを止める', async () => {
    // コードの規則：メールアドレスと電話番号
    assert.equal(findPersonalInfoByRules('連絡は taro.yamada@example.com までお願いします'), true);
    assert.equal(findPersonalInfoByRules('電話は090-1234-5678です'), true);
    assert.equal(findPersonalInfoByRules('０３－１２３４－５６７８にかけて'), true);
    assert.equal(findPersonalInfoByRules('0312345678'), true);
    // 値段・時刻・割引などは当たらない
    for (const ok of ['500円OFFのクーポンにして', '合計2,580円を大きく', '残り2:59の表示を消して', '1980円の豆を2袋', '定期便のチェックを外して']) {
      assert.equal(findPersonalInfoByRules(ok), false, ok);
    }
    await assert.rejects(assertPromptAllowed('私のメールは a@b.co です'), (err: unknown) => err instanceof PromptRejectedError && err.reason === 'personal_info');
    // Model Armor の設定がないときは、規則に当たらなければ通す
    await assertPromptAllowed('注文ボタンを押したら注文できるようにして');

    // Model Armor の結果の読み方
    const base = { sanitizationResult: { filterMatchState: 'MATCH_FOUND', invocationResult: 'SUCCESS', filterResults: {} as Record<string, unknown> } };
    const withFilter = (key: string, inner: string, value: unknown) => ({
      sanitizationResult: { ...base.sanitizationResult, filterResults: { [key]: { [inner]: value } } },
    });
    assert.equal(reasonFromModelArmorResult(withFilter('sdp', 'sdpFilterResult', { inspectResult: { matchState: 'MATCH_FOUND' } })), 'personal_info');
    assert.equal(reasonFromModelArmorResult(withFilter('pi_and_jailbreak', 'piAndJailbreakFilterResult', { matchState: 'MATCH_FOUND' })), 'prompt_injection');
    assert.equal(reasonFromModelArmorResult(withFilter('rai', 'raiFilterResult', { matchState: 'MATCH_FOUND' })), 'harmful');
    assert.equal(reasonFromModelArmorResult({ sanitizationResult: { filterMatchState: 'NO_MATCH_FOUND' } }), null);
    // 止めたときの文は、引っかかった文字そのものを繰り返さない
    assert.doesNotMatch(new PromptRejectedError('personal_info').message, /@|\d{3}/);
    // 題材の画面に最初から載っている架空のお届け先を引用した発言は、検査の前に取り除く
    assert.equal(
      stripThemeSamplePhrases('お届け先の 〒150-0001 東京都渋谷区神宮前1-2-3 山田 花子 の文字を大きくして'),
      'お届け先の    の文字を大きくして',
    );
  });

  it('Gemini や読み上げを呼ぶAPIは、接続元ごととサーバー全体の両方で1分あたりの回数を超えたら断る', () => {
    let t = 0;
    const limiter = new RateLimiter(
      { gemini: { perClient: 3, global: 5 }, tts: { perClient: 2, global: 10 }, record: { perClient: 2, global: 10 } },
      60_000,
      () => t,
    );
    // 接続元ごとの上限
    for (let i = 0; i < 3; i++) assert.equal(limiter.take('gemini', 'a').ok, true);
    const denied = limiter.take('gemini', 'a');
    assert.equal(denied.ok, false);
    assert.equal(!denied.ok && denied.retryAfterSec, 60);
    // 別の接続元は通るが、サーバー全体の上限に達したら断る
    assert.equal(limiter.take('gemini', 'b').ok, true);
    assert.equal(limiter.take('gemini', 'c').ok, true);
    assert.equal(limiter.take('gemini', 'd').ok, false);
    // 読み上げは別に数える
    assert.equal(limiter.take('tts', 'a').ok, true);
    // 1分たてば、また呼べる
    t = 60_000;
    assert.equal(limiter.take('gemini', 'a').ok, true);
    limiter.prune();

    // 数えるのはPOSTの決まったAPIだけ
    assert.equal(rateLimitGroupOf('POST', '/api/chat'), 'gemini');
    assert.equal(rateLimitGroupOf('POST', '/api/tts'), 'tts');
    assert.equal(rateLimitGroupOf('GET', '/api/status'), null);
    // 接続元は、X-Forwarded-For の最後（受付が付けた値）を使い、送る側が書ける先頭は使わない
    const req = (xff?: string) =>
      ({ headers: xff ? { 'x-forwarded-for': xff } : {}, socket: { remoteAddress: '10.0.0.9' } }) as never;
    assert.equal(clientKeyOf(req('1.1.1.1, 203.0.113.5')), '203.0.113.5');
    assert.equal(clientKeyOf(req()), '10.0.0.9');
  });

  it('画面下に固定した帯の裏に隠れた部品は、帯の注文ボタンを囲んでもいっしょに選ばない', () => {
    // ブラウザで測った配置：お支払い方法（644〜839）の中心が、画面下に固定した合計と注文ボタンの帯の裏にある
    const viewport = { width: 442, height: 806 };
    const layered = [
      { id: 'btn-buy-now', left: 16, top: 342, right: 426, bottom: 417 },
      { id: 'choice-payment', left: 16, top: 644, right: 426, bottom: 839 },
      { id: 'check-subscription', left: 16, top: 851, right: 426, bottom: 865 },
      { id: 'total-price', left: 18, top: 736, right: 75, bottom: 774, occludes: true, pinned: true },
      { id: 'btn-order', left: 89, top: 724, right: 424, bottom: 786, occludes: true, pinned: true },
    ];
    const visible = clipToVisibleRects(layered, viewport);
    // 画面の外の部品は外れ、帯に下の端を隠された部品は見えている部分だけになる
    assert.equal(visible.some((r) => r.id === 'check-subscription'), false);
    assert.deepEqual(visible.find((r) => r.id === 'choice-payment'), {
      id: 'choice-payment',
      left: 16,
      top: 644,
      right: 426,
      bottom: 724,
    });
    // 注文ボタンを楕円で囲む
    const loop = Array.from({ length: 37 }, (_, i) => {
      const a = (i / 36) * 2 * Math.PI;
      return { x: 256 + 190 * Math.cos(a), y: 755 + 45 * Math.sin(a) };
    });
    assert.ok(detectHitComponents(loop, layered).includes('choice-payment'), '隠れた部分を除かないと選ばれてしまう');
    assert.deepEqual(detectHitComponents(loop, visible), ['btn-order']);
    // 見えている部分を囲めば、隠れかけた部品も選べる
    const paymentLoop = [
      { x: 20, y: 650 },
      { x: 420, y: 650 },
      { x: 420, y: 715 },
      { x: 20, y: 715 },
    ];
    assert.deepEqual(detectHitComponents(paymentLoop, visible), ['choice-payment']);
  });

  it('計算式で合計金額を求め、漢数字でも表示できる', () => {
    const spec = cleanSpec();
    const state = createInitialState(spec);
    assert.equal(state.subtotal, 1980);
    assert.equal(state.shipping, 600);
    assert.equal(state.total, 2580);

    // 2袋以上で送料無料
    assert.equal(evaluateExpression('subtotal >= 3960 ? 0 : shippingFee', { subtotal: 3960, shippingFee: 600 }), 0);
    assert.equal(evaluateExpression('(1 + 2) * 3 - 4 / 2', {}), 7);

    assert.equal(toKanjiNumber(1980), '千九百八十');
    assert.equal(toKanjiNumber(3960), '三千九百六十');
    assert.equal(toKanjiNumber(12000), '一万二千');
    assert.equal(toKanjiNumber(0), '零');

    const total = findComponent(spec, 'total-price');
    assert.equal(renderLabel(total, state), '¥2,580');
    // 漢数字の妨害要素を入れると、合計も内訳も漢数字になる
    const kanjiSpec = withAnnoyance(cleanSpec(), 'kanji_price');
    const kanjiTotal = findComponent(kanjiSpec, 'total-price');
    assert.equal(renderLabel(kanjiTotal, state), '二千五百八十円');
    const kanjiSummary = findComponent(kanjiSpec, 'summary');
    const visibleRows = (s: typeof state) =>
      (kanjiSummary.rows ?? [])
        .filter((r) => !r.visibleWhen || matchesCondition(r.visibleWhen, s))
        .map((r) => renderTemplate(r.value, s, kanjiSummary.style.numberFormat));
    assert.deepEqual(visibleRows(state), ['千九百八十円', '六百円']);
    // 代引きを選ぶと、合計に入る手数料の行が内訳に出る
    const codState = withDerivedValues(spec, { ...state, paymentMethod: '代金引換' });
    assert.equal(codState.total, 2910);
    assert.deepEqual(visibleRows(codState), ['千九百八十円', '六百円', '三百三十円']);
    // 偽のカウントダウンは、画面の外では最初の値で表す
    const countdown = findComponent(withAnnoyance(cleanSpec(), 'countdown_pressure'), 'banner-countdown');
    assert.equal(renderLabel(countdown, state), '⏰ 残り 2:59 で売り切れ！！');
  });

  it('初期画面は毎回ランダムに妨害要素を組み込み、同じ乱数の元なら同じ画面になる', () => {
    const a = buildInitialSpec(ACTIVE_THEME, 1);
    const b = buildInitialSpec(ACTIVE_THEME, 1);
    assert.deepEqual(a.appliedAnnoyanceIds, b.appliedAnnoyanceIds);

    const expectedTotal = { easy: [2, 3], normal: [3, 5], hard: [8, 9] } as const;
    for (const level of ['easy', 'normal', 'hard'] as const) {
      const variations = new Set<string>();
      for (let seed = 1; seed <= 30; seed++) {
        const built = buildInitialSpec(ACTIVE_THEME, seed, level);
        variations.add(built.appliedAnnoyanceIds.join(','));
        const ids = built.appliedAnnoyanceIds;
        // 注文ボタンが反応しないことは毎回必ず入る
        assert.ok(ids.includes('broken_order_button'));
        const [min, max] = expectedTotal[level];
        assert.ok(ids.length >= min && ids.length <= max, `${level}:${seed} ${ids.length}個`);
        // 広告は1回に1つまで
        assert.ok(built.spec.components.filter((c) => c.role === 'popup').length <= 1);
        // 画面としては描画でき、完了イベントだけが失われている
        const validation = validateSandboxSpec(built.spec, { requireTaskCompletionEvent: true });
        assert.equal(validation.canRenderHtml, true);
        assert.equal(validation.hasTaskCompletionEvent, false);
        // 初期画面では誰も注文を完了できない
        for (const p of ALL_REVIEWER_PERSONAS) {
          assert.equal(simulatePersonaComputerUse(built.spec, p).outcome, 'abandoned', `${level}:${seed}:${p.id}`);
        }
      }
      assert.ok(variations.size > 10, `${level}: 乱数の元が違えば組み合わせも変わる`);
    }

    // 冗談枠は easy では出ない
    const jokeIds = new Set(ACTIVE_THEME.annoyances.filter((x) => x.kind === 'joke').map((x) => x.id));
    for (let seed = 1; seed <= 30; seed++) {
      const ids = buildInitialSpec(ACTIVE_THEME, seed, 'easy').appliedAnnoyanceIds;
      assert.ok(ids.every((id) => !jokeIds.has(id)));
    }
  });

  it('注文ボタンを直すだけで、誰かは注文できるようになる', () => {
    // 最初の改修で手応えが出ること。normal の画面で注文ボタンだけ直したとき、9人のうち誰かは達成か後悔に届く
    let seedsWithProgress = 0;
    for (let seed = 1; seed <= 30; seed++) {
      const spec = buildInitialSpec(ACTIVE_THEME, seed, 'normal').spec;
      findComponent(spec, 'btn-order').actions = [
        { type: 'complete_task', message: 'ご購入ありがとうございます' },
      ];
      const progressed = ALL_REVIEWER_PERSONAS.some(
        (p) => simulatePersonaComputerUse(spec, p).outcome !== 'abandoned',
      );
      if (progressed) seedsWithProgress++;
    }
    assert.ok(seedsWithProgress >= 25, `30回中${seedsWithProgress}回`);
  });

  it('まともな購入画面なら多くの人が注文でき、確認したい人は確認がないと押せない', () => {
    const spec = cleanSpec();
    assert.equal(outcomeOf(spec, 'kazuko'), 'achieved');
    assert.equal(outcomeOf(spec, 'kenta'), 'achieved');
    assert.equal(outcomeOf(spec, 'shota'), 'achieved');
    assert.equal(outcomeOf(spec, 'misaki'), 'achieved');
    assert.equal(outcomeOf(spec, 'harue'), 'abandoned');
  });

  it('確認を足すと確認したい人は助かるが、急いでいる人は手数が増えて離脱する', () => {
    const spec = cleanSpec();
    findComponent(spec, 'btn-order').behavior = { effect: 'confirm_dialog' };
    assert.equal(outcomeOf(spec, 'harue'), 'achieved');
    assert.equal(outcomeOf(spec, 'kenta'), 'abandoned');

    // 前回と同じお届け先・支払い方法を最初から埋めておけば、両方が注文できる
    for (const v of spec.state) {
      if (v.key === 'deliveryAddress') v.initialValue = '自宅';
      if (v.key === 'paymentMethod') v.initialValue = 'クレジットカード';
    }
    assert.equal(outcomeOf(spec, 'harue'), 'achieved');
    assert.equal(outcomeOf(spec, 'kenta'), 'achieved');
  });

  it('最初からチェックの入った定期便に気づかず注文すると後悔になる', () => {
    const spec = withAnnoyance(cleanSpec(), 'subscription_trap');
    // 細かい字を読み飛ばす人
    assert.equal(outcomeOf(spec, 'hina'), 'regretted');
    // 細かい字は読むが、小さすぎて読めない人
    assert.equal(outcomeOf(spec, 'kazuko'), 'regretted');
    // 細かい字を読み、読める人は気づいて外す。買えはするが、勝手に付いていたことに不満が残る
    assert.equal(outcomeOf(spec, 'shota'), 'bothered');
    assert.equal(outcomeOf(cleanSpec(), 'shota'), 'achieved');
  });

  it('妨害要素ごとに、困る人が変わる', () => {
    // 漢数字の合計は、金額を確かめたい人を離脱させる
    assert.equal(outcomeOf(withAnnoyance(cleanSpec(), 'kanji_price'), 'shota'), 'abandoned');
    // 小さな閉じるボタンの広告は、目が遠い人には閉じられない
    assert.equal(outcomeOf(withAnnoyance(cleanSpec(), 'coupon_popup'), 'kazuko'), 'abandoned');
    // 文字が小さいと読めない
    assert.equal(outcomeOf(withAnnoyance(cleanSpec(), 'tiny_text'), 'kazuko'), 'abandoned');
    // 注文ボタンが反応しなければ誰も完了できない
    assert.equal(outcomeOf(withAnnoyance(cleanSpec(), 'broken_order_button'), 'kenta'), 'abandoned');
  });

  it('偽のカウントダウンは、急かされるのが嫌いな人だけを不満にする', () => {
    const spec = withAnnoyance(cleanSpec(), 'countdown_pressure');
    // 損をしたくない人は急かされると嫌
    assert.equal(outcomeOf(spec, 'shota'), 'bothered');
    assert.equal(outcomeOf(spec, 'hina'), 'bothered');
    // さっと買いたい人は気にしない
    assert.equal(outcomeOf(spec, 'kenta'), 'achieved');
    // 帯を消せば、損をしたくない人も満足して買える
    assert.equal(outcomeOf(cleanSpec(), 'shota'), 'achieved');
  });

  it('合計金額を消す改修も取り消さず、結果はお客さんの反応で分かる', () => {
    const totalId = cleanSpec().components.find((c) => c.role === 'total')!.id;
    const result = applyPatchesToSpecWithValidation(cleanSpec(), '合計金額を消しました', [
      { op: 'remove_component', id: totalId },
    ]);
    assert.equal(result.rolledBack, false);
    assert.ok(!result.updatedSpec.components.some((c) => c.role === 'total'));
    // 金額を確かめたい人は買うのをやめ、さっと買いたい人は買える
    assert.equal(outcomeOf(result.updatedSpec, 'shota'), 'abandoned');
    assert.equal(outcomeOf(result.updatedSpec, 'kenta'), 'achieved');
  });

  it('固定の問題「売上を伸ばしたい店長の画面」は、直すほど結果が変わり、店長の反応も変わる', () => {
    const scenario = ACTIVE_THEME.scenarios[ACTIVE_THEME.defaultScenarioId];
    const build = (removed: string[] = []) => {
      // 実装と同じく、題材に定義した順に組み込む
      const spec = cleanSpec();
      for (const a of ACTIVE_THEME.annoyances) {
        if (scenario.annoyanceIds.includes(a.id) && !removed.includes(a.id)) a.apply(spec);
      }
      return spec;
    };
    const report = (spec: AppSpec) => {
      const results = ALL_REVIEWER_PERSONAS.map((p) => simulatePersonaComputerUse(spec, p));
      return {
        outcomes: results.map((r) => r.outcome),
        owner: buildOwnerReport(ACTIVE_THEME.owner, spec, results),
      };
    };

    // 既定の問題と、妨害要素を順に入れた画面は同じ
    assert.deepEqual(buildScenarioSpec(ACTIVE_THEME, ACTIVE_THEME.defaultScenarioId).spec, build());

    // 最初は誰も注文できず、店長も困っている
    const initial = report(build());
    assert.ok(initial.outcomes.every((o) => o === 'abandoned'));
    assert.equal(initial.owner.orders, 0);
    assert.match(initial.owner.comment, /注文がひとつも入らない/);

    // 注文ボタンを直すと、定期便に気づいて外したが不満の残る人と、気づかず後悔する人が現れる。
    // まだ誰も嫌な思いをせずには買えない。店長は定期便が取れて喜ぶ
    const fixed = report(build(['broken_order_button']));
    assert.ok(fixed.outcomes.includes('bothered'));
    assert.ok(fixed.outcomes.includes('regretted'));
    assert.ok(!fixed.outcomes.includes('achieved'));
    const subscription = fixed.owner.metrics.find((m) => m.label === '定期便')!;
    assert.ok(subscription.count > 0);
    assert.match(fixed.owner.comment, /定期便も\d+件取れたぞ/);

    // 定期便の欄や広告をただ消すと後悔はなくなるが、定期便やクーポンを希望していた客は設定がなくて不満（bothered）になり、店長も嘆く
    // 実際のプレイと同じく、店にあった部品を手直しで消す（最初から入れない画面は「店になかった」扱いになり不満は出ない）
    const noTrap = applyPatchesToSpecWithValidation(
      build(['broken_order_button', 'countdown_pressure', 'decoy_buy_now']),
      '定期便と広告を消しました',
      [
        { op: 'remove_component', id: 'check-subscription' },
        { op: 'remove_component', id: 'popup-coupon' },
      ],
    ).updatedSpec;
    const cleaned = report(noTrap);
    assert.ok(!cleaned.outcomes.includes('regretted'));
    assert.ok(cleaned.outcomes.includes('bothered'));
    assert.match(cleaned.owner.comment, /定期便の申し込み欄がなくなった/);
  });

  it('はっきりした改造の指示だけをすぐ改修に回し、迷う発言はお助けエージェントに判断させる', () => {
    // すぐ改修してよいもの
    for (const msg of [
      '邪魔な部品を消して、すぐ注文できるようにして！',
      '文字を大きくして',
      'この広告消して',
      '注文ボタンを動くように直す',
      '定期便のチェックを最初は外しておいて',
      // 対象の部品と目的がそろっていれば、直し方を言っていなくてもすぐ改修してよい
      '定期便を望む人だけが自分で選べるようにして',
      'この注意書きを目が遠い人でも読めるようにして',
      // 「ようにしたい」のように変える言葉がなくても、部品名と目的がそろえばAIの振り分けを待たずに改修する
      '定期便は、毎月届けてほしい人だけが自分で選べるようにしたい',
      'クーポン広告は、使いたい人には役立って、いらない人の邪魔にならないようにしたい',
      '注文する前に、スマホに不慣れな人でも内容を確かめられるようにしたい',
      // 「〜とはっきり」「〜とは別に」「〜とは違う」に含まれる「とは」を質問と誤認しない
      '背景とはっきり見分けられる濃い色にして',
      '注文ボタンとは別に、定期便の説明を書いて',
      '周りとは違う見やすい色にして',
    ]) {
      assert.equal(classifyUserMessageIntent(msg), 'apply', msg);
    }
    // 質問・感想・あいさつ・変え方が決まらない発言
    for (const msg of [
      'なんでこのボタン押しても動かないの？',
      'これ何が困る？',
      'このボタンの仕組みを教えて',
      'ダークパターンとは',
      '定期便とは何？',
      'なんで小さいの？大きくしてほしい',
      '赤いボタンが邪魔',
      '健太が困ってるのはどこ',
      'うーん',
      'ありがとう',
      '説明してほしい',
      // 目的だけで対象の部品がない発言は、AIが直す部品を選ばないよう聞き返しに回す
      '高齢の人でも迷わず買えるようにして',
      'お客さんが安心できるようにしてほしい',
      // 目的の形でも、質問・感想・「これ」だけで部品名がないものはAIに判断させる
      '定期便は望む人だけが選べるようにした方がいい？',
      '定期便は高齢の人を迷わせると思う',
      'これを目が遠い人でも読めるようにしたい',
    ]) {
      assert.equal(classifyUserMessageIntent(msg), 'triage', msg);
    }
  });

  it('デモ用の問題では、損をしたくない人が4手で、消すと困る場面と目的を伝えて立て直す場面を通って全員達成する', () => {
    const thrifty = getReviewersByCohort('thrifty');
    const outcomes = (spec: AppSpec) =>
      thrifty.map((p) => simulatePersonaComputerUse(spec, p));
    const step = (spec: AppSpec, message: string, patches: SpecPatch[]) => {
      const result = applyPatchesToSpecWithValidation(spec, message, patches);
      assert.equal(result.rolledBack, false, message);
      return result;
    };
    const demo = buildScenarioSpec(ACTIVE_THEME, 'demo').spec;

    // 1手目：注文ボタンを直す
    const order = findComponent(demo, 'btn-order');
    const fixed = step(demo, '注文ボタンを直しました', [
      {
        op: 'replace_component',
        component: { ...order, actions: [{ type: 'complete_task', message: 'ご購入ありがとうございます' }] },
      },
    ]).updatedSpec;

    // 2手目：広告を消さずに、閉じやすく、その場で500円引きになるクーポンにすると、定期便の罠が表に出る
    const popup = findComponent(fixed, 'popup-coupon');
    const couponFixed = step(fixed, '広告をその場で使えるクーポンにしました', [
      {
        op: 'replace_component',
        component: {
          ...popup,
          closePosition: 'bottom-bar',
          closeSize: 'lg',
          closeLabel: '500円引きを使う',
          actions: [{ type: 'set_state', targetKey: 'couponDiscount', value: 500 }],
        },
      },
    ]).updatedSpec;
    assert.ok(outcomes(couponFixed).some((r) => r.outcome === 'regretted'));

    // 3手目：定期便を消すと、毎月届けてほしいお客さんが不満になり、店長も嘆く
    const removed = step(couponFixed, '定期便を消しました', [
      { op: 'remove_component', id: 'check-subscription' },
    ]);
    const afterRemove = outcomes(removed.updatedSpec);
    assert.equal(afterRemove.filter((r) => r.outcome === 'achieved').length, 2);
    assert.ok(afterRemove.some((r) => r.outcome === 'bothered' && r.frictionPoint === '定期便の設定がなかった'));
    assert.match(removed.ownerReaction ?? '', /定期便/);

    // 4手目：望む人だけが選べる正直な定期便に戻すと、全員達成になり店長も喜ぶ
    const subscription = findComponent(demo, 'check-subscription');
    const restored = step(removed.updatedSpec, '定期便を選べるようにしました', [
      {
        op: 'upsert_state',
        variable: { key: 'subscription', label: '定期便', type: 'boolean', initialValue: false },
      },
      {
        op: 'insert_component',
        afterId: 'product-card',
        component: {
          ...subscription,
          label: '定期便にする 毎月お届け いつでも解約OK',
          style: { size: 'sm', variant: 'default' },
        },
      },
    ]);
    assert.deepEqual(
      outcomes(restored.updatedSpec).map((r) => r.outcome),
      ['achieved', 'achieved', 'achieved'],
    );
  });

  it('選ばなかった客層への影響を1行で示し、確認を足すと急ぐ人が困り、前回の設定を入れておくと全員が両立する', () => {
    const careful = getReviewersByCohort('careful');
    const others = ALL_REVIEWER_PERSONAS.filter((p) => !careful.some((c) => c.id === p.id));
    // 定期便を正直な選択欄にし、クーポン広告を閉じやすくした画面（定期便やクーポンを望む人も満たされる）
    const spec = withAnnoyance(withAnnoyance(cleanSpec(), 'subscription_trap'), 'coupon_popup');
    for (const v of spec.state) if (v.key === 'subscription') v.initialValue = false;
    Object.assign(findComponent(spec, 'check-subscription'), {
      label: '定期便にする 毎月お届け いつでも解約OK',
      style: { size: 'lg', variant: 'default' },
    });
    Object.assign(findComponent(spec, 'popup-coupon'), {
      closePosition: 'bottom-bar',
      closeSize: 'lg',
      closeLabel: '500円引きを使う',
      actions: [{ type: 'set_state', targetKey: 'couponDiscount', value: 500 }],
    });

    // 確認なし：ほかの客層の6人は全員買える
    const noConfirm = describeOtherCohortImpact(spec, others);
    assert.equal(noConfirm.achieved, 6);
    assert.match(noConfirm.line, /`6人` も、この画面で気持ちよく買えます/);

    // 確認を足すと、確実に買いたい人は助かるが、ほかの客層の急いでいる人が離脱する
    findComponent(spec, 'btn-order').behavior = { effect: 'confirm_dialog' };
    const withConfirm = describeOtherCohortImpact(spec, others);
    assert.ok(withConfirm.unhappy.some((u) => u.outcome === 'abandoned'));
    assert.match(withConfirm.line, /^ほかの客層では、`.+` が `.+` ため離脱しました/);
    assert.doesNotMatch(withConfirm.line, /[()（）「」『』]/);

    // 前回のお届け先と支払い方法を最初から入れておくと、両方が満たされる
    for (const v of spec.state) {
      if (v.key === 'deliveryAddress') v.initialValue = '自宅';
      if (v.key === 'paymentMethod') v.initialValue = 'クレジットカード';
    }
    assert.equal(describeOtherCohortImpact(spec, others).achieved, 6);
  });

  it('指示の型はコードの規則で決まったものを優先し、埋め込みは規則で決まらない指示にだけ使う', async () => {
    const originalFetch = globalThis.fetch;
    const originalProject = process.env.GOOGLE_CLOUD_PROJECT;
    let fetchCalls = 0;
    globalThis.fetch = (async () => {
      fetchCalls++;
      throw new Error('埋め込みを呼んではいけない');
    }) as typeof fetch;
    process.env.GOOGLE_CLOUD_PROJECT = 'test-project';
    try {
      for (const [text, expected] of [
        ['注文する前に、スマホに不慣れな人でも内容を確かめられるようにして', 'goal_directed'],
        ['定期便を消して', 'simple_delete'],
        ['定期便の初期チェックを外して解約条件を明記して', 'condition_design'],
        ['いい感じにして', 'vague_delegate'],
      ] as const) {
        const signal = await classifyPromptWithAgentPlatformEmbeddings(text, 'execute');
        assert.equal(signal.closestArchetype, expected, text);
        assert.equal(signal.analyzedVia, 'local_deterministic', text);
      }
      assert.equal((await classifyPromptWithAgentPlatformEmbeddings('店長、なぜ定期便を付けたの？', 'explore')).closestArchetype, 'inquiry');
      assert.equal(fetchCalls, 0);
    } finally {
      globalThis.fetch = originalFetch;
      if (originalProject === undefined) delete process.env.GOOGLE_CLOUD_PROJECT;
      else process.env.GOOGLE_CLOUD_PROJECT = originalProject;
    }
  });

  it('定期便やクーポンは、店に最初からなければ望む人も不満にならず、あったのに消されたときだけ不満になる', () => {
    // 最初から定期便もクーポンもない、まともな画面
    for (const p of getReviewersByCohort('thrifty')) {
      assert.equal(outcomeOf(cleanSpec(), p.id), 'achieved', p.name);
    }
    // 店にあった定期便とクーポン広告を消すと、望んでいた人は不満になる
    const withBoth = withAnnoyance(withAnnoyance(cleanSpec(), 'subscription_trap'), 'coupon_popup');
    const removed = applyPatchesToSpecWithValidation(withBoth, '定期便と広告を消しました', [
      { op: 'remove_component', id: 'check-subscription' },
      { op: 'remove_component', id: 'popup-coupon' },
    ]).updatedSpec;
    const frictions = getReviewersByCohort('thrifty').map(
      (p) => simulatePersonaComputerUse(removed, p).frictionPoint,
    );
    assert.ok(frictions.includes('定期便の設定がなかった'));
    assert.ok(frictions.includes('クーポンがなくなっていた'));

    // 罠の定期便を消すと、状態は罠を外した値に戻る。画面にない定期便に気づかず申し込む人はいない
    assert.equal(removed.state.find((v) => v.key === 'subscription')?.initialValue, false);
    for (const p of ALL_REVIEWER_PERSONAS) {
      assert.notEqual(simulatePersonaComputerUse(removed, p).finalState?.subscription, true, p.name);
    }
    const ownerReport = buildOwnerReport(
      ACTIVE_THEME.owner,
      removed,
      ALL_REVIEWER_PERSONAS.map((p) => simulatePersonaComputerUse(removed, p)),
    );
    const subscriptionMetric = ownerReport.metrics.find((m) => m.label === '定期便');
    assert.equal(subscriptionMetric?.missing, true);
    assert.equal(subscriptionMetric?.count, 0);

    // 同じ手直しで状態の値をはっきり指定したときは、その指定を優先する
    const explicit = applyPatchesToSpecWithValidation(withBoth, '定期便を消して状態を指定しました', [
      { op: 'remove_component', id: 'check-subscription' },
      {
        op: 'upsert_state',
        variable: { key: 'subscription', label: '定期便', type: 'boolean', initialValue: true },
      },
    ]).updatedSpec;
    assert.equal(explicit.state.find((v) => v.key === 'subscription')?.initialValue, true);
  });

  it('店長には、店にもともとない欄は伝えず、あったのに消された欄だけを「ない」と伝える', () => {
    const spec = buildScenarioSpec(ACTIVE_THEME, ACTIVE_THEME.defaultScenarioId).spec;
    // 本番の問題に会員登録の欄はもともとない
    assert.ok(!describeOwnerContext(spec, undefined).some((f) => f.includes('会員登録')));
    assert.ok(describeOwnerContext(spec, undefined).some((f) => f.includes('定期便に最初からチェックが入っている')));
    // 定期便の欄を消すと、なくなったことが伝わる
    const removed = applyPatchesToSpecWithValidation(spec, '定期便を消しました', [
      { op: 'remove_component', id: 'check-subscription' },
    ]).updatedSpec;
    assert.ok(describeOwnerContext(removed, undefined).some((f) => f.includes('定期便の欄がない')));
  });

  it('作業ログは型をそろえた値で言葉にし、変わっていない値や英字だけの内部の値は出さない', () => {
    const spec = withAnnoyance(cleanSpec(), 'subscription_trap');
    const stepTexts = (patches: SpecPatch[], base = spec) =>
      (applyPatchesToSpecWithValidation(base, 'テスト', patches).workSteps ?? []).map((s) => s.text);
    const subscriptionFalse = {
      op: 'upsert_state',
      // Geminiは真偽値を文字列で返すことがある
      variable: { key: 'subscription', label: '定期便', type: 'boolean', initialValue: 'false' },
    } as unknown as SpecPatch;

    // チェックあり → チェックなし は、文字列の "false" でも言葉で出す
    const changed = stepTexts([subscriptionFalse]);
    assert.ok(changed.some((t) => t.includes('`チェックあり` から `チェックなし`')), changed.join('\n'));
    assert.ok(!changed.some((t) => t.includes('false')));

    // すでにチェックなしなら、変更として出さない
    const unchecked = applyPatchesToSpecWithValidation(spec, 'テスト', [subscriptionFalse]).updatedSpec;
    assert.ok(!stepTexts([subscriptionFalse], unchecked).some((t) => t.includes('最初の状態')));

    // 英字だけの内部の値は出さない
    const added = stepTexts([
      { op: 'upsert_state', variable: { key: 'step', label: '進捗ステップ', type: 'string', initialValue: 'input' } },
    ]);
    assert.ok(added.some((t) => t.includes('を覚える欄を追加')));
    assert.ok(!added.some((t) => t.includes('input')));
  });

  it('吹き出しは句点で区切るが、「！？」のように続く句点は同じ吹き出しに入れる', () => {
    assert.deepEqual(splitIntoChatChunks('おいおい、消しちゃったのかい！？売上が不安だよ'), [
      'おいおい、消しちゃったのかい！？',
      '売上が不安だよ',
    ]);
    assert.deepEqual(splitIntoChatChunks('ありがとう。助かったよ！'), ['ありがとう。', '助かったよ！']);
    assert.deepEqual(splitIntoChatChunks('★今だけ！今すぐ購入★を消したのか'), ['★今だけ！今すぐ購入★を消したのか']);
  });

  it('手早く注文できるようにしても、広告に手を付けなければ広告に当たり、見ていないクーポンは見た扱いにしない', () => {
    // 前回の支払い方法を入れて確認画面を足し、定期便を正直にした。広告には手を付けていない
    const spec = buildScenarioSpec(ACTIVE_THEME, 'demo').spec;
    findComponent(spec, 'btn-order').actions = [{ type: 'complete_task', message: 'ご購入ありがとうございます' }];
    findComponent(spec, 'btn-order').behavior = { effect: 'confirm_dialog' };
    for (const v of spec.state) {
      if (v.key === 'subscription') v.initialValue = false;
      if (v.key === 'paymentMethod') v.initialValue = 'クレジットカード';
    }
    Object.assign(findComponent(spec, 'check-subscription'), {
      label: '定期便にする 毎月お届け いつでも解約OK',
      style: { size: 'lg', variant: 'default' },
    });
    // 広告の閉じるボタンが見えない人は、手早く注文できても広告ですり抜けられない
    assert.equal(outcomeOf(spec, 'ren'), 'abandoned');
    assert.equal(outcomeOf(spec, 'kazuko'), 'abandoned');
    // クーポンを望む人は、広告を閉じさせられて不満になる
    assert.equal(outcomeOf(spec, 'shota'), 'bothered');
  });

  it('クリアのレポートでは、選ばなかった客層のお客さん一人ひとりが今の画面をどう思いそうかを出す', () => {
    // デモの4手を終えた画面：損をしたくない人は全員達成し、スマホに不慣れな人だけが確認できずに離脱しそう
    const spec = buildScenarioSpec(ACTIVE_THEME, 'demo').spec;
    findComponent(spec, 'btn-order').actions = [{ type: 'complete_task', message: 'ご購入ありがとうございます' }];
    Object.assign(findComponent(spec, 'popup-coupon'), { closePosition: 'bottom-bar', closeSize: 'lg', closeLabel: 'あとで見る' });
    for (const v of spec.state) if (v.key === 'subscription') v.initialValue = false;
    Object.assign(findComponent(spec, 'check-subscription'), {
      label: '定期便にする 毎月お届け いつでも解約OK',
      style: { size: 'sm', variant: 'default' },
    });
    const others = ALL_REVIEWER_PERSONAS.filter((p) => p.cohortId !== 'thrifty');
    const report = describeOtherCohortReport(spec, others, undefined, ACTIVE_THEME.cohorts);
    assert.equal(report.total, 6);
    assert.equal(report.lines.length, 6);
    assert.equal(report.achieved, 5);
    assert.ok(
      report.lines.some(
        (l) => l === '`スマホに不慣れなお客さん` は `注文を確定する前に内容を確かめる画面がなかった` ため離脱しそう',
      ),
    );
    // 客層ごとにまとまり、客層の名前が付く。次にどの客層を選べばその人のために直せるかが分かる
    assert.deepEqual(
      report.groups.map((g) => [g.label, g.total, g.achieved]),
      [
        ['確実に買いたい人', 3, 2],
        ['さっと買いたい人', 3, 3],
      ],
    );
    assert.ok(report.groups[0].lines.some((l) => l.startsWith('`スマホに不慣れなお客さん`')));
    assert.ok(report.lines.some((l) => l.endsWith('は気持ちよく買えそう')));
    for (const line of report.lines) assert.doesNotMatch(line, /[()（）「」『』]/);
  });

  it('振り返りは挑戦ごとの番号の記録だけで作り、やり直す前の挑戦やほかの人の記録を混ぜない', async () => {
    clearPromptTelemetryEvents();
    const base = { cohortId: 'thrifty', sessionMode: 'execute' as const, recipient: 'assistant', hasPenAnnotation: false };
    await recordPromptTelemetryEvent({ ...base, runId: 'run-1', turnIndex: 5, promptText: '定期便を消して', changedComponentIds: ['check-subscription'], removedNeededFeature: true });
    await recordPromptTelemetryEvent({ ...base, runId: 'run-other', turnIndex: 7, promptText: '広告を消して', changedComponentIds: ['popup-coupon'] });
    const second = await recordPromptTelemetryEvent({ ...base, runId: 'run-2', turnIndex: 0, promptText: '定期便の初期チェックを外して', changedComponentIds: ['check-subscription'] });
    // 別の挑戦で同じ部品を触っていても、手戻りとは数えない
    assert.equal(second.isReworkOnSameComponent, false);
    const summary = await finalizeTelemetryOnEvaluate('thrifty', 'run-2');
    assert.equal(summary.executeTurns, 1);
    assert.equal(summary.reworkTurns, 0);
    assert.ok(!summary.systemLogLines.some((l) => l.includes('{{turn:5}}') || l.includes('{{turn:7}}')));

    // 別のインスタンスが Firestore に残した記録（turn 0・1）と、今のインスタンスの記録（turn 2）を突合して振り返れる
    const cloudDoc0 = {
      name: 'projects/test-proj/databases/(default)/documents/prompt_telemetry/pt-cloud-0',
      fields: toFirestoreTelemetryFields({
        ...second,
        id: 'pt-cloud-0',
        runId: 'run-multi',
        turnIndex: 0,
        sessionMode: 'explore',
        recipient: 'owner',
        changedComponentIds: [],
        removedNeededFeature: false,
        nlSignal: extractDeterministicNlSignal('店長、定期便ってどうして付けてるの？'),
        embeddingSignal: classifyDeterministicArchetype('店長、定期便ってどうして付けてるの？', 'explore'),
        createdAt: '2026-11-08T05:00:00.000Z',
      }),
    };
    const cloudDoc1 = {
      name: 'projects/test-proj/databases/(default)/documents/prompt_telemetry/pt-cloud-1',
      fields: toFirestoreTelemetryFields({
        ...second,
        id: 'pt-cloud-1',
        runId: 'run-multi',
        turnIndex: 1,
        sessionMode: 'execute',
        changedComponentIds: ['check-subscription'],
        isReworkOnSameComponent: false,
        removedNeededFeature: true,
        nlSignal: extractDeterministicNlSignal('定期便を消して'),
        embeddingSignal: classifyDeterministicArchetype('定期便を消して', 'execute'),
        createdAt: '2026-11-08T05:01:00.000Z',
      }),
    };
    const parsed0 = parseFirestoreTelemetryDoc(cloudDoc0, Date.parse('2026-11-08T06:00:00.000Z'));
    const parsed1 = parseFirestoreTelemetryDoc(cloudDoc1, Date.parse('2026-11-08T06:00:00.000Z'));
    assert.ok(parsed0 && parsed1);
    // 期限（expireAt: 24時間後）を過ぎた記録は、TTL の削除前でも除外される
    assert.equal(parseFirestoreTelemetryDoc(cloudDoc0, Date.parse('2026-11-10T06:00:00.000Z')), null);

    const localTurn2 = {
      ...second,
      id: 'pt-local-2',
      runId: 'run-multi',
      turnIndex: 2,
      sessionMode: 'execute' as const,
      changedComponentIds: ['check-subscription'],
      isReworkOnSameComponent: true,
      removedNeededFeature: false,
      nlSignal: extractDeterministicNlSignal('定期便を望む人だけが自分で選べるようにして'),
      embeddingSignal: classifyDeterministicArchetype('定期便を望む人だけが自分で選べるようにして', 'execute'),
      createdAt: '2026-11-08T05:02:00.000Z',
    };
    const merged = mergeTelemetryEvents([localTurn2], [parsed1, parsed0]);
    assert.deepEqual(merged.map((e) => e.turnIndex), [0, 1, 2]);
    const mergedSummary = summarizePromptTelemetry(merged, 'thrifty');
    assert.equal(mergedSummary.exploreTurns, 1);
    assert.equal(mergedSummary.executeTurns, 2);
    assert.equal(mergedSummary.reworkTurns, 1);
    assert.deepEqual(mergedSummary.turningPoint, { beforeTurn: 1, afterTurn: 2 });

    clearPromptTelemetryEvents();
  });

  it('公開する集計は3件未満の全体・客層区分を伏せ、題材の語彙だけを3回以上のときに出す', () => {
    // 全体の挑戦が3件未満なら、すべて伏せる
    const tooFew = formatPublicAggregateStats(
      {
        runTotals: { total_runs: 2, cleared_runs: 2, reset_runs: 0, abandoned_runs: 0 },
        cohortStats: [{ cohort_id: 'thrifty', run_count: 2, cleared_count: 2 }],
      },
      TARGET_COHORTS,
    );
    assert.equal(tooFew.available, true);
    assert.equal(tooFew.suppressed, true);
    assert.equal(tooFew.totalRuns, undefined);
    assert.ok(tooFew.cohorts.every((c) => c.suppressed && c.runCount === undefined));

    // 全体が3件以上でも、3件未満の客層（careful: 2件、quick: 0件）は伏せ、3件以上の客層（thrifty: 4件）だけ出す
    const enough = formatPublicAggregateStats(
      {
        runTotals: {
          total_runs: 6,
          cleared_runs: 4,
          reset_runs: 1,
          abandoned_runs: 1,
          avg_execute_turns_on_clear: 3.75,
        },
        cohortStats: [
          { cohort_id: 'thrifty', run_count: 6, execute_run_count: 4, cleared_count: 3, avg_execute_turns_on_clear: 3.666 },
          { cohort_id: 'careful', run_count: 4, execute_run_count: 2, cleared_count: 1, avg_execute_turns_on_clear: 4.0 },
        ],
        promptTotals: {
          total_prompts: 20,
          execute_prompts: 15,
          rework_prompts: 3,
          removed_needed_prompts: 2,
          execute_run_count: 5,
        },
        archetypeStats: [
          { archetype: 'goal_directed', cnt: 6 },
          { archetype: 'condition_design', cnt: 5 },
          { archetype: 'inquiry', cnt: 5 },
          { archetype: 'simple_delete', cnt: 4 },
        ],
        entityStats: [
          { term: '定期便', cnt: 8 },
          { term: '山田太郎', cnt: 10 }, // 題材の語彙にない言葉は出さない
          { term: 'クーポン', cnt: 2 }, // 3回未満は出さない
        ],
      },
      TARGET_COHORTS,
    );
    assert.equal(enough.suppressed, false);
    assert.equal(enough.totalRuns, 6);
    assert.equal(enough.clearRatePercent, 67);
    assert.equal(enough.avgExecuteTurnsOnClear, 3.8);
    assert.equal(enough.reworkRatePercent, 20);
    assert.equal(enough.removedNeededFeatureRatePercent, 13);
    assert.deepEqual(enough.topEntities, [{ term: '定期便', count: 8 }]);

    const thriftyCohort = enough.cohorts.find((c) => c.cohortId === 'thrifty');
    const carefulCohort = enough.cohorts.find((c) => c.cohortId === 'careful');
    assert.equal(thriftyCohort?.suppressed, false);
    assert.equal(thriftyCohort?.runCount, 6);
    assert.equal(thriftyCohort?.executeRunCount, 4);
    assert.equal(thriftyCohort?.clearRatePercent, 75);
    assert.equal(thriftyCohort?.avgExecuteTurnsOnClear, 3.7);
    assert.equal(carefulCohort?.suppressed, true);
    assert.equal(carefulCohort?.runCount, undefined);
  });

  it('チェック欄にAIが付けた閉じるボタンの設定は作業ログに出さず、店長には直近の申し込み件数を伝える', () => {
    const spec = withAnnoyance(cleanSpec(), 'subscription_trap');
    const sub = findComponent(spec, 'check-subscription');
    const steps = (applyPatchesToSpecWithValidation(spec, 'テスト', [
      { op: 'replace_component', component: { ...sub, closeSize: 'md', closeLabel: '閉じる', style: { ...sub.style, size: 'sm' } } },
    ]).workSteps ?? []).map((s) => s.text);
    assert.ok(steps.some((t) => t.includes('文字サイズ')), steps.join('\n'));
    assert.ok(!steps.some((t) => t.includes('閉じるボタン')), steps.join('\n'));

    const facts = describeOwnerContext(spec, {
      screenLabel: '改造1',
      isCurrentScreen: true,
      reviews: [{ name: '画面が割れているお客さん', outcome: 'achieved', actionTrace: [] }],
      ownerMetrics: [{ label: '定期便', count: 2, missing: false }],
    });
    assert.ok(facts.some((f) => f.includes('定期便の申し込みが2件あった')), facts.join('\n'));
  });

  it('3つの顧客層にそれぞれ3人のペルソナがいる', () => {
    assert.equal(ALL_REVIEWER_PERSONAS.length, 9);
    for (const p of ALL_REVIEWER_PERSONAS) {
      assert.ok(p.greeting.trim().length > 0, `${p.name} の来店時の一言が空です`);
    }
    for (const cohort of ACTIVE_THEME.cohorts) {
      const reviewers = getReviewersByCohort(cohort.id);
      assert.equal(reviewers.length, 3);
      for (const r of reviewers) {
        assert.match(r.name, /お客さん$/);
        assert.equal(r.icon, '客');
      }
    }
  });

  it('@メンションのときだけ宛先を切り替え、文頭や文中に名前があるだけでは切り替えない', () => {
    // 明示的な @ メンションのみ切り替える
    assert.equal(detectExplicitRecipientMention('@店長 定期便はどうする？'), 'owner');
    assert.equal(detectExplicitRecipientMention('＠店長、広告を消してもいい？'), 'owner');
    assert.equal(detectExplicitRecipientMention('@お客さん 何が嫌だった？'), 'customer');
    assert.equal(detectExplicitRecipientMention('@目が遠いお客さん 文字は読めた？'), 'customer');
    assert.equal(detectExplicitRecipientMention('@お助けAI これ何の部品？'), 'assistant');
    assert.equal(detectExplicitRecipientMention('＠お助けエージェント 直して'), 'assistant');

    // @ なしで「店長」等を入力した途中や文中言及では切り替えない
    for (const msg of [
      '店長',
      '店長、広告を消してもいい？',
      'ねえ店長！売上って大事？',
      '店長が付けた広告を消して',
      '店長の言ってる定期便って何？',
      'お客さん',
      'お客さん、どこで困ったの？',
      'お客さんはどこで困ったの？',
      '目が遠いお客さんが買えなかった原因を教えて',
      'お客さんの言ってるボタンを直して',
      'お助けAI、注文ボタンを直して',
      'お助けAIに頼む前にどう思う？',
    ]) {
      assert.equal(detectExplicitRecipientMention(msg), null, msg);
    }
  });

  it('実装AIが文字列で返した値を、状態の型に合わせて直す', () => {
    const previous = cleanSpec();
    const raw = structuredClone(previous) as unknown as AppSpec;
    const order = raw.components.find((c) => c.id === 'btn-order')!;
    order.visibleWhen = { stateKey: 'quantity', operator: 'gt', value: '0' };
    raw.components.push({
      id: 'check-x',
      type: 'checkbox',
      role: 'trap',
      label: 'x',
      stateKey: 'quantity',
      trapWhen: { stateKey: 'quantity', operator: 'gt', value: '5' },
      trapSafeValue: '1',
      style: { size: 'xs', variant: 'muted' },
      actions: [],
    });
    const normalized = normalizeUpdatedSpec(raw, previous);
    assert.equal(normalized.components.find((c) => c.id === 'btn-order')!.visibleWhen!.value, 0);
    const trap = normalized.components.find((c) => c.id === 'check-x')!;
    assert.equal(trap.trapWhen!.value, 5);
    assert.equal(trap.trapSafeValue, 1);
    assert.deepEqual(normalized.derived, previous.derived);
  });

  it('判断記録は決まった値だけを残し、自由に書かれた理由は保存しない', async () => {
    // 理由の文には学習者を特定できる言葉が混ざりうるので、渡されても残さない
    const decisionRes = await recordHumanDecisionWithFunctionCalling({
      stage: 'review_disposition',
      targetId: 'kazuko',
      chosenLabel: '和子:直す',
      reason: '目が遠い人でも読めるように文字を大きくしたいから',
    });
    assert.equal(decisionRes.decision.stage, 'review_disposition');
    assert.equal(decisionRes.decision.reason, '');

    const noReasonRes = await recordHumanDecisionWithFunctionCalling({
      stage: 'select_cohort',
      targetId: 'careful',
      chosenLabel: '確実に買いたい人',
      reason: '',
    });
    assert.equal(noReasonRes.decision.reason, '');

    // 手直しの記録には自由入力の文章を残さず、決まったラベルに置き換える
    const applyDecisionRes = await recordHumanDecisionWithFunctionCalling({
      stage: 'apply_change',
      targetId: 'btn-order',
      chosenLabel: '山田太郎の住所に合わせて注文ボタンを直して',
      reason: '連絡先は test@example.com です',
    });
    assert.equal(applyDecisionRes.decision.chosenLabel, '手直し指示');
    assert.equal(applyDecisionRes.decision.reason, '');
  });

  it('AIの接続設定がない場合は代替処理をせずエラーにする', async () => {
    assert.equal(getAiConnectionStatus().configured, false);
    const spec = cleanSpec();

    await assert.rejects(
      chatWithStudio({ currentSpec: spec, targetComponentIds: [], userMessage: '文字を大きくして' }),
      AiNotConfiguredError,
    );
    await assert.rejects(
      applyOptionToSpec({
        currentSpec: spec,
        targetComponentIds: [],
        userInstruction: '文字を大きくして',
      }),
      AiNotConfiguredError,
    );
    await assert.rejects(evaluateAppWithPersonas({ currentSpec: spec }), AiNotConfiguredError);
  });

  it('差分パッチ適用で対象外の部品が一切崩れず、不正な操作は安全に無視される', () => {
    const initial = buildScenarioSpec(ACTIVE_THEME, ACTIVE_THEME.defaultScenarioId).spec;

    // 1. 注文ボタンを直し、おとりボタンを消し、存在しないIDへの操作や重複見出しを混ぜる
    const { spec: patched, warnings, touchedComponentIds } = applySpecPatches(initial, [
      {
        op: 'replace_component',
        component: {
          id: 'btn-order',
          type: 'button',
          role: 'primary_action',
          label: '注文を確定する',
          // sticky を省略しても元の btn-order から引き継がれること
          style: { size: 'lg', variant: 'primary', fullWidth: true, align: 'center' },
          actions: [{ type: 'complete_task', message: 'ご購入ありがとうございます' }],
        },
      },
      { op: 'remove_component', id: 'btn-buy-now' },
      { op: 'remove_component', id: 'non-existent-id' },
      {
        op: 'insert_component',
        afterId: 'product-card',
        component: {
          id: 'heading-address-dup',
          type: 'heading',
          role: 'info',
          label: 'お届け先',
          style: { size: 'sm', variant: 'default' },
          actions: [],
        },
      },
    ]);

    assert.ok(touchedComponentIds.has('btn-order'));
    assert.ok(touchedComponentIds.has('btn-buy-now'));
    assert.ok(warnings.some((w) => w.includes('non-existent-id')));
    assert.ok(warnings.some((w) => w.includes('同じ見出しの重複追加')));

    // 触っていない部品（進行表示・商品カード・選択カード・内訳・合計）が完全にそのままであること
    for (const id of ['progress', 'product-card', 'choice-address', 'choice-payment', 'summary', 'total-price']) {
      assert.deepEqual(findComponent(patched, id), findComponent(initial, id), `${id} が変化していないこと`);
    }
    // 注文ボタンの sticky も維持されていること
    assert.equal(findComponent(patched, 'btn-order').style.sticky, true);
  });

  it('改修検証と店長の変更直後リアクションが一貫して動く', () => {
    const initial = buildScenarioSpec(ACTIVE_THEME, ACTIVE_THEME.defaultScenarioId).spec;

    // 注文ボタンが壊れたままでも、先に広告を削除する改修は通る
    const step1 = applyPatchesToSpecWithValidation(initial, '広告を消しました', [
      { op: 'remove_component', id: 'popup-coupon' },
    ]);
    assert.equal(step1.rolledBack, false);
    assert.ok(step1.ownerReaction?.includes('広告'));

    // 注文ボタンを動くように直すと、店長が喜ぶ
    const step2 = applyPatchesToSpecWithValidation(step1.updatedSpec, '注文ボタンを直しました', [
      {
        op: 'replace_component',
        component: {
          ...findComponent(step1.updatedSpec, 'btn-order'),
          actions: [{ type: 'complete_task', message: 'ご購入ありがとうございます' }],
        },
      },
    ]);
    assert.equal(step2.rolledBack, false);
    assert.equal(step2.validation?.hasTaskCompletionEvent, true);
    assert.ok(step2.ownerReaction?.includes('注文ボタン'));

    // 定期便のチェック欄を消すと、店長が売上の心配をする
    const step3 = applyPatchesToSpecWithValidation(step2.updatedSpec, '定期便の欄を消しました', [
      { op: 'remove_component', id: 'check-subscription' },
    ]);
    assert.equal(step3.rolledBack, false);
    assert.ok(step3.ownerReaction?.includes('定期便'));
    assert.equal(
      buildOwnerChangeReaction(ACTIVE_THEME.owner, step2.updatedSpec, step3.updatedSpec),
      step3.ownerReaction,
    );
  });

  it('改造の作業内容を、学習者が読める短い言葉の行にする', () => {
    const initial = buildScenarioSpec(ACTIVE_THEME, ACTIVE_THEME.defaultScenarioId).spec;
    const order = initial.components.find((c) => c.id === 'btn-order')!;
    const patches: SpecPatch[] = [
      {
        op: 'replace_component',
        component: { ...order, actions: [{ type: 'complete_task', message: 'ご購入ありがとうございます' }] },
      },
      { op: 'remove_component', id: 'popup-coupon' },
      { op: 'upsert_state', variable: { key: 'subscription', label: '定期便', type: 'boolean', initialValue: false } },
    ];
    const result = applyPatchesToSpecWithValidation(initial, '直しました', patches);
    const texts = (result.workSteps ?? []).map((s) => s.text);
    assert.equal(texts.length, 3);
    assert.equal(texts[0], '`注文を確定する` の押したときの動きを `反応なし` から `注文完了` に変更');
    assert.match(texts[1], /^`.*` を削除$/);
    assert.equal(texts[2], '`定期便` の最初の状態を `チェックあり` から `チェックなし` に変更');
    // どの作業行にも括弧（「」や（））が含まれないこと
    for (const t of texts) {
      assert.ok(!/[「」『』（）()]/.test(t), `括弧が含まれています: ${t}`);
    }
    // 変えた部品はスマホで光らせられるよう、部品の名前が付いている
    assert.equal(result.workSteps?.[0].componentId, 'btn-order');
    assert.equal(result.workSteps?.[2].componentId, 'check-subscription');
    // 偽のカウントダウンも、画面に出る文字で書く
    const removedBanner = applyPatchesToSpecWithValidation(initial, '消しました', [
      { op: 'remove_component', id: 'banner-countdown' },
    ]);
    assert.match(removedBanner.workSteps?.[0].text ?? '', /2:59/);
  });

  it('店長が気にする変化は、何を消したかを区別した事実として取り出せる', () => {
    const initial = buildScenarioSpec(ACTIVE_THEME, ACTIVE_THEME.defaultScenarioId).spec;
    const removed = applyPatchesToSpecWithValidation(initial, '消しました', [
      { op: 'remove_component', id: 'btn-buy-now' },
      { op: 'remove_component', id: 'banner-countdown' },
    ]);
    const facts = removed.ownerFacts ?? [];
    assert.equal(facts.length, 2);
    assert.ok(facts.some((f) => f.includes('ボタン') && f.includes('今すぐ購入')));
    assert.ok(facts.some((f) => f.includes('煽り') && f.includes('2:59')));
    // AIを使えないときの決まったせりふは、同じものを重ねない
    assert.equal(removed.ownerReaction?.split('\n').length, 1);
  });

  it('客層選択はシステムメッセージで出され、別ページ遷移時にもシステム通知が生成される', () => {
    const cohortLine = ACTIVE_THEME.introScript.find((l) => l.showCohortSelector);
    assert.ok(cohortLine);
    assert.equal(cohortLine.speaker, 'system');

    const spec = cleanSpec();
    const state = createInitialState(spec);
    const topRes = applyActions(spec, state, [
      { type: 'open_page', message: 'ショップのトップページ' },
    ]);
    assert.equal(topRes.openedPage, 'ショップのトップページ');
    assert.deepEqual(topRes.messages, [
      { text: 'ショップのトップページに移動してしまいました', type: 'warning' },
    ]);

    const signupRes = applyActions(spec, state, [{ type: 'open_page', message: '会員登録' }]);
    assert.equal(signupRes.openedPage, '会員登録');
    assert.deepEqual(signupRes.messages, [
      { text: '会員登録のページに移動してしまいました', type: 'warning' },
    ]);

    // 導入がシステムメッセージのシーン提示から始まること
    assert.equal(ACTIVE_THEME.introScript[0]?.speaker, 'system');

    // 全員達成時の店長の締めセリフが、店の指標達成有無に応じて変化すること
    const withSubClosing = buildOwnerClosingLines(ACTIVE_THEME.owner, {
      orders: 3,
      customers: 3,
      metrics: [{ label: '定期便', count: 1, missing: false }],
      comment: '',
    });
    assert.equal(withSubClosing.length, 2);
    assert.ok(withSubClosing[1].includes('定期便も1件'));

    const withoutSubClosing = buildOwnerClosingLines(ACTIVE_THEME.owner, {
      orders: 3,
      customers: 3,
      metrics: [{ label: '定期便', count: 0, missing: true }],
      comment: '',
    });
    assert.equal(withoutSubClosing.length, 2);
    assert.ok(withoutSubClosing[1].includes('まずはちゃんと買ってもらえる画面が一番'));
    assert.ok(withSubClosing[0].includes('3人とも'));
  });

  it('代引き手数料が合計金額に反映され、定義順が逆でも収束計算され、作業ログも日本語になる', () => {
    const spec = cleanSpec();
    const initial = createInitialState(spec);
    assert.equal(initial.codFee, 0);
    assert.equal(initial.total, 2580);

    // 代金引換を選ぶと +330 円が合計に入る
    const codState = withDerivedValues(spec, { ...initial, paymentMethod: '代金引換' });
    assert.equal(codState.codFee, 330);
    assert.equal(codState.total, 2910);

    // AIが後から計算値を足して、total の後ろに extraFee が並んだ場合でも total に反映される
    const patched = applyPatchesToSpecWithValidation(spec, '手数料を反映しました', [
      { op: 'upsert_derived', key: 'total', expression: 'subtotal + shipping + codFee' },
      { op: 'upsert_derived', key: 'codFee', expression: "paymentMethod == '代金引換' ? 330 : 0" },
    ]);
    assert.equal(patched.rolledBack, false);
    const reorderedSpec: AppSpec = {
      ...patched.updatedSpec,
      derived: [
        { key: 'subtotal', expression: 'unitPrice * quantity' },
        { key: 'shipping', expression: 'subtotal >= 3960 ? 0 : shippingFee' },
        { key: 'total', expression: 'subtotal + shipping + extraFee' },
        { key: 'extraFee', expression: "paymentMethod == '代金引換' ? 330 : 0" },
      ],
    };
    const reorderedState = withDerivedValues(reorderedSpec, {
      ...createInitialState(reorderedSpec),
      paymentMethod: '代金引換',
    });
    assert.equal(reorderedState.extraFee, 330);
    assert.equal(reorderedState.total, 2910);

    // 作業ログに codFee などの英語キーが出ないこと
    const stepTexts = (patched.workSteps ?? []).map((s) => s.text);
    assert.ok(stepTexts.some((t) => t.includes('代引き手数料')));
    assert.ok(stepTexts.every((t) => !t.includes('codFee')));
  });

  it('定期便を正直で読みやすい選択欄に直すと、注意書きを読む客が自ら申し込み店長も喜ぶ', () => {
    const base = cleanSpec();
    // 最初は極小文字で最初からチェック入りの定期便罠がある画面
    const trapSpec: AppSpec = {
      ...base,
      state: [
        ...base.state,
        { key: 'subscription', label: '定期便', type: 'boolean', initialValue: true },
      ],
      components: [
        ...base.components,
        {
          id: 'check-subscription',
          type: 'checkbox',
          role: 'trap',
          label: '定期便でのお届けに同意する（毎月自動発送・解約は電話のみ）',
          stateKey: 'subscription',
          style: { variant: 'muted', size: 'xs' },
          actions: [],
        },
      ],
    };

    // 正直な定期便（最初からOFF・読める文字サイズ・解約縛りなし）に改修する
    const reformed = applyPatchesToSpecWithValidation(trapSpec, '定期便を正直な選択欄に直しました', [
      {
        op: 'upsert_state',
        variable: { key: 'subscription', label: '定期便', type: 'boolean', initialValue: false },
      },
      {
        op: 'replace_component',
        component: {
          id: 'check-subscription',
          type: 'checkbox',
          role: 'trap',
          label: '定期便でお届け（毎回10%引き・いつでも解約OK）',
          stateKey: 'subscription',
          style: { variant: 'default', size: 'sm' },
          actions: [],
        },
      },
    ]);

    assert.equal(reformed.rolledBack, false);
    // 店長が前向きな反応を返すこと
    assert.ok(reformed.ownerReaction?.includes('消さずに分かりやすく残してくれた'));
    assert.ok((reformed.ownerFacts ?? []).some((f) => f.includes('正直で分かりやすい形に整えた')));

    // 後悔したくない客層（no_regret）で試すと、注意書きを読む客（美穂）が納得して定期便を申し込み、3人とも達成かつ店長の定期便件数も1件になる
    const noRegretPersonas = getReviewersByCohort('no_regret');
    const results = noRegretPersonas.map((p) => simulatePersonaComputerUse(reformed.updatedSpec, p));
    assert.equal(results.filter((r) => r.outcome === 'achieved').length, 3);
    const ownerReport = buildOwnerReport(ACTIVE_THEME.owner, reformed.updatedSpec, results);
    const subMetric = ownerReport.metrics.find((m) => m.label === '定期便');
    assert.ok((subMetric?.count ?? 0) >= 1);

    // 店長の関心事（定期便・広告・煽り・今すぐ購入）と無関係な改修では、店長の事実リストが空になり毎回同じ小言を繰り返さない
    const unrelated = applyPatchesToSpecWithValidation(reformed.updatedSpec, '合計欄を大きくしました', [
      {
        op: 'replace_component',
        component: {
          ...reformed.updatedSpec.components.find((c) => c.id === 'card-total')!,
          style: { variant: 'primary', size: 'lg' },
        },
      },
    ]);
    assert.deepEqual(unrelated.ownerFacts, []);
    assert.equal(unrelated.ownerReaction, '');

    // プロトタイプ汚染キー（__proto__, constructor, prototype）や不正CSS値が安全に拒否・無害化されること
    const state = createInitialState(reformed.updatedSpec);
    assert.equal(evaluateExpression('constructor', state), 0);
    assert.equal(evaluateExpression('__proto__', state), 0);
    assert.equal(renderTemplate('{constructor}{__proto__}', state), '');
    const unsafePatched = applySpecPatches(reformed.updatedSpec, [
      {
        op: 'upsert_state',
        variable: { key: '__proto__', label: '危険', type: 'string', initialValue: 'polluted' },
      },
      { op: 'upsert_derived', key: 'constructor', expression: '1 + 1' },
    ]);
    assert.equal(unsafePatched.spec.state.some((s) => s.key === '__proto__'), false);
    assert.equal(unsafePatched.spec.derived?.some((d) => d.key === 'constructor'), false);

    // カスタムHTML・カスタムCSS・テーマ色のサニタイズ検証
    assert.equal(sanitizeCssColor('#ff0000', '#ffffff'), '#ff0000');
    assert.equal(sanitizeCssColor('red; background: url(https://evil.example)', '#ffffff'), '#ffffff');
    const sanitizedCss = sanitizeCustomCss(
      'padding: 8px; background: url("https://evil.example/x.png"); @import "https://evil.example"; width: expression(alert(1)); background-image: u/**/rl("https://evil.example/y.png"); list-style: u\\rl("https://evil.example/z.png");',
    );
    assert.ok(!sanitizedCss.includes('url('));
    assert.ok(!sanitizedCss.includes('evil.example'));
    assert.ok(!sanitizedCss.includes('@import'));
    assert.ok(!sanitizedCss.includes('expression('));
    const sanitizedHtml = sanitizeCustomHtml(
      '<div>安全<!-- 隠しコメント --><script>alert(1)</script><style>body{display:none}</style><template><img src="x" onerror="alert(3)"></template><a href="javascript:alert(1)" ping="https://evil.example" onclick="alert(2)">リンク</a></div>',
    );
    assert.ok(!sanitizedHtml.includes('<script'));
    assert.ok(!sanitizedHtml.includes('<style'));
    assert.ok(!sanitizedHtml.includes('<template'));
    assert.ok(!sanitizedHtml.includes('<!--'));
    assert.ok(!sanitizedHtml.includes('javascript:'));
    assert.ok(!sanitizedHtml.includes('ping='));
    assert.ok(!sanitizedHtml.includes('onclick='));

    // API 入力の境界検証（AppSpec 上限・部品座標・Base64・対象ID）
    assert.equal(isValidAppSpec(reformed.updatedSpec), true);
    assert.equal(
      isValidAppSpec({
        ...reformed.updatedSpec,
        components: Array.from({ length: 61 }, (_, i) => ({
          id: `c-${i}`,
          type: 'text',
        })),
      }),
      false,
    );
    assert.equal(
      isValidAppSpec({
        ...reformed.updatedSpec,
        state: [...reformed.updatedSpec.state, { key: '__proto__', label: '危険', type: 'boolean', initialValue: false }],
      }),
      false,
    );
    assert.deepEqual(
      parseCustomHtmlActions(
        reformed.updatedSpec,
        '<button data-toggle-state="__proto__" data-increment-state="constructor" data-set-state="prototype:1">x</button>',
      ),
      [],
    );
    assert.deepEqual(sanitizeComponentIds(['btn-order', '', 123, 'a'.repeat(81)]), ['btn-order']);
    assert.deepEqual(
      sanitizeComponentBounds([
        { id: 'btn-order', left: 10, top: 20, right: 100, bottom: 60 },
        { id: 'bad-nan', left: Number.NaN, top: 0, right: 10, bottom: 10 },
        { id: 'bad-inverted', left: 50, top: 20, right: 10, bottom: 60 },
        { id: 'bad-huge', left: 0, top: 0, right: 999_999, bottom: 60 },
        { id: '', left: 0, top: 0, right: 10, bottom: 10 },
      ]),
      [{ id: 'btn-order', left: 10, top: 20, right: 100, bottom: 60 }],
    );
    assert.equal(sanitizeBase64('data:image/png;base64,iVBORw0KGgo='), 'data:image/png;base64,iVBORw0KGgo=');
    assert.equal(sanitizeBase64('<script>alert(1)</script>'), undefined);
  });

  it('広告を削除せずに閉じるボタンの位置・大きさ・文字を変えたりバナー化する工夫も通り、客も店長も納得する', () => {
    const withPopup = withAnnoyance(cleanSpec(), 'coupon_popup');
    // 初期状態では目が遠い客（和子）は閉じるボタンが極小で閉じられず離脱し、翔太も不満になる
    assert.equal(outcomeOf(withPopup, 'kazuko'), 'abandoned');
    assert.equal(outcomeOf(withPopup, 'shota'), 'bothered');

    // 1. 広告を削除せず、閉じるボタンを下部バー（大きく押しやすい「あとで見る」）に改造する
    const popupComp = findComponent(withPopup, 'popup-coupon');
    const renovatedPopup = applyPatchesToSpecWithValidation(
      withPopup,
      '広告の閉じるボタンを下部に大きく配置しました',
      [
        {
          op: 'replace_component',
          component: {
            ...popupComp,
            closePosition: 'bottom-bar',
            closeSize: 'lg',
            closeLabel: 'あとで見る',
            style: { ...popupComp.style, skin: 'native', variant: 'default' },
            actions: [{ type: 'set_state', targetKey: 'couponDiscount', value: 500 }],
          },
        },
      ],
    );
    assert.equal(renovatedPopup.rolledBack, false);
    // 作業ログに位置変更や文字変更が出ること
    const stepTexts = (renovatedPopup.workSteps ?? []).map((s) => s.text);
    assert.ok(stepTexts.some((t) => t.includes('閉じるボタンの位置')));
    // 店長も広告を残して工夫したことを喜ぶ
    assert.ok(renovatedPopup.ownerReaction?.includes('広告を消さずに押しやすく工夫して残してくれた'));
    // 目が遠い和子も翔太も嫌な思いをせず達成できる
    assert.equal(outcomeOf(renovatedPopup.updatedSpec, 'kazuko'), 'achieved');
    assert.equal(outcomeOf(renovatedPopup.updatedSpec, 'shota'), 'achieved');

    // 2. 広告を画面埋め尽くしポップアップからインラインの控えめなバナーに改造した場合も達成・店長納得になる
    const bannerized = applyPatchesToSpecWithValidation(
      withPopup,
      '広告を画面内の控えめなクーポンバナーに変えました',
      [
        {
          op: 'replace_component',
          component: {
            ...popupComp,
            type: 'status_banner',
            role: 'info',
            label: '🎁 今なら300円OFFクーポンをご利用いただけます',
            style: { variant: 'default', size: 'sm', skin: 'native' },
            actions: [{ type: 'set_state', targetKey: 'couponDiscount', value: 500 }],
          },
        },
      ],
    );
    assert.equal(bannerized.rolledBack, false);
    assert.ok(bannerized.ownerReaction?.includes('広告を消さずに押しやすく工夫して残してくれた'));
    assert.equal(outcomeOf(bannerized.updatedSpec, 'kazuko'), 'achieved');
    assert.equal(outcomeOf(bannerized.updatedSpec, 'shota'), 'achieved');
  });

  it('クーポンを使いたい人は、クーポンで実際に合計が下がったときだけ満足する', () => {
    // クーポン広告があるときだけ、割引額の受け皿が用意され、合計の式から引かれる
    assert.equal(cleanSpec().state.some((v) => v.key === 'couponDiscount'), false);
    const withPopup = withAnnoyance(cleanSpec(), 'coupon_popup');
    assert.equal(withPopup.state.find((v) => v.key === 'couponDiscount')?.initialValue, 0);
    assert.match(withPopup.derived?.find((d) => d.key === 'total')?.expression ?? '', /- couponDiscount$/);
    const popupComp = findComponent(withPopup, 'popup-coupon');
    const replacePopup = (component: AppComponent) =>
      applyPatchesToSpecWithValidation(withPopup, 'クーポンを直しました', [
        { op: 'replace_component', component },
      ]).updatedSpec;
    const shota = (spec: AppSpec) => simulatePersonaComputerUse(spec, persona('shota'));

    // 閉じやすくしただけで、押すと会員登録に飛ぶままなら、使えない
    const closableOnly = replacePopup({ ...popupComp, closePosition: 'bottom-bar', closeSize: 'lg' });
    assert.equal(shota(closableOnly).outcome, 'bothered');
    assert.equal(shota(closableOnly).frictionPoint, COUPON_FRICTIONS.unusable);

    // 押せるが合計が変わらないクーポンでは、満足しない
    const noDiscount = replacePopup({
      ...popupComp,
      type: 'status_banner',
      role: 'info',
      style: { variant: 'default', size: 'sm', skin: 'native' },
      actions: [{ type: 'set_state', targetKey: 'couponApplied', value: true }],
    });
    assert.equal(shota(noDiscount).outcome, 'bothered');
    assert.equal(shota(noDiscount).frictionPoint, COUPON_FRICTIONS.noDiscount);

    // 独自の部品でも、data-set-state で割引額を入れれば合計が下がり、満足する
    const custom = replacePopup({
      ...popupComp,
      type: 'custom',
      role: 'info',
      label: '500円OFFクーポン',
      style: { variant: 'default', size: 'md', skin: 'native' },
      customHtml: '<button type="button" data-set-state="couponDiscount:500">500円OFFクーポンを使う</button>',
      actions: [],
    });
    const result = shota(custom);
    assert.equal(result.outcome, 'achieved');
    assert.equal(result.finalState?.total, (withPopup.state.find((v) => v.key === 'unitPrice')?.initialValue as number) + 600 - 500);
  });

  it('サンドボックス内で自由なカスタムHTML部品やワンタップ入力ショートカットを追加して検証できる', () => {
    const base = cleanSpec();
    // 急いでいる健太でも確認ダイアログ付きだと手数が増えて離脱するが、ワンタップで住所と支払いを埋める＋α機能を追加すると達成できる
    findComponent(base, 'btn-order').behavior = { effect: 'confirm_dialog' };
    assert.equal(outcomeOf(base, 'kenta'), 'abandoned');

    const withShortcut = applyPatchesToSpecWithValidation(
      base,
      '前回と同じ設定でワンタップ入力できるカスタム部品を追加しました',
      [
        {
          op: 'insert_component',
          afterId: 'product-card',
          component: {
            id: 'custom-quick-preset',
            type: 'custom',
            role: 'info',
            label: '⚡ いつものお届け先・支払い方法をまとめてセット',
            customHtml:
              '<div class="quick-box"><p>前回の設定：自宅 / クレジットカード</p><button type="button" data-set-state="deliveryAddress" data-value="自宅">ワンタップでセット</button></div>',
            customCss: '.quick-box { padding: 8px; background: #f8fafc; border-radius: 8px; }',
            style: { variant: 'primary', size: 'md' },
            actions: [
              { type: 'set_state', targetKey: 'deliveryAddress', value: '自宅' },
              { type: 'set_state', targetKey: 'paymentMethod', value: 'クレジットカード' },
            ],
          },
        },
      ],
    );
    assert.equal(withShortcut.rolledBack, false);
    assert.equal(outcomeOf(withShortcut.updatedSpec, 'kenta'), 'achieved');
    assert.equal(outcomeOf(withShortcut.updatedSpec, 'harue'), 'achieved');

    // チャット経由での画面巻き戻し判定：全体巻き戻しと特定部品の復元を区別する
    assert.equal(detectCheckpointRevertCommand('1つ前の画面に戻して'), 'undo');
    assert.equal(detectCheckpointRevertCommand('やっぱり元に戻して！'), 'undo');
    assert.equal(detectCheckpointRevertCommand('さっきの変更を取り消して'), 'undo');
    assert.equal(detectCheckpointRevertCommand('一番最初の画面に戻して'), 'reset');
    assert.equal(detectCheckpointRevertCommand('最初からやり直して'), 'reset');
    assert.equal(detectCheckpointRevertCommand('定期便の欄を戻して'), null);
    assert.equal(detectCheckpointRevertCommand('消した広告を戻して'), null);
  });

  it('定期便やクーポンを単に削除すると希望していた客が不満になり、正直で使いやすい形に整えると客も売り手も満足する', () => {
    const initial = buildScenarioSpec(ACTIVE_THEME, ACTIVE_THEME.defaultScenarioId).spec;

    // 1. 注文ボタンだけ直し、広告・煽り・おとり・定期便をすべて単純削除した場合
    const deletedAll = applyPatchesToSpecWithValidation(initial, '邪魔な部品を全部消しました', [
      {
        op: 'replace_component',
        component: {
          ...findComponent(initial, 'btn-order'),
          actions: [{ type: 'complete_task', message: 'ご購入ありがとうございます' }],
        },
      },
      { op: 'remove_component', id: 'popup-coupon' },
      { op: 'remove_component', id: 'banner-countdown' },
      { op: 'remove_component', id: 'btn-buy-now' },
      { op: 'remove_component', id: 'check-subscription' },
    ]);
    assert.equal(deletedAll.rolledBack, false);

    // 定期便を希望していた ren（損をしたくない人）は設定がなくなって bothered になり、
    // 定期便を求めない takumi（さっと買いたい人）と shigeru（確実に買いたい人）は邪魔なものがなくなって満足して買える
    const renSim = simulatePersonaComputerUse(deletedAll.updatedSpec, persona('ren'));
    assert.equal(renSim.outcome, 'bothered', 'ren should be bothered when subscription is deleted');
    assert.match(renSim.frictionPoint, /定期便の設定がなかった/);
    assert.match(renSim.simulatedComment, /毎月届けてほしいのに設定がなくて残念/);
    for (const id of ['shigeru', 'takumi']) {
      const sim = simulatePersonaComputerUse(deletedAll.updatedSpec, persona(id));
      assert.equal(sim.outcome, 'achieved', `${id} should achieve when dark patterns are removed`);
    }
    // クーポンを希望していた shota / hina はクーポンがなくなって bothered になる
    for (const id of ['shota', 'hina']) {
      const sim = simulatePersonaComputerUse(deletedAll.updatedSpec, persona(id));
      assert.equal(sim.outcome, 'bothered', `${id} should be bothered when coupon is deleted`);
      assert.match(sim.frictionPoint, /クーポンがなくなっていた/);
      assert.match(sim.simulatedComment, /クーポンは無くなってしまったのか/);
    }

    // 2. 削除ではなく、定期便を正直なオプトイン欄に、クーポン広告を押しやすい下部閉じるボタン付きに改善した場合（損をしたくない客層）
    const popupComp = findComponent(initial, 'popup-coupon');
    const subComp = findComponent(initial, 'check-subscription');
    const balancedThrifty = applyPatchesToSpecWithValidation(
      initial,
      'クーポンと定期便を使いやすく整えました',
      [
        {
          op: 'replace_component',
          component: {
            ...findComponent(initial, 'btn-order'),
            actions: [{ type: 'complete_task', message: 'ご購入ありがとうございます' }],
          },
        },
        { op: 'remove_component', id: 'banner-countdown' },
        { op: 'remove_component', id: 'btn-buy-now' },
        {
          op: 'replace_component',
          component: {
            ...popupComp,
            closePosition: 'bottom-bar',
            closeSize: 'lg',
            closeLabel: '500円引きを使う',
            style: { ...popupComp.style, skin: 'native', variant: 'default' },
            actions: [{ type: 'set_state', targetKey: 'couponDiscount', value: 500 }],
          },
        },
        {
          op: 'upsert_state',
          variable: { key: 'subscription', label: '定期便', type: 'boolean', initialValue: false },
        },
        {
          op: 'replace_component',
          component: {
            ...subComp,
            label: '定期お届け便で注文する・いつでも解約OK',
            style: { variant: 'default', size: 'sm' },
          },
        },
      ],
    );
    assert.equal(balancedThrifty.rolledBack, false);

    // 損をしたくない客層（shota, hina, ren）の3人全員が achieved になり、店長の定期便も獲得できる
    const thriftyPersonas = getReviewersByCohort('thrifty');
    const thriftyResults = thriftyPersonas.map((p) =>
      simulatePersonaComputerUse(balancedThrifty.updatedSpec, p),
    );
    assert.deepEqual(
      thriftyResults.map((r) => r.outcome),
      ['achieved', 'achieved', 'achieved'],
    );
    const ownerReport = buildOwnerReport(
      ACTIVE_THEME.owner,
      balancedThrifty.updatedSpec,
      thriftyResults,
    );
    const subMetric = ownerReport.metrics.find((m) => m.label === '定期便');
    assert.equal(subMetric?.missing, false);
    assert.ok((subMetric?.count ?? 0) >= 1);
  });

  it('プロンプト・テレメトリが探索と実行を分離し、単純削除から条件設計への改善を行動指標として集計する', async () => {
    clearPromptTelemetryEvents();
    const initial = buildScenarioSpec(ACTIVE_THEME, ACTIVE_THEME.defaultScenarioId).spec;

    // 1. 一次特徴（NLシグナル）と型分類（Embeddingシグナル）の検証
    const deleteNl = extractDeterministicNlSignal('定期便を消して');
    assert.deepEqual(deleteNl.entities, ['定期便']);
    assert.equal(deleteNl.hasConditionConstraint, false);
    assert.equal(
      classifyDeterministicArchetype('定期便を消して', 'execute').closestArchetype,
      'simple_delete',
    );

    const conditionNl = extractDeterministicNlSignal(
      '定期便の初期チェックを外して解約条件を明記して',
    );
    assert.ok(conditionNl.entities.includes('定期便'));
    assert.ok(conditionNl.entities.includes('チェック'));
    assert.ok(conditionNl.entities.includes('解約条件'));
    assert.equal(conditionNl.hasConditionConstraint, true);
    assert.equal(
      classifyDeterministicArchetype(
        '定期便の初期チェックを外して解約条件を明記して',
        'execute',
      ).closestArchetype,
      'condition_design',
    );

    // 誰がどうできるようにしたいかという目的を言葉にした指示は goal_directed として数える
    assert.equal(
      classifyDeterministicArchetype('定期便を望む人だけが自分で選べるようにして', 'execute')
        .closestArchetype,
      'goal_directed',
    );
    assert.equal(
      classifyDeterministicArchetype('いい感じに', 'execute').closestArchetype,
      'vague_delegate',
    );

    assert.equal(cosineSimilarity([1, 0], [1, 0]), 1);
    assert.equal(cosineSimilarity([1, 0], [0, 1]), 0);

    // 2. 探索型（店長・客への質問）は往復数のペナルティ（初回成功率の分母）に入らない
    await recordPromptTelemetryEvent({
      cohortId: 'careful',
      turnIndex: 0,
      promptText: '店長、定期便ってどうして付けてるの？',
      sessionMode: 'explore',
      recipient: 'owner',
      hasPenAnnotation: false,
    });
    await recordPromptTelemetryEvent({
      cohortId: 'careful',
      turnIndex: 1,
      promptText: '客B、どこで困ったの？',
      sessionMode: 'explore',
      recipient: 'shigeru',
      hasPenAnnotation: false,
    });

    // 3. 実行1回目：定期便を単純削除してしまった場合（removedNeededFeature: true）
    const deletedPatch: SpecPatch[] = [{ op: 'remove_component', id: 'check-subscription' }];
    const deletedResult = applyPatchesToSpecWithValidation(
      initial,
      '定期便を消しました',
      deletedPatch,
    );
    const deletedStats = inspectAppliedChangeForTelemetry(
      initial,
      deletedResult.updatedSpec,
      deletedPatch,
    );
    assert.equal(deletedStats.removedNeededFeature, true);

    await recordPromptTelemetryEvent({
      cohortId: 'careful',
      turnIndex: 2,
      promptText: '定期便を消して',
      sessionMode: 'execute',
      recipient: 'assistant',
      hasPenAnnotation: true,
      changedComponentIds: deletedStats.changedComponentIds,
      removedNeededFeature: deletedStats.removedNeededFeature,
    });

    // 4. 実行2回目：同じ定期便部品を、初期チェック解除＋解約条件明記の形で整え直した場合
    const subComp = findComponent(initial, 'check-subscription');
    const restoredPatches: SpecPatch[] = [
      {
        op: 'insert_component',
        afterId: 'summary-price',
        component: {
          ...subComp,
          label: '定期お届け便で注文する・いつでも解約OK',
          style: { variant: 'default', size: 'sm' },
        },
      },
      {
        op: 'upsert_state',
        variable: { key: 'subscription', label: '定期便', type: 'boolean', initialValue: false },
      },
    ];
    const restoredResult = applyPatchesToSpecWithValidation(
      deletedResult.updatedSpec,
      '定期便の条件を整えました',
      restoredPatches,
    );
    const restoredStats = inspectAppliedChangeForTelemetry(
      deletedResult.updatedSpec,
      restoredResult.updatedSpec,
      restoredPatches,
    );
    assert.equal(restoredStats.removedNeededFeature, false);

    const secondExecEvent = await recordPromptTelemetryEvent({
      cohortId: 'careful',
      turnIndex: 3,
      promptText: '定期便の初期チェックを外して解約条件を明記して',
      sessionMode: 'execute',
      recipient: 'assistant',
      hasPenAnnotation: true,
      changedComponentIds: restoredStats.changedComponentIds,
      removedNeededFeature: restoredStats.removedNeededFeature,
    });
    assert.equal(secondExecEvent.isReworkOnSameComponent, true);
    // 指示の本文は記録に残さない
    assert.equal('promptText' in secondExecEvent, false);

    // 5. クリア時のサマリー生成とシステムメッセージ行（括弧なし・インラインコード付き）の確認
    const summary = await finalizeTelemetryOnEvaluate('careful');
    assert.equal(summary.exploreTurns, 2);
    assert.equal(summary.executeTurns, 2);
    assert.equal(summary.reworkTurns, 1);
    // 必要な機能を消した指示は、直し直しとは別に数える。1回で思いどおりに直った割合では、これまでどおり差し引く
    assert.equal(summary.removedNeededTurns, 1);
    assert.equal(summary.firstShotSuccessRate, 50);
    assert.ok(summary.turningPoint);
    assert.deepEqual(summary.turningPoint, { beforeTurn: 2, afterTurn: 3 });
    assert.equal(summary.systemLogLines.length, 3);
    assert.match(
      summary.systemLogLines[0],
      /質問した `2回` 直してと頼んだ `2回` うち目的を伝えた `0回` 同じ所を直し直した `1回` 必要な機能を消した `1回` 1回で思いどおりに直った割合 `50%`/,
    );
    // 目的を伝えた指示がないときも、数える決まりを示す
    assert.match(summary.systemLogLines[1], /目的を伝えた指示はありませんでした 誰が、どうできるようにしたいかを伝えると数えます/);
    // サーバーの振り返り文には本文がなく、何番目の発言かの目印だけが入る
    assert.match(summary.systemLogLines[2], /`\{\{turn:2\}\}` から条件を具体化した `\{\{turn:3\}\}`/);
    // ブラウザが手元の本文に置き換える
    const turnTexts = ['店長、定期便ってどうして付けてるの？', '客B、どこで困ったの？', '定期便を消して', '定期便の初期チェックを外して解約条件を明記して'];
    assert.match(
      fillTurnRefs(summary.systemLogLines[2], turnTexts),
      /`定期便を消して` から条件を具体化した `定期便の初期チェックを外して解約条件を…`/,
    );
    // 長い指示は、文の中で折り返さない長さに縮め、句読点か助詞の切れ目で切る
    assert.equal(
      fillTurnRefs('{{turn:0}}', ['このクーポン広告のポップアップは表示しないでください。最初に開いたときも出さないで']),
      'このクーポン広告のポップアップは表示しないで…',
    );
    assert.equal(fillTurnRefs('{{turn:0}}', ['このボタンは押しても先に進まず迷わせるだけなので、画面から消してください']), 'このボタンは押しても先に…');
    // 本文が手元にないとき（画面を開き直したなど）は、その指示、と書く
    assert.match(fillTurnRefs(summary.systemLogLines[2], []), /`その指示` から条件を具体化した `その指示`/);
    // Natural Language API が取り出した語でも、題材の語彙にないもの（人名など）は残さない
    assert.deepEqual(limitEntitiesToVocabulary(['山田太郎', '定期便のチェック', '090-1234-5678', '定期便']), ['定期便']);
    for (const line of summary.systemLogLines) {
      assert.doesNotMatch(line, /[()（）「」『』]/);
    }

    // 必要な機能を消したまま直し直さなかったときは、直し直した回数には入れず、割合からは差し引く
    const removedOnlySummary = summarizePromptTelemetry([
      { ...secondExecEvent, turnIndex: 0, isReworkOnSameComponent: false, removedNeededFeature: true },
    ]);
    assert.equal(removedOnlySummary.reworkTurns, 0);
    assert.equal(removedOnlySummary.removedNeededTurns, 1);
    assert.equal(removedOnlySummary.firstShotSuccessRate, 0);
    assert.match(removedOnlySummary.systemLogLines[0], /同じ所を直し直した `0回` 必要な機能を消した `1回`/);

    // 理由を書いただけの指示は目的に数えず、誰がどうできるようにしたいかを書いた指示は数える
    assert.equal(extractDeterministicNlSignal('押しても先に進まず迷わせるだけなので、画面から消してください').hasGoalStatement, false);
    assert.equal(extractDeterministicNlSignal('自分で選んだ人だけ定期便になるようにしてください').hasGoalStatement, true);
    const goalByShapeSummary = summarizePromptTelemetry([
      {
        ...secondExecEvent,
        turnIndex: 4,
        isReworkOnSameComponent: false,
        removedNeededFeature: false,
        embeddingSignal: { ...secondExecEvent.embeddingSignal, closestArchetype: 'condition_design' },
        nlSignal: { ...secondExecEvent.nlSignal, hasGoalStatement: true },
      },
    ]);
    assert.equal(goalByShapeSummary.goalDirectedTurns, 1);
    assert.match(goalByShapeSummary.systemLogLines[1], /目的を伝えた指示 `\{\{turn:4\}\}` 誰が、どうできるようにしたいかを伝えた指示を数えています/);

    // 6. 手戻りなしで一発到達したケースの初回成功率100%確認
    const oneShotSummary = summarizePromptTelemetry([
      {
        ...secondExecEvent,
        isReworkOnSameComponent: false,
        removedNeededFeature: false,
      },
    ]);
    assert.equal(oneShotSummary.firstShotSuccessRate, 100);
    assert.match(oneShotSummary.systemLogLines[0], /1回で思いどおりに直った割合 `100%`/);
    // 目的を伝えた指示がないときも、0回と出して、数える決まりを次の行で示す
    assert.match(oneShotSummary.systemLogLines[0], /うち目的を伝えた `0回`/);
    assert.match(oneShotSummary.systemLogLines[1], /目的を伝えた指示はありませんでした/);

    // 目的だけを伝えると、AIが部品を書き換えるときに label や style を省くことがある。元の部品から補い、描画が落ちない
    const couponComp = findComponent(initial, 'popup-coupon');
    const { label: _omitLabel, style: _omitStyle, ...couponWithoutLabel } = couponComp;
    const partialResult = applyPatchesToSpecWithValidation(initial, 'クーポンを整えました', [
      { op: 'replace_component', component: couponWithoutLabel as typeof couponComp },
    ]);
    const patchedCoupon = findComponent(partialResult.updatedSpec, 'popup-coupon');
    assert.equal(patchedCoupon.label, couponComp.label);
    assert.deepEqual(patchedCoupon.style, couponComp.style);

    // 7. 単純削除のあと、目的を伝えた指示で直したケース
    const goalPrompt = '定期便を望む人だけが自分で選べるようにして';
    const goalSummary = summarizePromptTelemetry([
      {
        ...secondExecEvent,
        turnIndex: 0,
        embeddingSignal: classifyDeterministicArchetype('定期便を消して', 'execute'),
        isReworkOnSameComponent: false,
        removedNeededFeature: true,
      },
      {
        ...secondExecEvent,
        turnIndex: 1,
        embeddingSignal: classifyDeterministicArchetype(goalPrompt, 'execute'),
        isReworkOnSameComponent: true,
        removedNeededFeature: false,
      },
    ]);
    assert.equal(goalSummary.goalDirectedTurns, 1);
    assert.match(goalSummary.systemLogLines[0], /うち目的を伝えた `1回`/);
    assert.match(
      fillTurnRefs(goalSummary.systemLogLines[1], ['定期便を消して', goalPrompt]),
      /目的を伝えた指示 `定期便を望む人だけが自分で選べるようにして`/,
    );
    assert.match(
      fillTurnRefs(goalSummary.systemLogLines[2], ['定期便を消して', goalPrompt]),
      /`定期便を消して` から目的を伝えた `定期便を望む人だけが自分で選べるようにして`/,
    );
    for (const line of goalSummary.systemLogLines) {
      assert.doesNotMatch(line, /[()（）「」『』]/);
    }

    // 8. 別の部品（広告の単純削除 → 定期便の条件設計）は転換点として結ばない
    const unrelatedSummary = summarizePromptTelemetry([
      {
        ...secondExecEvent,
        turnIndex: 0,
        changedComponentIds: ['popup-coupon'],
        nlSignal: extractDeterministicNlSignal('あと最初に出てくる500円OFFの広告もじゃまだから出ないようにして'),
        embeddingSignal: classifyDeterministicArchetype('あと最初に出てくる500円OFFの広告もじゃまだから出ないようにして', 'execute'),
        isReworkOnSameComponent: false,
        removedNeededFeature: true,
      },
      {
        ...secondExecEvent,
        turnIndex: 1,
        changedComponentIds: ['check-subscription'],
        nlSignal: extractDeterministicNlSignal('勝手に定期便になるのひどい。チェックは最初外して、字ももっと大きくして'),
        embeddingSignal: classifyDeterministicArchetype('勝手に定期便になるのひどい。チェックは最初外して、字ももっと大きくして', 'execute'),
        isReworkOnSameComponent: false,
        removedNeededFeature: false,
      },
    ]);
    assert.equal(unrelatedSummary.turningPoint, undefined);

    // 9. クリア後レポートのお助けAI 3通フィードバック（概要・よかった点・次にどうすると良くなるか）
    const clearFactsWithTurningPoint = buildClearFeedbackFacts({
      cohortLabel: '損をしたくない人',
      summary: goalSummary,
      owner: {
        orders: 3,
        customers: 3,
        metrics: [{ label: 'おトク定期便', count: 1, missing: false }],
        comment: '',
        closingLines: [],
      },
      otherCohorts: {
        achieved: 4,
        total: 6,
        unhappy: [{ name: '和子', reason: '文字が小さくて読めなかった' }],
      },
      couponStatus: 'usable_discount',
    });
    assert.equal(clearFactsWithTurningPoint.fallbackLines.length, 3);
    assert.match(clearFactsWithTurningPoint.fallbackLines[0], /`損をしたくない人`/);
    assert.match(clearFactsWithTurningPoint.fallbackLines[0], /おトク定期便も `1件` 入って/);
    const filledGoodPoint = fillTurnRefs(clearFactsWithTurningPoint.fallbackLines[1], [
      '定期便を消して',
      goalPrompt,
    ]);
    assert.match(filledGoodPoint, /`定期便を消して`/);
    assert.match(filledGoodPoint, /`定期便を望む人だけが自分で選べるようにして`/);
    assert.match(clearFactsWithTurningPoint.fallbackLines[2], /`和子` が `文字が小さくて読めなかった`/);
    for (const line of clearFactsWithTurningPoint.fallbackLines) {
      assert.doesNotMatch(line, /[()（）「」『』]/);
    }

    // 定期便やクーポンを消してクリアした場合は、消さずに正直な形へ変える伸びしろを提案する
    const clearFactsDeleted = buildClearFeedbackFacts({
      cohortLabel: '急いでいる人',
      summary: oneShotSummary,
      owner: {
        orders: 3,
        customers: 3,
        metrics: [{ label: 'おトク定期便', count: 0, missing: true }],
        comment: '',
        closingLines: [],
      },
      otherCohorts: {
        achieved: 6,
        total: 6,
        unhappy: [],
      },
      couponStatus: 'removed',
    });
    assert.match(clearFactsDeleted.fallbackLines[0], /おトク定期便は消えて0件になりましたが/);
    assert.match(clearFactsDeleted.fallbackLines[2], /おトク定期便を消してクリアしましたが/);
    assert.match(clearFactsDeleted.fallbackLines[2], /最初のチェックを外して条件が分かる形で残すと両立できます/);
    for (const line of clearFactsDeleted.fallbackLines) {
      assert.doesNotMatch(line, /[()（）「」『』]/);
    }

    // 10. 赤ペンで囲んだときの候補は、手直しの答えではなく調べるための質問候補3つにし、店長にも指した部品を伝える
    assert.deepEqual(CIRCLE_INQUIRY_FOLLOW_UPS, [
      'この部品の仕組みを聞く',
      'お客さんの反応を聞く',
      '店長の考えを聞く',
    ]);
    for (const inquiry of CIRCLE_INQUIRY_FOLLOW_UPS) {
      assert.equal(classifyUserMessageIntent(inquiry), 'triage');
      assert.equal(classifyDeterministicArchetype(inquiry, 'explore').closestArchetype, 'inquiry');
    }
    const ownerFactsWithTarget = describeOwnerContext(initial, undefined, ['check-subscription']);
    assert.ok(ownerFactsWithTarget.some((f) => f.includes('赤ペンで指して聞いている部品')));

    // 11. 誤送信を取り消して送り直したとき（同じ turnIndex の上書き・turnCount 指定）に、取り消した発言が集計に残らない
    await recordPromptTelemetryEvent({
      runId: 'run-undo-test',
      cohortId: 'thrifty',
      turnIndex: 0,
      promptText: '間違えて送った質問',
      sessionMode: 'explore',
      recipient: 'owner',
      hasPenAnnotation: false,
    });
    await recordPromptTelemetryEvent({
      runId: 'run-undo-test',
      cohortId: 'thrifty',
      turnIndex: 1,
      promptText: 'これも取り消す質問',
      sessionMode: 'explore',
      recipient: 'customer',
      hasPenAnnotation: false,
    });
    // turnIndex: 0 から送り直した場合は turnIndex >= 0 の古い記録が置き換わる
    await recordPromptTelemetryEvent({
      runId: 'run-undo-test',
      cohortId: 'thrifty',
      turnIndex: 0,
      promptText: '定期便を望む人だけが自分で選べるようにして',
      sessionMode: 'execute',
      recipient: 'assistant',
      hasPenAnnotation: false,
      changedComponentIds: ['check-subscription'],
      removedNeededFeature: false,
    });
    const undoSummary = await finalizeTelemetryOnEvaluate('thrifty', 'run-undo-test', 1);
    assert.equal(undoSummary.exploreTurns, 0);
    assert.equal(undoSummary.executeTurns, 1);
  });

  it('お客さんの反応を見る前の予想と結果を突き合わせ、困りごとを見抜けたかも含めて振り返り文を作れる', () => {
    const sampleResults = [
      { reviewerId: 'p1', outcome: 'achieved' as const },
      { reviewerId: 'p2', outcome: 'bothered' as const },
      { reviewerId: 'p3', outcome: 'regretted' as const },
    ];

    // 予想せずすぐ見た回
    const skipped = scorePredictionRound(sampleResults, undefined);
    assert.deepEqual(skipped, { predicted: false, total: 0, hits: 0, caughtIssueCount: 0 });
    assert.equal(formatPredictionReportLine([skipped]), null);

    // 1回目：不満や後悔も含めて予想し、3人中2人的中（うち困りごと1人的中）
    const round1 = scorePredictionRound(sampleResults, {
      p1: 'achieved',
      p2: 'bothered',
      p3: 'achieved',
    });
    assert.deepEqual(round1, { predicted: true, total: 3, hits: 2, caughtIssueCount: 1 });

    // 2回目：全員達成した回で3人とも的中
    const round2 = scorePredictionRound(
      [
        { reviewerId: 'p1', outcome: 'achieved' },
        { reviewerId: 'p2', outcome: 'achieved' },
        { reviewerId: 'p3', outcome: 'achieved' },
      ],
      { p1: 'achieved', p2: 'achieved', p3: 'achieved' },
    );
    assert.deepEqual(round2, { predicted: true, total: 3, hits: 3, caughtIssueCount: 0 });

    const summary = summarizePredictions([skipped, round1, round2]);
    assert.equal(summary.totalRounds, 3);
    assert.equal(summary.predictedRounds, 2);
    assert.equal(summary.total, 6);
    assert.equal(summary.hits, 5);
    assert.equal(summary.caughtIssue, true);

    const reportWithIssue = formatPredictionReportLine([skipped, round1, round2]);
    assert.equal(
      reportWithIssue,
      'お客さんの反応の予想は `6人中5人` 当たり、どこで困るかも見抜けました',
    );
    const reportAchievedOnly = formatPredictionReportLine([round2]);
    assert.equal(reportAchievedOnly, 'お客さんの反応の予想は `3人中3人` 当たりました');

    const matchedBadge = formatPredictionBadgeText('bothered', 'bothered');
    assert.equal(matchedBadge.matched, true);
    assert.equal(matchedBadge.text, '○ 買えたけど不満');

    const missedBadge = formatPredictionBadgeText('achieved', 'bothered');
    assert.equal(missedBadge.matched, false);
    assert.equal(missedBadge.text, '× 買えたけど不満');
  });
});


describe('判定の再現性', () => {
  /** 店にクーポン広告がある画面に、独自の部品で作ったクーポンを置いた画面 */
  function specWithCustomCoupon(): AppSpec {
    const spec = withAnnoyance(cleanSpec(), 'coupon_popup');
    spec.components = spec.components.filter((c) => c.id !== 'popup-coupon');
    const idx = spec.components.findIndex((c) => c.id === 'summary');
    spec.components.splice(idx, 0, {
      id: 'custom-coupon',
      type: 'custom',
      role: 'info',
      label: 'その場で使える割引',
      style: { size: 'md', variant: 'default' },
      customHtml:
        '<div class="coupon"><p>会員登録なしで今すぐ使えます</p><button type="button" data-set-state="couponDiscount:500">500円引きを使う</button></div>',
      actions: [],
    });
    return spec;
  }

  const COUPON_PROFILE: CustomComponentProfile = {
    purpose: 'coupon',
    userChooses: true,
    enabledByDefault: false,
    textSize: 'md',
    tapTargetSize: 'md',
    hasHiddenCondition: false,
    hardToClose: false,
    pressure: false,
  };

  /** 本文の部品を上から並べ、画面下の固定バーを本文の下に置いた、決まった条件の座標に見立てた値 */
  function boundsLike(spec: AppSpec, overrides: Record<string, Partial<ComponentBoundsInput>> = {}) {
    let y = 72;
    const out: ComponentBoundsInput[] = [];
    for (const c of spec.components) {
      if (c.type === 'popup' || c.style.sticky) continue;
      const h = c.type === 'choice_cards' ? 136 : c.type === 'summary' ? 56 : c.type === 'product_card' ? 104 : 28;
      out.push({ id: c.id, left: 12, top: y, right: 308, bottom: y + h, ...overrides[c.id] });
      y += h + 8;
    }
    out.push({ id: 'total-price', left: 13, top: y + 16, right: 54, bottom: y + 43, ...overrides['total-price'] });
    out.push({ id: 'btn-order', left: 64, top: y + 8, right: 307, bottom: y + 52, ...overrides['btn-order'] });
    return out;
  }

  it('同じ画面の定義と同じ客層なら、評価を何回呼んでも判定は一致する', () => {
    const spec = withAnnoyance(withAnnoyance(cleanSpec(), 'subscription_trap'), 'coupon_popup');
    const bounds = boundsLike(spec);
    for (const cohort of TARGET_COHORTS) {
      const reviewers = getReviewersByCohort(cohort.id);
      const runs = [0, 1, 2].map(() => {
        // 呼ぶたびに別のオブジェクトにしても、中身が同じなら同じ結果になる
        const map = simulateReviewers(structuredClone(spec), reviewers, {}, structuredClone(bounds));
        return reviewers.map((p) => {
          const r = map.get(p.id)!;
          return [r.outcome, r.frictionPoint, r.actionTrace.join('|')];
        });
      });
      assert.deepEqual(runs[1], runs[0], cohort.id);
      assert.deepEqual(runs[2], runs[0], cohort.id);
    }
  });

  it('独自の部品は、2回目以降は保存した読み取り結果を使い、判定が一致する', async () => {
    clearCustomProfileMemoryCache();
    const spec = specWithCustomCoupon();
    // 呼ぶたびに違う答えを返す読み取り（temperature で揺れる Gemini に見立てる）
    let calls = 0;
    const flaky = async (): Promise<CustomComponentProfile> => {
      calls++;
      return calls === 1 ? COUPON_PROFILE : { ...COUPON_PROFILE, purpose: 'ad', textSize: 'xs' };
    };
    const thrifty = getReviewersByCohort('thrifty');
    const outcomesOf = (profiles: Record<string, CustomComponentProfile>) => {
      const map = simulateReviewers(spec, thrifty, profiles, boundsLike(spec));
      return thrifty.map((p) => map.get(p.id)!.outcome);
    };

    const first = await resolveCustomComponentProfiles(spec, flaky, 'test-model');
    const second = await resolveCustomComponentProfiles(structuredClone(spec), flaky, 'test-model');
    assert.equal(calls, 1);
    assert.deepEqual(second.profiles, first.profiles);
    assert.deepEqual(outcomesOf(second.profiles), outcomesOf(first.profiles));

    // 部品のIDが違っても、中身が同じなら同じ読み取り結果を使う
    const renamed = structuredClone(spec);
    findComponent(renamed, 'custom-coupon').id = 'custom-coupon-2';
    const third = await resolveCustomComponentProfiles(renamed, flaky, 'test-model');
    assert.equal(calls, 1);
    assert.deepEqual(third.profiles['custom-coupon-2'], first.profiles['custom-coupon']);

    // 中身が変われば読み取り直す
    const edited = structuredClone(spec);
    findComponent(edited, 'custom-coupon').customHtml += '<small>毎月自動で届きます</small>';
    await resolveCustomComponentProfiles(edited, flaky, 'test-model');
    assert.equal(calls, 2);

    // 読み取りに失敗したときは保存せず、その回はHTMLに書かれた動きと文字だけで判定し、次は読み取り直す
    clearCustomProfileMemoryCache();
    let failing = 0;
    const timeout = async (): Promise<CustomComponentProfile> => {
      failing++;
      throw new Error('timeout');
    };
    const failed = await resolveCustomComponentProfiles(spec, timeout, 'test-model');
    assert.deepEqual(failed.fallbackIds, ['custom-coupon']);
    assert.deepEqual(failed.profiles, {});
    await resolveCustomComponentProfiles(spec, timeout, 'test-model');
    assert.equal(failing, 2);
    const recovered = await resolveCustomComponentProfiles(spec, async () => COUPON_PROFILE, 'test-model');
    assert.deepEqual(recovered.fallbackIds, []);
    clearCustomProfileMemoryCache();
  });

  it('独自の部品で作った、その場で割引になり自分で選べるクーポンは、既存のクーポン部品と同じく達成になる', () => {
    const spec = specWithCustomCoupon();
    const thrifty = getReviewersByCohort('thrifty');
    const custom = simulateReviewers(spec, thrifty, { 'custom-coupon': COUPON_PROFILE }, boundsLike(spec));

    // 同じ場所に既存の部品で作ったクーポン
    const native = structuredClone(spec);
    const idx = native.components.findIndex((c) => c.id === 'custom-coupon');
    native.components[idx] = {
      id: 'custom-coupon',
      type: 'button',
      role: 'info',
      label: '500円OFFクーポンを使う',
      style: { size: 'md', variant: 'default' },
      actions: [{ type: 'set_state', targetKey: 'couponDiscount', value: 500 }],
    };
    const standard = simulateReviewers(native, thrifty, {}, boundsLike(native));

    for (const p of thrifty) {
      assert.equal(custom.get(p.id)!.outcome, standard.get(p.id)!.outcome, p.id);
    }
    const shota = custom.get('shota')!;
    assert.equal(shota.outcome, 'achieved');
    assert.ok(shota.actionTrace.some((t) => t.includes('特典を適用した')));
    assert.equal(shota.finalState?.total, 1980 + 600 - 500);

    // 判定用の部品はコードの規則で組み立てる。同じ読み取り結果なら、同じ部品になる
    const a = prepareSimulationInput(spec, { 'custom-coupon': COUPON_PROFILE });
    const b = prepareSimulationInput(structuredClone(spec), { 'custom-coupon': { ...COUPON_PROFILE } });
    assert.deepEqual(a.spec, b.spec);
    const proxy = findComponent(a.spec, 'custom-coupon');
    assert.equal(proxy.customHtml, undefined);
    assert.deepEqual(proxy.actions, [{ type: 'set_state', targetKey: 'couponDiscount', value: 500 }]);

    // 最初から申し込まれた独自の定期便の欄は、読まない人が気づかずに後悔する罠として扱う
    const sub = withAnnoyance(cleanSpec(), 'subscription_trap');
    findComponent(sub, 'check-subscription').type = 'custom';
    findComponent(sub, 'check-subscription').customHtml =
      '<label><input type="checkbox" data-state-key="subscription" checked>毎月お届け</label>';
    const trapMap = simulateReviewers(sub, [persona('hina')], {
      'check-subscription': { ...COUPON_PROFILE, purpose: 'subscription', userChooses: false, enabledByDefault: true },
    });
    assert.equal(trapMap.get('hina')!.outcome, 'regretted');
  });

  it('ボタンの大きさを変えると、指がずれやすいお客さんの結果が変わる', () => {
    const spec = cleanSpec();
    const shigeru = persona('shigeru');
    const small = boundsLike(spec, { 'btn-order': { bottom: undefined } }).map((b) =>
      b.id === 'btn-order' ? { ...b, bottom: b.top + 30 } : b,
    );
    const large = boundsLike(spec).map((b) => (b.id === 'btn-order' ? { ...b, bottom: b.top + 54 } : b));
    const r1 = simulatePersonaComputerUse(spec, shigeru, small);
    const r2 = simulatePersonaComputerUse(spec, shigeru, large);
    assert.equal(r1.outcome, 'abandoned');
    assert.ok(r1.actionTrace.some((t) => t.includes('指がズレて外れた')));
    assert.equal(r2.outcome, 'achieved');

    // 座標のない見積もりでも、部品の大きさの指定が効く
    const tiny = cleanSpec();
    findComponent(tiny, 'btn-order').style = { size: 'xs', variant: 'primary', align: 'center' };
    const big = cleanSpec();
    findComponent(big, 'btn-order').style = { size: 'lg', variant: 'primary', align: 'center', fullWidth: true };
    assert.equal(outcomeOf(tiny, 'takumi'), 'abandoned');
    assert.equal(outcomeOf(big, 'takumi'), 'achieved');
  });

  it('見えている画面の大きさやスクロール位置が変わっても、判定用の座標と判定は変わらない', () => {
    const spec = withAnnoyance(withAnnoyance(cleanSpec(), 'subscription_trap'), 'coupon_popup');
    const base = boundsLike(spec);
    const stickyIds = new Set(['total-price', 'btn-order']);
    const bodyTop = 60;
    const flowBottom = Math.max(...base.filter((b) => !stickyIds.has(b.id)).map((b) => b.bottom));
    const stickyTop = 520;
    /** 決まった座標の画面を、倍率 scale・スクロール位置 scrollTop で描画して測ったときの値に戻す */
    const measuredAt = (scale: number, scrollTop: number) => {
      const contentHeight = flowBottom + 8 - bodyTop;
      const barTopInContent = bodyTop + contentHeight + 8;
      const rects: MeasuredComponentRect[] = base.map((b) => {
        const sticky = stickyIds.has(b.id);
        const top = sticky ? stickyTop + (b.top - barTopInContent) : b.top - scrollTop;
        const bottom = sticky ? stickyTop + (b.bottom - barTopInContent) : b.bottom - scrollTop;
        return { id: b.id, left: b.left * scale, top: top * scale, right: b.right * scale, bottom: bottom * scale, sticky };
      });
      return canonicalizeMeasuredBounds(rects, {
        viewportWidth: FIXED_LAYOUT_VIEWPORT.width * scale,
        scrollTop: scrollTop * scale,
        bodyTop: bodyTop * scale,
        bodyScrollHeight: contentHeight * scale,
        stickyBarTop: stickyTop * scale,
      });
    };
    const reference = measuredAt(1, 0);
    for (const [scale, scrollTop] of [
      [1, 137],
      [1.5, 0],
      [1.5, 211],
      [0.8, 64],
    ]) {
      assert.deepEqual(measuredAt(scale, scrollTop), reference, `scale ${scale} scroll ${scrollTop}`);
    }
    // 固定バーの部品は本文の下に並び、本文の部品と重ならない
    const orderBox = reference.find((b) => b.id === 'btn-order')!;
    assert.ok(reference.filter((b) => !stickyIds.has(b.id)).every((b) => b.bottom <= orderBox.top));
    for (const p of ALL_REVIEWER_PERSONAS) {
      assert.deepEqual(
        simulatePersonaComputerUse(spec, p, measuredAt(1.5, 211)).outcome,
        simulatePersonaComputerUse(spec, p, reference).outcome,
        p.id,
      );
    }

    // 確認画面のように表示条件でだけ出る部品も、その画面を描画して測る。広告は出る前の状態で描く
    const confirm = cleanSpec();
    confirm.state.push({ key: 'step', label: '画面', type: 'string', initialValue: 'input' });
    confirm.components.push({
      id: 'btn-confirm-order',
      type: 'button',
      role: 'primary_action',
      label: 'この内容で注文する',
      visibleWhen: { stateKey: 'step', operator: 'eq', value: 'confirm' },
      style: { size: 'lg', variant: 'primary' },
      actions: [{ type: 'complete_task' }],
    });
    const variants = fixedLayoutStateVariants(withAnnoyance(confirm, 'coupon_popup'));
    assert.equal(variants.length, 2);
    assert.equal(variants[0].couponPopup, 'waiting');
    assert.equal(variants[1].step, 'confirm');
    assert.deepEqual(
      mergeBoundsVariants([[{ id: 'a', left: 0, top: 0, right: 1, bottom: 1 }], [{ id: 'a', left: 5, top: 5, right: 9, bottom: 9 }, { id: 'b', left: 0, top: 0, right: 2, bottom: 2 }]]),
      [{ id: 'a', left: 0, top: 0, right: 1, bottom: 1 }, { id: 'b', left: 0, top: 0, right: 2, bottom: 2 }],
    );
  });

  it('デモ用の問題で4つの指示を適用した後の画面は、決まった条件で測った座標でも3人全員が達成になる', () => {
    const demo = buildScenarioSpec(ACTIVE_THEME, 'demo').spec;
    const step = (spec: AppSpec, patches: SpecPatch[]) => {
      const result = applyPatchesToSpecWithValidation(spec, '手直ししました', patches);
      assert.equal(result.rolledBack, false);
      return result.updatedSpec;
    };
    const order = findComponent(demo, 'btn-order');
    const s1 = step(demo, [
      { op: 'replace_component', component: { ...order, actions: [{ type: 'complete_task', message: 'ご購入ありがとうございます' }] } },
    ]);
    const popup = findComponent(s1, 'popup-coupon');
    const s2 = step(s1, [
      {
        op: 'replace_component',
        component: {
          ...popup,
          closePosition: 'bottom-bar',
          closeSize: 'lg',
          closeLabel: '500円引きを使う',
          actions: [{ type: 'set_state', targetKey: 'couponDiscount', value: 500 }],
        },
      },
    ]);
    const s3 = step(s2, [{ op: 'remove_component', id: 'check-subscription' }]);
    const sub = findComponent(demo, 'check-subscription');
    const s4 = step(s3, [
      { op: 'upsert_state', variable: { key: 'subscription', label: '定期便', type: 'boolean', initialValue: false } },
      {
        op: 'insert_component',
        afterId: 'product-card',
        component: { ...sub, label: '定期便にする 毎月お届け いつでも解約OK', style: { size: 'sm', variant: 'default' } },
      },
    ]);
    // ブラウザで、決まった大きさ（320×583）で画面外に描画して測った座標
    const fixedBounds = JSON.parse(
      readFileSync(new URL('./fixtures/demo-final-fixed-bounds.json', import.meta.url), 'utf8'),
    ) as ComponentBoundsInput[];
    assert.deepEqual(
      new Set(fixedBounds.map((b) => b.id)),
      new Set(s4.components.filter((c) => c.type !== 'popup').map((c) => c.id)),
    );
    const thrifty = getReviewersByCohort('thrifty');
    for (let i = 0; i < 3; i++) {
      const map = simulateReviewers(s4, thrifty, {}, fixedBounds);
      assert.deepEqual(
        thrifty.map((p) => map.get(p.id)!.outcome),
        ['achieved', 'achieved', 'achieved'],
      );
    }
  });
});


describe('任意の選択欄と押しやすいチェック欄', () => {
  const loadBounds = (name: string) =>
    JSON.parse(readFileSync(new URL(`./fixtures/${name}`, import.meta.url), 'utf8')) as ComponentBoundsInput[];

  /** デモで定期便を消したあとの画面と、実装AIが定期便の欄を必須で作り直す差分 */
  function demoAfterRemoval() {
    const demo = buildScenarioSpec(ACTIVE_THEME, 'demo').spec;
    const order = findComponent(demo, 'btn-order');
    let spec = applyPatchesToSpecWithValidation(demo, 'a', [
      { op: 'replace_component', component: { ...order, actions: [{ type: 'complete_task', message: 'ご購入ありがとうございます' }] } },
    ]).updatedSpec;
    const popup = findComponent(spec, 'popup-coupon');
    spec = applyPatchesToSpecWithValidation(spec, 'b', [
      {
        op: 'replace_component',
        component: {
          ...popup,
          closePosition: 'bottom-bar',
          closeSize: 'lg',
          closeLabel: '500円引きを使う',
          actions: [{ type: 'set_state', targetKey: 'couponDiscount', value: 500 }],
        },
      },
    ]).updatedSpec;
    spec = applyPatchesToSpecWithValidation(spec, 'c', [{ op: 'remove_component', id: 'check-subscription' }]).updatedSpec;
    const requiredPatches: SpecPatch[] = [
      { op: 'upsert_state', variable: { key: 'subscription', label: '定期便', type: 'boolean', initialValue: false } },
      {
        op: 'insert_component',
        afterId: 'product-card',
        component: {
          id: 'check-subscription',
          type: 'checkbox',
          role: 'required_input',
          label: '毎月届く定期便にする いつでも解約できます',
          stateKey: 'subscription',
          style: { size: 'sm', variant: 'default' },
          actions: [{ type: 'toggle_state', targetKey: 'subscription' }],
        },
      },
    ];
    return { spec, requiredPatches };
  }

  it('「自分で選べるようにしたい」という指示で必須になった選択欄は、任意の欄に直される', () => {
    const { spec, requiredPatches } = demoAfterRemoval();
    const result = applyPatchesToSpecWithValidation(
      spec,
      '定期便を選べるようにしました',
      requiredPatches,
      '定期便は、毎月届けてほしい人だけが自分で選べるようにしたい',
    );
    assert.equal(result.rolledBack, false);
    const sub = findComponent(result.updatedSpec, 'check-subscription');
    assert.equal(sub.role, 'info');
    // 任意の欄なので、望まない人は申し込まず、望む人は自分で選ぶ
    const map = simulateReviewers(result.updatedSpec, getReviewersByCohort('thrifty'), {}, loadBounds('demo-final-fixed-bounds.json'));
    assert.deepEqual([...map.values()].map((r) => r.outcome), ['achieved', 'achieved', 'achieved']);
    assert.equal(map.get('hina')!.finalState?.subscription, false);
    assert.equal(map.get('ren')!.finalState?.subscription, true);

    // 元の画面ですでに必須だった欄は、そのまま残す
    const alreadyRequired = structuredClone(result.updatedSpec);
    findComponent(alreadyRequired, 'check-subscription').role = 'required_input';
    const kept = relaxUnrequestedRequiredChoices(alreadyRequired, alreadyRequired, '文字を大きくして');
    assert.equal(findComponent(kept, 'check-subscription').role, 'required_input');
    // 文字の入力欄のような、真偽ではない必須の欄には触れない
    const withText = structuredClone(spec);
    withText.state.push({ key: 'memo', label: 'メモ', type: 'string', initialValue: '' });
    withText.components.push({
      id: 'input-memo', type: 'input', role: 'required_input', label: 'メモ', stateKey: 'memo',
      style: { size: 'md', variant: 'default' }, actions: [],
    });
    assert.equal(
      findComponent(relaxUnrequestedRequiredChoices(withText, spec, 'メモ欄を足して'), 'input-memo').role,
      'required_input',
    );
  });

  it('「定期便を必須にして」という指示では、必須のまま残る', () => {
    const { spec, requiredPatches } = demoAfterRemoval();
    for (const instruction of ['定期便を必須にして', '定期便に必ずチェックしないと注文できないようにして']) {
      const result = applyPatchesToSpecWithValidation(spec, '定期便を必須にしました', requiredPatches, instruction);
      assert.equal(findComponent(result.updatedSpec, 'check-subscription').role, 'required_input', instruction);
    }
    // 必須にすると、望まない人も申し込んだ扱いになる。悪い判断の結果もそのまま体験できる
    const forced = applyPatchesToSpecWithValidation(spec, 'x', requiredPatches, '定期便を必須にして').updatedSpec;
    const map = simulateReviewers(forced, [persona('hina')], {}, loadBounds('demo-final-fixed-bounds.json'));
    assert.notEqual(map.get('hina')!.finalState?.subscription, false);
    assert.ok(map.get('hina')!.actionTrace.some((t) => t.includes('定期便') && t.includes('入力した')));
  });

  it('指のずれが±14pxのお客さんでも、説明文の行まで押せるチェック欄なら押せる', () => {
    const { spec, requiredPatches } = demoAfterRemoval();
    // 必須の欄は押して埋める必要がある。学習者が必須を求めた画面で、押しやすさだけを比べる
    const forced = applyPatchesToSpecWithValidation(spec, 'x', requiredPatches, '定期便を必須にして').updatedSpec;
    const ren = persona('ren');
    assert.equal(ren.constraints.jitterPx, 14);
    // ブラウザで測った、説明文まで含めて押しやすい高さにした行
    const rowBounds = loadBounds('demo-final-fixed-bounds.json');
    const row = rowBounds.find((b) => b.id === 'check-subscription')!;
    assert.ok(row.bottom - row.top >= 28, `行の高さ ${row.bottom - row.top}px`);
    assert.equal(simulatePersonaComputerUse(forced, ren, rowBounds).outcome, 'achieved');
    // 文字の行の高さしかない欄（行を広げる前の14px）では、押し損ねて離脱する
    const textOnly = rowBounds.map((b) => (b.id === 'check-subscription' ? { ...b, bottom: b.top + 14 } : b));
    const missed = simulatePersonaComputerUse(forced, ren, textOnly);
    assert.equal(missed.outcome, 'abandoned');
    assert.ok(missed.actionTrace.some((t) => t.includes('指がズレて外れた')));
  });

  it('既定の問題の罠の定期便は、小さい字の行のまま罠として判定される', () => {
    // 最初からチェックが入った小さい字の定期便は、押せる行も小さいまま
    const salesPush = buildScenarioSpec(ACTIVE_THEME, 'sales_push').spec;
    assert.equal(findComponent(salesPush, 'check-subscription').style.size, 'xs');
    const demo = buildScenarioSpec(ACTIVE_THEME, 'demo').spec;
    const order = findComponent(demo, 'btn-order');
    const fixed = applyPatchesToSpecWithValidation(demo, 'a', [
      { op: 'replace_component', component: { ...order, actions: [{ type: 'complete_task', message: 'ご購入ありがとうございます' }] } },
    ]).updatedSpec;
    const bounds = loadBounds('demo-step1-fixed-bounds.json');
    const trapRow = bounds.find((b) => b.id === 'check-subscription')!;
    assert.ok(trapRow.bottom - trapRow.top < 20, `罠の行の高さ ${trapRow.bottom - trapRow.top}px`);
    const map = simulateReviewers(fixed, ALL_REVIEWER_PERSONAS, {}, bounds);
    // 細かい字を読まない人は気づかずに後悔し、読める人は気づいて外すが不満が残る
    assert.equal(map.get('hina')!.outcome, 'regretted');
    assert.equal(map.get('misaki')!.outcome, 'regretted');
    assert.equal(map.get('shota')!.outcome, 'bothered');
    assert.ok(map.get('shota')!.actionTrace.some((t) => t.includes('定期便') && t.includes('気づいて直した')));
  });

  it('つまずき部品ID（frictionComponentIds）の特定と、2画面確認でのステップ表示除外・小さいボタン連打離脱の正確な理由判定', () => {
    // 1. 広告が閉じられない kazuko は popup-coupon、注文ボタンまで進んで止まる shota は btn-order が frictionComponentIds に入る
    const demo = buildScenarioSpec(ACTIVE_THEME, 'demo').spec;
    const kazukoDemo = simulatePersonaComputerUse(demo, persona('kazuko'));
    assert.equal(kazukoDemo.outcome, 'abandoned');
    assert.deepEqual(kazukoDemo.frictionComponentIds, ['popup-coupon']);
    const shotaDemo = simulatePersonaComputerUse(demo, persona('shota'));
    assert.equal(shotaDemo.outcome, 'abandoned');
    assert.deepEqual(shotaDemo.frictionComponentIds, ['btn-order']);

    // 2. 2画面の確認ステップにしたとき、progress_steps が size: sm でも確認項目（summary）が md なら kazuko は不満にならず achieved になる
    const twoScreenSpec: AppSpec = {
      ...cleanSpec(),
      state: [
        ...cleanSpec().state,
        { key: 'step', label: '画面ステップ', type: 'string', initialValue: 'input' },
      ],
      components: [
        {
          id: 'progress',
          type: 'progress_steps',
          role: 'info',
          label: 'ご注文手続き',
          options: ['1. 選択', '2. 確認'],
          style: { size: 'sm', variant: 'muted' },
          actions: [],
        },
        ...cleanSpec().components.map((c) =>
          c.id === 'btn-order'
            ? {
                ...c,
                label: '確認画面へ進む',
                visibleWhen: { stateKey: 'step', operator: 'eq' as const, value: 'input' },
                actions: [{ type: 'set_state' as const, targetKey: 'step', value: 'confirm' }],
              }
            : { ...c, visibleWhen: { stateKey: 'step', operator: 'eq' as const, value: 'input' } },
        ),
        {
          id: 'confirm-detail',
          type: 'summary',
          role: 'info',
          label: 'お届け内容の確認',
          rows: [{ label: 'お届け先', value: '{deliveryAddress}' }],
          visibleWhen: { stateKey: 'step', operator: 'eq', value: 'confirm' },
          style: { size: 'md', variant: 'default', fullWidth: true },
          actions: [],
        },
        {
          id: 'confirm-summary',
          type: 'summary',
          role: 'total',
          label: 'ご注文内容の確認',
          rows: [{ label: 'お支払い合計', value: '{total}円' }],
          visibleWhen: { stateKey: 'step', operator: 'eq', value: 'confirm' },
          style: { size: 'md', variant: 'default', fullWidth: true },
          actions: [],
        },
        {
          id: 'btn-final-order',
          type: 'button',
          role: 'primary_action',
          label: '注文を確定する',
          visibleWhen: { stateKey: 'step', operator: 'eq', value: 'confirm' },
          style: { size: 'lg', variant: 'primary', fullWidth: true },
          actions: [{ type: 'complete_task', message: 'ご注文ありがとうございます' }],
        },
      ],
    };
    const kazukoConfirmOk = simulatePersonaComputerUse(twoScreenSpec, persona('kazuko'));
    assert.equal(kazukoConfirmOk.outcome, 'achieved');
    assert.deepEqual(kazukoConfirmOk.frictionComponentIds, []);

    // もし確認画面の案内項目（confirm-detail）自体が size: sm で読めない場合は、frictionComponentIds に confirm-detail が入る
    const twoScreenTinyDetail: AppSpec = {
      ...twoScreenSpec,
      components: twoScreenSpec.components.map((c) =>
        c.id === 'confirm-detail' ? { ...c, style: { ...c.style, size: 'sm' } } : c,
      ),
    };
    const kazukoConfirmSmall = simulatePersonaComputerUse(twoScreenTinyDetail, persona('kazuko'));
    assert.equal(kazukoConfirmSmall.outcome, 'bothered');
    assert.equal(kazukoConfirmSmall.frictionPoint, '確認画面の文字が小さかった');
    assert.deepEqual(kazukoConfirmSmall.frictionComponentIds, ['confirm-detail']);

    // 3. 小さいボタンを指のズレで外し続けて手数上限を超えたときは「手順が多すぎる」ではなく「ボタンや選択欄が小さくて押せない」になる
    const tinyOrderBtnSpec: AppSpec = {
      ...cleanSpec(),
      components: cleanSpec()
        .components.filter((c) => c.id !== 'btn-clear')
        .map((c) =>
          c.id === 'btn-order'
            ? {
                ...c,
                style: { ...c.style, size: 'sm', fullWidth: false, sticky: false, rotateDeg: 10 },
              }
            : c,
        ),
    };
    const takumiMissed = simulatePersonaComputerUse(tinyOrderBtnSpec, persona('takumi'));
    assert.equal(takumiMissed.outcome, 'abandoned');
    assert.equal(takumiMissed.frictionPoint, 'ボタンや選択欄が小さくて押せない');
    assert.deepEqual(takumiMissed.frictionComponentIds, ['btn-order']);

    // 4. おとりボタンで手数を消費して注文ボタン到達時に手数切れになった場合や、確認画面なしで離脱した場合も、途中の原因部品が frictionComponentIds に含まれる
    const salesPush = buildScenarioSpec(ACTIVE_THEME, 'sales_push').spec;
    const salesPushFixedOrderAndPopup: AppSpec = {
      ...salesPush,
      components: salesPush.components
        .filter((c) => c.id !== 'popup-coupon')
        .map((c) =>
          c.id === 'btn-order'
            ? { ...c, actions: [{ type: 'complete_task', message: '完了' }] }
            : c,
        ),
    };
    const kentaAfterPopupFix = simulatePersonaComputerUse(salesPushFixedOrderAndPopup, persona('kenta'));
    assert.equal(kentaAfterPopupFix.outcome, 'abandoned');
    assert.deepEqual(kentaAfterPopupFix.frictionComponentIds, ['btn-buy-now']);

    const harueAfterPopupFix = simulatePersonaComputerUse(salesPushFixedOrderAndPopup, persona('harue'));
    assert.equal(harueAfterPopupFix.outcome, 'abandoned');
    assert.deepEqual(harueAfterPopupFix.frictionComponentIds, ['btn-order']);

    // 5. 画面上に存在する部品の手直し・削除には赤ペン囲みを必須とし、削除済み部品の復活だけ赤ペンなしでも許可する
    assert.equal(canApplyWithoutRedPen(salesPush, '500円OFFの広告ポップアップを消してください'), false);
    assert.equal(canApplyWithoutRedPen(salesPush, '注文を確定するボタンを押したら注文完了するようにして'), false);
    assert.equal(canApplyWithoutRedPen(salesPush, '定期便のチェックを最初は外しておいて'), false);
    assert.equal(canApplyWithoutRedPen(salesPush, '売り切れカウントダウンの表示を消してください'), false);
    assert.equal(canApplyWithoutRedPen(salesPush, '今すぐ購入ボタンを消して'), false);
    assert.equal(
      canApplyWithoutRedPen(salesPushFixedOrderAndPopup, 'さっき消したクーポンをその場で使えるように戻して'),
      true,
    );

    // 6. 部品の言及表示では生のラベル文言ではなく機能名を返す
    assert.equal(
      describeComponentFunctionName(findComponent(salesPush, 'popup-coupon')),
      'クーポン広告のポップアップ',
    );
    assert.equal(
      describeComponentFunctionName(findComponent(salesPush, 'btn-order')),
      '注文確定ボタン',
    );
    assert.equal(
      describeComponentFunctionName(findComponent(salesPush, 'btn-buy-now')),
      '今すぐ購入ボタン',
    );
    assert.equal(
      describeComponentFunctionName(findComponent(salesPush, 'check-subscription')),
      '定期便の選択欄',
    );
    assert.equal(
      describeComponentFunctionName(findComponent(salesPush, 'product-card')),
      '商品と数量の選択欄',
    );
    assert.equal(
      describeComponentFunctionName(findComponent(salesPush, 'banner-countdown')),
      '売り切れカウントダウンの表示',
    );

    // 7. セキュリティ強化：スラッシュや引用符直後のイベント属性・SVG/MathML危険タグ・position:fixed・プロトタイプ汚染キーを拒否する
    const sanitizedSlashEvent = sanitizeCustomHtml(
      '<img/src="x"/onerror="alert(1)"><div title="a"onclick="alert(2)">安全</div><svg><foreignObject><script>alert(3)</script></foreignObject></svg>',
    );
    assert.ok(!sanitizedSlashEvent.includes('onerror'));
    assert.ok(!sanitizedSlashEvent.includes('onclick'));
    assert.ok(!sanitizedSlashEvent.toLowerCase().includes('foreignobject'));

    const sanitizedFixedCss = sanitizeCustomCss('.box { position: fixed; z-index: 9999; }');
    assert.ok(!/position\s*:\s*fixed/i.test(sanitizedFixedCss));
    assert.ok(sanitizedFixedCss.includes('position: relative'));

    const pollutedStateKeySpec = structuredClone(salesPush);
    pollutedStateKeySpec.components[0].stateKey = '__proto__';
    assert.equal(isValidAppSpec(pollutedStateKeySpec), false);

    const pollutedVisibleWhenSpec = structuredClone(salesPush);
    pollutedVisibleWhenSpec.components[0].visibleWhen = {
      stateKey: 'constructor',
      operator: 'eq',
      value: true,
    };
    assert.equal(isValidAppSpec(pollutedVisibleWhenSpec), false);

    const pollutedActionTargetSpec = structuredClone(salesPush);
    pollutedActionTargetSpec.components[0].actions = [
      { type: 'set_state', targetKey: 'prototype', value: 'x' },
    ];
    assert.equal(isValidAppSpec(pollutedActionTargetSpec), false);

    // 部品IDのプロトタイプ汚染キー・壊れた rows・深すぎる計算式・全角メールアドレス・CSS image-set を拒否する
    const pollutedCompIdSpec = structuredClone(salesPush);
    pollutedCompIdSpec.components[0].id = 'constructor';
    assert.equal(isValidAppSpec(pollutedCompIdSpec), false);
    assert.deepEqual(sanitizeComponentIds(['btn-order', '__proto__', 'constructor', 'toString']), ['btn-order']);

    const brokenRowsSpec = structuredClone(salesPush);
    (brokenRowsSpec.components[0] as unknown as { rows: unknown }).rows = [null];
    assert.equal(isValidAppSpec(brokenRowsSpec), false);

    assert.throws(() => evaluateExpression('('.repeat(60) + '1' + ')'.repeat(60), {}), /深すぎます/);
    assert.equal(findPersonalInfoByRules('連絡先は ｔｅｓｔ＠ｅｘａｍｐｌｅ．ｃｏｍ です'), true);
    assert.ok(!sanitizeCustomCss('.box { background-image: image-set("https://evil.example/a.png" 1x); }').includes('evil.example'));

    // 入れ子タグ偽装・数値文字参照による javascript: スキーム・部品ID偽装属性・外部画像URLも除去する
    const nestedSanitized = sanitizeCustomHtml(
      '<scr<script>alert(1)</script>ipt>evil()</script><a href="&#106;avascript:alert(1)" data-component-id="btn-order" popover>リンク</a><img src="https://evil.example/pixel.png">',
    );
    assert.ok(!nestedSanitized.toLowerCase().includes('<script'));
    assert.ok(!nestedSanitized.toLowerCase().includes('javascript'));
    assert.ok(!nestedSanitized.includes('data-component-id'));
    assert.ok(!nestedSanitized.includes('popover'));
    assert.ok(!nestedSanitized.includes('evil.example'));

    const oversizedCaptionSpec = structuredClone(salesPush);
    oversizedCaptionSpec.components[0].caption = 'あ'.repeat(600);
    assert.equal(isValidAppSpec(oversizedCaptionSpec), false);
  });

  it('満足が増えなかったときの一言は、未達成のお客さんが気にする部品が前と変わったときだけ出す', () => {
    const activeReviewerIds = ['r1', 'r2', 'r3'];
    const basePrev = [
      { reviewerId: 'r1', outcome: 'achieved' as const, frictionComponentIds: [] },
      { reviewerId: 'r2', outcome: 'abandoned' as const, frictionComponentIds: ['popup-coupon'] },
      { reviewerId: 'r3', outcome: 'bothered' as const, frictionComponentIds: ['popup-coupon'] },
    ];

    // 1回目の反応確認（前回がない）では出さない
    assert.equal(
      shouldShowNextFrictionNotice({
        changedSinceLastReview: true,
        previousResults: undefined,
        currentResults: basePrev,
        activeReviewerIds,
      }),
      false,
    );

    // 広告を直して、隠れていた定期便の罠に変わったときは出す
    assert.equal(
      shouldShowNextFrictionNotice({
        changedSinceLastReview: true,
        previousResults: basePrev,
        currentResults: [
          { reviewerId: 'r1', outcome: 'achieved', frictionComponentIds: [] },
          { reviewerId: 'r2', outcome: 'regretted', frictionComponentIds: ['check-subscription'] },
          { reviewerId: 'r3', outcome: 'bothered', frictionComponentIds: ['check-subscription'] },
        ],
        activeReviewerIds,
      }),
      true,
    );

    // 同じ部品でまだつまずいているときは出さない
    assert.equal(
      shouldShowNextFrictionNotice({
        changedSinceLastReview: true,
        previousResults: basePrev,
        currentResults: [
          { reviewerId: 'r1', outcome: 'achieved', frictionComponentIds: [] },
          { reviewerId: 'r2', outcome: 'abandoned', frictionComponentIds: ['popup-coupon'] },
          { reviewerId: 'r3', outcome: 'bothered', frictionComponentIds: ['popup-coupon'] },
        ],
        activeReviewerIds,
      }),
      false,
    );

    // 必要な部品を消して指摘部品が空になったときは出さない
    assert.equal(
      shouldShowNextFrictionNotice({
        changedSinceLastReview: true,
        previousResults: basePrev,
        currentResults: [
          { reviewerId: 'r1', outcome: 'achieved', frictionComponentIds: [] },
          { reviewerId: 'r2', outcome: 'bothered', frictionComponentIds: [] },
          { reviewerId: 'r3', outcome: 'bothered', frictionComponentIds: [] },
        ],
        activeReviewerIds,
      }),
      false,
    );

    // 満足が減った改悪のときは出さない
    assert.equal(
      shouldShowNextFrictionNotice({
        changedSinceLastReview: true,
        previousResults: basePrev,
        currentResults: [
          { reviewerId: 'r1', outcome: 'bothered', frictionComponentIds: ['btn-order'] },
          { reviewerId: 'r2', outcome: 'regretted', frictionComponentIds: ['check-subscription'] },
          { reviewerId: 'r3', outcome: 'bothered', frictionComponentIds: ['check-subscription'] },
        ],
        activeReviewerIds,
      }),
      false,
    );

    // 前の反応確認から手直ししていないときは出さない
    assert.equal(
      shouldShowNextFrictionNotice({
        changedSinceLastReview: false,
        previousResults: basePrev,
        currentResults: [
          { reviewerId: 'r1', outcome: 'achieved', frictionComponentIds: [] },
          { reviewerId: 'r2', outcome: 'regretted', frictionComponentIds: ['check-subscription'] },
          { reviewerId: 'r3', outcome: 'bothered', frictionComponentIds: ['check-subscription'] },
        ],
        activeReviewerIds,
      }),
      false,
    );
  });
});

import assert from 'node:assert/strict';
import { it } from 'node:test';
import { ACTIVE_THEME, getReviewersByCohort } from '../src/shared/initialTheme.ts';
import { simulatePersonaComputerUse } from '../src/shared/sandboxValidator.ts';
import { evaluateAppWithPersonas } from '../src/server/gemini.ts';

it('独自部品の最終評価に不満が残る場合はクリア台詞と振り返りを生成しない', async () => {
  const originalFetch = globalThis.fetch;
  const originalApiKey = process.env.GEMINI_API_KEY;
  process.env.GEMINI_API_KEY = 'test-key';
  const spec = structuredClone(ACTIVE_THEME.baseSpec);
  spec.components[0].customHtml = '<p>追加の案内</p>';
  const reviewers = getReviewersByCohort('quick');
  assert.ok(reviewers.every((p) => simulatePersonaComputerUse(spec, p).outcome === 'achieved'));
  const generatedSchemas: string[][] = [];
  let dissatisfied = true;

  globalThis.fetch = async (input, init) => {
    const request = new Request(input, init);
    const body = await request.json();
    const properties = body.generationConfig.responseSchema.properties;
    generatedSchemas.push(Object.keys(properties));
    const reply = 'comments' in properties
      ? { comments: reviewers.map((p, index) => ({
          reviewerId: p.id,
          outcome: dissatisfied && index === 0 ? 'bothered' : 'achieved',
          frictionPoint: '追加の案内',
          comment: '案内が気になりました',
        })) }
      : { lines: ['ありがとうございました'], overview: '振り返り', goodPoints: 'よかった点', nextImprovements: '次の改善' };
    return Response.json({ candidates: [{ content: { role: 'model', parts: [{ text: JSON.stringify(reply) }] } }] });
  };

  try {
    const result = await evaluateAppWithPersonas({ currentSpec: spec, cohortId: 'quick' });
    assert.equal(result.results[0].outcome, 'bothered');
    assert.equal(result.owner?.closingLines, undefined);
    assert.equal(result.telemetrySummary?.assistantFeedbackLines, undefined);
    assert.deepEqual(generatedSchemas, [['comments']]);

    dissatisfied = false;
    generatedSchemas.length = 0;
    const cleared = await evaluateAppWithPersonas({ currentSpec: spec, cohortId: 'quick' });
    assert.ok(cleared.results.every((r) => r.outcome === 'achieved'));
    assert.ok(cleared.owner?.closingLines?.length);
    assert.equal(cleared.telemetrySummary?.assistantFeedbackLines?.length, 3);
    assert.equal(generatedSchemas.length, 3);
    assert.deepEqual(generatedSchemas[0], ['comments']);

    // 独自部品がなければ、AIが不満を返してもコード判定を維持する。
    dissatisfied = true;
    delete spec.components[0].customHtml;
    const standard = await evaluateAppWithPersonas({ currentSpec: spec, cohortId: 'quick' });
    assert.ok(standard.results.every((r) => r.outcome === 'achieved'));
    assert.ok(standard.owner?.closingLines?.length);
    assert.equal(standard.telemetrySummary?.assistantFeedbackLines?.length, 3);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalApiKey === undefined) delete process.env.GEMINI_API_KEY;
    else process.env.GEMINI_API_KEY = originalApiKey;
  }
});

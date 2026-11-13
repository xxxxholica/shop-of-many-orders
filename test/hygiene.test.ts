import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  findForbiddenContent,
  validateAuthorEmail,
  validateCommitMessage,
  validateSubjectDescription,
} from '../scripts/check-hygiene.mjs';

describe('check-hygiene', () => {
  it('正しい Conventional Commits を通す', () => {
    const msg = [
      'feat: チェックポイント保存とロールバック機能を追加',
      '',
      '- 改造ステップごとの状態を履歴に記録し、任意の時点へ戻せるようにする',
    ].join('\n');
    assert.deepEqual(validateCommitMessage(msg), []);
  });

  it('Merge / Revert コミットを通す', () => {
    assert.deepEqual(validateCommitMessage("Merge branch 'fix/sandbox-reset'"), []);
    assert.deepEqual(validateCommitMessage('Revert "feat(sandbox): チェックポイントを追加"'), []);
  });

  it('type が無い、または英語のみの要約を弾く', () => {
    const noType = validateCommitMessage('sandbox: チェックポイント機能を追加');
    assert.equal(noType.length, 1);

    const englishOnly = validateCommitMessage('feat(sandbox): add checkpoint rollback');
    assert.equal(englishOnly.length, 1);

    // Dependabot 等の自動生成コミットでは allowEnglish: true で Conventional Commits 形式のみを通す
    assert.deepEqual(validateCommitMessage('chore(deps-dev): bump vite', { allowEnglish: true }), []);
    assert.equal(validateCommitMessage('bump vite', { allowEnglish: true }).length, 1);
  });

  it('1行目と本文の間に空行がない場合を弾く', () => {
    const msg = 'fix(studio): ボタンの配置を修正\n2行目に直接本文を書く';
    const errs = validateCommitMessage(msg);
    assert.ok(errs.some((e) => e.includes('空行')));
  });

  it('1行目の PR 番号と、要約の鉤括弧・丸括弧を弾く(scope の括弧は通す)', () => {
    assert.deepEqual(validateCommitMessage('feat(studio,sandbox): お題アプリの初期状態リセット機能を追加'), []);

    const prNumber = validateCommitMessage('fix(sandbox): 状態復元の不具合を直す (#12)');
    assert.equal(prNumber.length, 1);
    assert.ok(prNumber[0].includes('PR 番号'));

    for (const subject of [
      'feat(studio): 「ダメな目覚まし時計」を追加',
      'feat(studio): 『ダメな目覚まし時計』を追加',
      'feat(studio): ダメな目覚まし時計(お題1)を追加',
      'feat(studio): ダメな目覚まし時計（お題1）を追加',
    ]) {
      assert.equal(validateCommitMessage(subject).length, 1, subject);
    }

    // 本文の括弧は対象外
    assert.deepEqual(validateCommitMessage('fix(studio): ボタンを直す\n\n- 初期化前に生成する(画面崩れ対策)'), []);
  });

  it('PR タイトルの PR 番号と括弧を検出する', () => {
    assert.deepEqual(validateSubjectDescription('ボタンを直す', 'PR タイトル'), []);
    assert.equal(validateSubjectDescription('「ボタン」を直す', 'PR タイトル').length, 1);
  });

  it('Claude-Session やセッション URL および Claude Co-Authored-By を含むコミットメッセージを弾く', () => {
    const trailer = 'feat: 画面を整理\n\nClaude-' + 'Session: https://claude.ai/code/session_01AbCdEf';
    const errs = validateCommitMessage(trailer);
    assert.ok(errs.length >= 1);

    const coAuthor =
      'feat: 画面を整理\n\nCo-Authored-By: ' + 'Claude Opus 5.5 <noreply@' + 'anthropic.com>';
    const coAuthorErrs = validateCommitMessage(coAuthor);
    assert.ok(coAuthorErrs.length >= 1);
  });

  it('Author メールアドレスの .local や個人のフリーメールを弾く', () => {
    assert.deepEqual(validateAuthorEmail('202958124+xxxxholica@users.noreply.github.com'), []);

    assert.equal(validateAuthorEmail('user@MyMacBook-Air.local').length, 1);
    assert.equal(validateAuthorEmail('someone@gmail.com').length, 1);
  });

  it('ファイルや PR 本文内のローカルパス・セッション URL・APIキー・秘密鍵を検出する', () => {
    assert.deepEqual(findForbiddenContent('const path = "./public/assets";', 'src/main.ts'), []);

    const macLiteral = findForbiddenContent('const p = "/U' + 'sers/alice/project";', 'src/main.ts');
    assert.equal(macLiteral.length, 1);

    const winLiteral = findForbiddenContent('cloned to C:\\U' + 'sers\\alice\\Downloads', 'PR 本文');
    assert.equal(winLiteral.length, 1);

    const apiKeyLiteral = findForbiddenContent('const key = "AIza' + 'SyA1234567890abcdefghijklmnopqrstuv";', 'src/config.ts');
    assert.equal(apiKeyLiteral.length, 1);

    const privKeyLiteral = findForbiddenContent('-----BEGIN ' + 'PRIVATE KEY-----', 'service-account.json');
    assert.equal(privKeyLiteral.length, 1);

    const opensshKeyLiteral = findForbiddenContent('-----BEGIN OPENSSH ' + 'PRIVATE KEY-----', 'id_ed25519');
    assert.equal(opensshKeyLiteral.length, 1);

    const linuxLiteral = findForbiddenContent('const p = "/h' + 'ome/alice/project";', 'src/main.ts');
    assert.equal(linuxLiteral.length, 1);

    const oauthLiteral = findForbiddenContent('Bearer ya29.' + 'a0AfH6SMBx1234567890abcdefghij', 'src/auth.ts');
    assert.equal(oauthLiteral.length, 1);

    const saEmailLiteral = findForbiddenContent('bot@my-proj-123.iam.' + 'gserviceaccount.com', 'config.json');
    assert.equal(saEmailLiteral.length, 1);

    const ghPatLiteral = findForbiddenContent('token = "ghp_' + '1234567890abcdefghijklmnopqrstuvwx"', '.env');
    assert.equal(ghPatLiteral.length, 1);
  });
});

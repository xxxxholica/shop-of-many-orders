#!/usr/bin/env node
// コミットメッセージ・Author・追跡ファイル・PR本文に規則違反やプライベート情報・秘密情報の混入がないか検査する。
// 使い方:
//   node scripts/check-hygiene.mjs commit-msg <msg-file>   # .githooks/commit-msg から呼ばれる
//   node scripts/check-hygiene.mjs pre-commit              # .githooks/pre-commit から呼ばれる
//   node scripts/check-hygiene.mjs [ci] [--range <A..B>]   # CI / 手動検査

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const ALLOWED_TYPES = [
  'feat',
  'fix',
  'perf',
  'refactor',
  'docs',
  'test',
  'chore',
  'style',
  'build',
  'ci',
  'revert',
  'security',
  'bench',
  'balance',
];

const CONVENTIONAL_RE = new RegExp(
  `^(${ALLOWED_TYPES.join('|')})(\\([a-zA-Z0-9_., /-]+\\))?!?: (\\S.*)$`,
);
const JAPANESE_CHAR_RE = /[ぁ-んァ-ヶー一-龠々]/;
// 件名の要約(scope の括弧より後)に書かないもの
const PR_NUMBER_RE = /\s*\(#\d+\)/g;
const SUBJECT_BRACKET_RE = /[「」『』()（）]/;

// 検査対象外の拡張子・パス
const BINARY_EXT_RE = /\.(png|jpg|jpeg|gif|webp|ico|mp3|ogg|wav|woff|woff2|ttf|otf|glb|gltf|fbx|bin|mp4|webm)$/i;
const SELF_ALLOWLIST = new Set([
  'scripts/check-hygiene.mjs',
  'test/hygiene.test.ts',
]);

// 禁止パターン(文字列リテラルで自身にヒットしないよう組み立てる)
const FORBIDDEN_PATTERNS = [
  {
    name: 'Claude-Session トレーラー',
    re: new RegExp('Claude-' + 'Session', 'i'),
  },
  {
    name: 'Claude Co-Authored-By トレーラー',
    re: new RegExp('Co-Authored-By:.*(?:' + 'Claude|' + 'anthropic\\.com)', 'i'),
  },
  {
    name: 'Claude Code セッションURL',
    re: new RegExp('claude\\.ai/code/' + 'session_[a-zA-Z0-9]+', 'i'),
  },
  {
    name: 'macOS ローカルホームパス (/Users/<user>)',
    re: /(?:^|[\s"'`(=:/])(\/Users\/[a-zA-Z0-9._-]+)/,
  },
  {
    name: 'Windows ローカルホームパス (C:\\Users\\<user>)',
    re: /(?:^|[\s"'`(=])([A-Za-z]:[\\/]+Users[\\/]+[a-zA-Z0-9._-]+)/,
  },
  {
    name: 'Linux ローカルホームパス (/home/<user>/)',
    re: /(?:^|[\s"'`(=:/])(\/home\/(?!node\b|nonroot\b|runner\b)[a-zA-Z0-9._-]+\/)/,
  },
  {
    name: 'Google API キー (AIza...)',
    re: new RegExp('(AIza' + '[0-9A-Za-z_-]{35})'),
  },
  {
    name: 'Google OAuth アクセストークン (ya29....)',
    re: new RegExp('(ya29\\.' + '[0-9A-Za-z_-]{20,})'),
  },
  {
    name: 'Google OAuth クライアントシークレット (GOCSPX-...)',
    re: new RegExp('(GOCSPX-' + '[0-9A-Za-z_-]{20,})'),
  },
  {
    name: 'GCP サービスアカウントメール (*.iam.gserviceaccount.com)',
    re: new RegExp('([a-z0-9-]+@[a-z0-9-]+\\.iam\\.' + 'gserviceaccount\\.com)', 'i'),
  },
  {
    name: 'GitHub パーソナルアクセストークン (ghp_... / github_pat_...)',
    re: new RegExp('((?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_' + '[A-Za-z0-9_]{30,}))'),
  },
  {
    name: '秘密鍵 (PRIVATE KEY)',
    re: new RegExp('(-----BEGIN (?:[A-Z0-9 ]+ )?' + 'PRIVATE KEY-----)'),
  },
];

const PRIVATE_EMAIL_DOMAIN_RE = /@(gmail\.com|googlemail\.com|yahoo\.co\.jp|yahoo\.com|icloud\.com|me\.com|outlook\.com|hotmail\.com|live\.jp)$/i;
const LOCAL_HOSTNAME_EMAIL_RE = /\.local$/i;

/**
 * コミットメッセージ全文を検証し、エラーメッセージの配列を返す(問題なければ空配列)。
 * @param {string} raw
 * @param {{ allowEnglish?: boolean }} [opts]
 * @returns {string[]}
 */
export function validateCommitMessage(raw, opts = {}) {
  const errors = [];
  // Git が付与するコメント行(# ...)を除去
  const lines = raw
    .replace(/\r\n/g, '\n')
    .split('\n')
    .filter((line) => !line.startsWith('#'));

  while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
    lines.pop();
  }

  if (lines.length === 0) {
    return ['コミットメッセージが空です。'];
  }

  const subject = lines[0].trim();
  const isAutoMergeOrRevert = /^Merge\b/.test(subject) || /^Revert\b/.test(subject);

  if (!isAutoMergeOrRevert) {
    const m = subject.match(CONVENTIONAL_RE);
    if (!m) {
      errors.push(
        `1行目が Conventional Commits 形式ではありません: "${subject}"\n` +
          `  期待する形式: <type>(<scope>): <日本語の要約> (例: feat(sandbox): チェックポイント機能を追加)\n` +
          `  使える type: ${ALLOWED_TYPES.join(', ')}`,
      );
    } else if (!opts.allowEnglish) {
      const desc = m[3];
      if (!JAPANESE_CHAR_RE.test(desc)) {
        errors.push(`1行目の要約には日本語を含めてください: "${subject}"`);
      }
      errors.push(...validateSubjectDescription(desc, '1行目'));
    }

    if (lines.length >= 2 && lines[1].trim() !== '') {
      errors.push('1行目(件名)と本文の間には空行を1行入れてください。');
    }
  }

  const fullText = lines.join('\n');
  for (const { name, re } of FORBIDDEN_PATTERNS) {
    const match = fullText.match(re);
    if (match) {
      errors.push(`コミットメッセージに禁止パターン(${name})が含まれています: "${match[1] ?? match[0]}"`);
    }
  }

  return errors;
}

/**
 * 件名の要約部分に PR 番号「(#123)」や鉤括弧・丸括弧がないか検査する。
 * @param {string} desc `<type>(<scope>): ` より後ろの部分
 * @param {string} label
 * @returns {string[]}
 */
export function validateSubjectDescription(desc, label) {
  const errors = [];
  const withoutPr = desc.replace(PR_NUMBER_RE, '');
  if (withoutPr !== desc) {
    errors.push(`${label}に PR 番号 "(#番号)" を書かないでください: "${desc}"`);
  }
  const bracket = withoutPr.match(SUBJECT_BRACKET_RE);
  if (bracket) {
    errors.push(
      `${label}の要約に括弧 "${bracket[0]}" を使わないでください(「」『』()（） は不可、scope の括弧は可): "${desc}"`,
    );
  }
  return errors;
}

/**
 * Git の Author / Committer メールアドレスを検証する。
 * @param {string} email
 * @param {string} [label]
 * @returns {string[]}
 */
export function validateAuthorEmail(email, label = 'Author') {
  const trimmed = (email ?? '').trim();
  if (!trimmed) {
    return [`${label} のメールアドレスが空です。`];
  }
  if (LOCAL_HOSTNAME_EMAIL_RE.test(trimmed)) {
    return [
      `${label} メールアドレスにローカルホスト名(.local)が使われています: <${trimmed}>\n` +
        `  GitHub の noreply アドレス (<ID>+<user>@users.noreply.github.com) 等を設定してください。`,
    ];
  }
  if (PRIVATE_EMAIL_DOMAIN_RE.test(trimmed)) {
    return [
      `${label} メールアドレスに個人のフリーメールが使われています: <${trimmed}>\n` +
        `  GitHub の noreply アドレス (<ID>+<user>@users.noreply.github.com) またはプロジェクト用アドレスを設定してください。`,
    ];
  }
  return [];
}

/**
 * テキスト内容に禁止パターンが含まれていないか検査する。
 * @param {string} content
 * @param {string} label
 * @returns {string[]}
 */
export function findForbiddenContent(content, label) {
  const errors = [];
  const lines = content.replace(/\r\n/g, '\n').split('\n');
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    for (const { name, re } of FORBIDDEN_PATTERNS) {
      const m = line.match(re);
      if (m) {
        errors.push(`${label}:${i + 1}: 禁止パターン(${name}) "${m[1] ?? m[0]}" が含まれています`);
      }
    }
  }
  return errors;
}

function git(args) {
  return execFileSync('git', args, { encoding: 'utf-8' }).trim();
}

function checkFiles(filePaths) {
  const errors = [];
  for (const relPath of filePaths) {
    if (!relPath || SELF_ALLOWLIST.has(relPath) || BINARY_EXT_RE.test(relPath) || relPath.endsWith('package-lock.json')) {
      continue;
    }
    if (!existsSync(relPath)) continue;
    let content;
    try {
      content = readFileSync(relPath, 'utf-8');
    } catch {
      continue;
    }
    // バイナリ判定(NULバイトを含む場合はスキップ)
    if (content.includes('\0')) continue;
    errors.push(...findForbiddenContent(content, relPath));
  }
  return errors;
}

function checkCommitRange(range) {
  const errors = [];
  let raw;
  try {
    raw = git(['log', range, '--format=%H%x00%an%x00%ae%x00%cn%x00%ce%x00%B%x01']);
  } catch {
    return errors;
  }
  if (!raw) return errors;

  const records = raw
    .split('\x01')
    .map((r) => r.trim())
    .filter(Boolean);

  for (const rec of records) {
    const [sha, authorName, authorEmail, committerName, committerEmail, body] = rec.split('\x00');
    const firstLine = (body ?? '').split('\n')[0].trim();
    // GitHub Actions の pull_request checkout が自動生成する仮マージコミット(refs/pull/*/merge)は除外
    if (/^Merge [0-9a-f]{40} into [0-9a-f]{40}$/i.test(firstLine) && committerEmail === 'noreply@github.com') {
      continue;
    }
    const shortSha = sha.slice(0, 7);
    const isDependabot = authorName === 'dependabot[bot]' || authorEmail.includes('dependabot[bot]');
    for (const err of validateAuthorEmail(authorEmail, `コミット ${shortSha} (${authorName}) の Author`)) {
      errors.push(err);
    }
    for (const err of validateAuthorEmail(committerEmail, `コミット ${shortSha} (${committerName}) の Committer`)) {
      errors.push(err);
    }
    for (const err of validateCommitMessage(body, { allowEnglish: isDependabot })) {
      errors.push(`コミット ${shortSha}: ${err}`);
    }
  }
  return errors;
}

function runCli() {
  const args = process.argv.slice(2);
  const mode = args[0] ?? 'ci';
  const errors = [];

  if (mode === 'commit-msg') {
    const msgFile = args[1];
    if (!msgFile) {
      console.error('使い方: node scripts/check-hygiene.mjs commit-msg <msg-file>');
      process.exit(1);
    }
    const raw = readFileSync(msgFile, 'utf-8');
    errors.push(...validateCommitMessage(raw));
  } else if (mode === 'pre-commit') {
    const userEmail = git(['config', 'user.email']);
    errors.push(...validateAuthorEmail(userEmail, 'git config user.email'));

    const staged = git(['diff', '--cached', '--name-only', '--diff-filter=ACMR'])
      .split('\n')
      .filter(Boolean);
    errors.push(...checkFiles(staged));
  } else {
    // CI / 全体検査モード
    const tracked = git(['ls-files']).split('\n').filter(Boolean);
    errors.push(...checkFiles(tracked));

    // コミット範囲の決定
    const rangeIdx = args.indexOf('--range');
    let range = rangeIdx !== -1 ? args[rangeIdx + 1] : process.env.HYGIENE_COMMIT_RANGE;
    if (!range && process.env.GITHUB_BASE_REF) {
      range = `origin/${process.env.GITHUB_BASE_REF}..HEAD`;
    }
    if (range) {
      errors.push(...checkCommitRange(range));
    }

    // GitHub Actions の PR イベント本文・タイトル検査
    if (process.env.GITHUB_EVENT_PATH && existsSync(process.env.GITHUB_EVENT_PATH)) {
      try {
        const ev = JSON.parse(readFileSync(process.env.GITHUB_EVENT_PATH, 'utf-8'));
        if (ev.pull_request) {
          const prTitle = ev.pull_request.title ?? '';
          const prBody = ev.pull_request.body ?? '';
          const isDependabotPr = ev.pull_request.user?.login === 'dependabot[bot]';
          errors.push(...findForbiddenContent(prTitle, 'PR タイトル'));
          // Squash merge ではこれがコミットの件名になる(Dependabot PR 単体の CI では英語タイトルを許容する)
          if (!isDependabotPr) {
            errors.push(...validateSubjectDescription(prTitle.match(CONVENTIONAL_RE)?.[3] ?? prTitle, 'PR タイトル'));
          }
          errors.push(...findForbiddenContent(prBody, 'PR 本文'));
        }
      } catch {
        // イベント JSON が読めない場合は無視
      }
    }
  }

  if (errors.length > 0) {
    console.error('\n[check-hygiene] 規則違反またはプライベート情報・秘密情報の混入が検出されました:\n');
    for (const err of errors) {
      console.error(`  ✖ ${err}`);
    }
    console.error('');
    process.exit(1);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  runCli();
}

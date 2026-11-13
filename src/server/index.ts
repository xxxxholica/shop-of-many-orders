import { existsSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { handleApiRequest } from './api.ts';
import { describeErrorForLog } from './logSafe.ts';
import { COMMON_SECURITY_HEADERS } from './rateLimit.ts';

const PORT = Number(process.env.PORT || 8080);
const DIST_DIR = resolve(process.cwd(), 'dist');

const MIME_TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

const SECURITY_HEADERS: Record<string, string> = {
  ...COMMON_SECURITY_HEADERS,
  'X-Frame-Options': 'SAMEORIGIN',
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Content-Security-Policy':
    "default-src 'self'; img-src 'self' data: blob:; media-src 'self' data: blob:; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com data:; connect-src 'self'; frame-ancestors 'self'; object-src 'none'; base-uri 'self'; form-action 'self'",
};

const ROBOTS_TXT = 'User-agent: *\nDisallow: /\n';

const server = createServer(async (req, res) => {
  try {
    const handled = await handleApiRequest(req, res);
    if (handled) return;

    // Cloud Run 等での静的ファイル配信 (dist/)
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405, SECURITY_HEADERS);
      res.end('Method Not Allowed');
      return;
    }
    const rawUrlPath = (req.url ?? '/').split('?')[0];
    if (rawUrlPath === '/robots.txt') {
      res.writeHead(200, { ...SECURITY_HEADERS, 'Content-Type': 'text/plain; charset=utf-8' });
      res.end(ROBOTS_TXT);
      return;
    }
    let urlPath = '/';
    try {
      urlPath = decodeURIComponent(rawUrlPath);
    } catch {
      res.writeHead(400, { ...SECURITY_HEADERS, 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Bad Request');
      return;
    }
    if (urlPath.includes('\0')) {
      res.writeHead(400, { ...SECURITY_HEADERS, 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Bad Request');
      return;
    }
    const safeRelPath = normalize(urlPath).replace(/^(\.\.(\/|\\|$))+/, '');
    if (safeRelPath.split(/[/\\]/).some((seg) => seg.startsWith('.') && seg !== '.')) {
      res.writeHead(403, SECURITY_HEADERS);
      res.end('Forbidden');
      return;
    }
    let targetPath = resolve(DIST_DIR, safeRelPath === '/' ? 'index.html' : `.${safeRelPath}`);

    if (targetPath !== DIST_DIR && !targetPath.startsWith(DIST_DIR + sep)) {
      res.writeHead(403, SECURITY_HEADERS);
      res.end('Forbidden');
      return;
    }

    if (!existsSync(targetPath) || statSync(targetPath).isDirectory()) {
      targetPath = join(DIST_DIR, 'index.html');
    }

    if (!existsSync(targetPath)) {
      res.writeHead(404, { ...SECURITY_HEADERS, 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('Not Found (build dist/ first)');
      return;
    }

    const ext = extname(targetPath).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';
    const content = readFileSync(targetPath);
    res.writeHead(200, { ...SECURITY_HEADERS, 'Content-Type': contentType });
    res.end(content);
  } catch (err) {
    console.error('[server] リクエスト処理に失敗しました:', describeErrorForLog(err));
    if (!res.headersSent) {
      res.writeHead(500, { ...SECURITY_HEADERS, 'Content-Type': 'text/plain; charset=utf-8' });
    }
    res.end('Internal Server Error');
  }
});

server.requestTimeout = 30_000;
server.headersTimeout = 15_000;

server.listen(PORT, () => {
  console.log(`[shop-of-many-orders] Server listening on http://localhost:${PORT}`);
});

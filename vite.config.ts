import { defineConfig } from 'vite';
import { handleApiRequest } from './src/server/api.ts';

// 開発サーバーの Host ヘッダー検証:
// 既定では localhost / ローカル IP のみ許可し、モバイル実機検証などでトンネルを使う場合のみ
// 環境変数 VITE_ALLOWED_HOSTS (カンマ区切り) で明示的に許可ホストを追加する。
const extraAllowedHosts = process.env.VITE_ALLOWED_HOSTS
  ? process.env.VITE_ALLOWED_HOSTS.split(',').map((h) => h.trim()).filter(Boolean)
  : undefined;

export default defineConfig({
  plugins: [
    {
      name: 'shop-of-many-orders-api',
      configureServer(server) {
        server.middlewares.use(async (req, res, next) => {
          const handled = await handleApiRequest(req, res);
          if (!handled) {
            next();
          }
        });
      },
    },
  ],
  server: {
    ...(extraAllowedHosts ? { allowedHosts: extraAllowedHosts } : {}),
  },
  build: {
    rolldownOptions: {
      // 書き写したアイコンの著作権表示を、配信するJSにも残す。ライセンスの条件のため
      output: { comments: { legal: true } },
    },
  },
});

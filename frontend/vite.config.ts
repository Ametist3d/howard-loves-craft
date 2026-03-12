import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  // Load .env from project root (one level up from frontend/)
  const env = loadEnv(mode, path.resolve(__dirname, '..'), '');

  const fePort = parseInt(env.VITE_FE_PORT || '3000', 10);
  const bePort = parseInt(env.VITE_BE_PORT || '8000', 10);
  const beHost = env.BE_HOST || '127.0.0.1';

  return {
    server: {
      port: fePort,
      host: '0.0.0.0',   // bind to all interfaces so LAN clients can reach it
      proxy: {
        '/api': {
          target: `http://${beHost}:${bePort}`,
          changeOrigin: true,
          secure: false,
        },
      },
    },
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
  };
});
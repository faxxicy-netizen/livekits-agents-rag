import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'node:path';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  build: { rollupOptions: { output: { manualChunks: { livekit: ['livekit-client'] } } } },
  server: { proxy: { '/api': 'http://127.0.0.1:8000' } },
});

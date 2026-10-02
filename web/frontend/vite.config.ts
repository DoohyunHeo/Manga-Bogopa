import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 빌드 결과는 web/dist — 파이썬 서버(web/server.py)가 그대로 내준다. 개발할 때는 `npm run dev`로
// 이 화면만 따로 띄우고 /api는 파이썬 서버(7860)로 넘긴다.
export default defineConfig({
  plugins: [react()],
  base: './',
  build: { outDir: '../dist', emptyOutDir: true, assetsInlineLimit: 0 },
  server: { proxy: { '/api': { target: 'http://127.0.0.1:7860', changeOrigin: true } } },
})

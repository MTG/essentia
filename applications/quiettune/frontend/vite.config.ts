import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: { rollupOptions: { output: { manualChunks: { charts: ['recharts'], waveform: ['wavesurfer.js'] } } } },
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8000' } },
})

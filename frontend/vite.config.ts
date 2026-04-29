import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// In GitHub Codespaces, the API runs on a different forwarded URL.
// Set VITE_API_TARGET in your shell to override (e.g. http://localhost:8000).
const apiTarget = process.env.VITE_API_TARGET ?? 'http://localhost:8000'
const wsTarget = process.env.VITE_WS_TARGET ?? 'ws://localhost:8000'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    // Allow Codespaces forwarded hostnames
    allowedHosts: 'all',
    proxy: {
      '/api': {
        target: apiTarget,
        changeOrigin: true,
      },
      '/ws': {
        target: wsTarget,
        ws: true,
        changeOrigin: true,
      },
    },
  },
})

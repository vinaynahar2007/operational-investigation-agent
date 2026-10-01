import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies /api to the Python backend so the frontend never
// needs to know the backend address. All investigation logic stays server-side.
export default defineConfig({
  plugins: [react()],
  server: {
    // Bind all interfaces so the demo is reachable via localhost and
    // 127.0.0.1 on a local machine, regardless of IPv4/IPv6 preference.
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
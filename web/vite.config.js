import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In dev, Vite serves the React app and forwards /api to FastAPI on :8000.
// In production FastAPI serves the built files itself, so there is no proxy.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
// Deploy se pehle 'credit-risk-project' ko apne actual repo name se replace kar dein
export default defineConfig({
  plugins: [react()],
  base: '/credit-risk-project/',
})

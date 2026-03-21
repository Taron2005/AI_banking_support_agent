import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: "127.0.0.1",
    open: true,
    // Dev: use relative API_BASE (see App.jsx) so requests hit this app’s FastAPI on :8000.
    proxy: {
      "^/(health|ready|chat|api)(/|$)": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});

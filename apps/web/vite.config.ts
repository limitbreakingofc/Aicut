import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": "http://127.0.0.1:43210",
      "/mcp": "http://127.0.0.1:43210",
      "/openapi.json": "http://127.0.0.1:43210",
      "/health": "http://127.0.0.1:43210",
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});

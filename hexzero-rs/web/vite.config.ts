import { defineConfig } from "vite";
import { resolve } from "path";
import { existsSync, readFileSync, mkdirSync, copyFileSync, readdirSync } from "fs";

export default defineConfig({
  // Serve wasm from the hex-wasm pkg directory during dev.
  server: {
    fs: { allow: [".", "../crates/hex-wasm/pkg"] },
  },
  // Workers need to be bundled as ES modules for WASM import.
  worker: {
    format: "es",
  },
  // onnxruntime-web loads .wasm files relative to its module URL;
  // Vite's dep pre-bundling breaks this by moving JS to .vite/deps/
  // without the .wasm files.  Exclude it so it's served from node_modules/.
  optimizeDeps: {
    exclude: ["onnxruntime-web"],
  },
  plugins: [
    {
      // Dev: serve onnxruntime-web's runtime files from /ort/ via middleware.
      // Build: copy them into dist/ort/ so the production bundle can find them.
      name: "ort-wasm-serve",
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const url = req.url?.split("?")[0];
          if (url?.startsWith("/ort/ort-wasm-")) {
            const filename = url.slice(5); // strip "/ort/"
            const filepath = resolve(
              "node_modules/onnxruntime-web/dist",
              filename,
            );
            if (existsSync(filepath)) {
              const ct = filename.endsWith(".wasm")
                ? "application/wasm"
                : "application/javascript";
              res.setHeader("Content-Type", ct);
              res.setHeader("Cache-Control", "public, max-age=86400");
              res.end(readFileSync(filepath));
              return;
            }
          }
          next();
        });
      },
      writeBundle(options) {
        // Copy ORT runtime files (ort-wasm-*) into dist/ort/
        const outDir = options.dir ?? resolve("dist");
        const ortDist = resolve("node_modules/onnxruntime-web/dist");
        const ortOut = resolve(outDir, "ort");
        mkdirSync(ortOut, { recursive: true });
        for (const file of readdirSync(ortDist)) {
          if (file.startsWith("ort-wasm-") && (file.endsWith(".wasm") || file.endsWith(".mjs"))) {
            copyFileSync(resolve(ortDist, file), resolve(ortOut, file));
          }
        }
      },
    },
  ],
});

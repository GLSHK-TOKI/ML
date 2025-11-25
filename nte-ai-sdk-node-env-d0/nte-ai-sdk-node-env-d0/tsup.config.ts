import { defineConfig } from "tsup";

export default defineConfig({
	entry: ["src/index.ts"],
	splitting: true,
	skipNodeModulesBundle: true,
	sourcemap: false,
	minify: false,
	format: ["esm"],
	clean: true,
	dts: true,
	outDir: "dist",
	treeshake: true,
	watch: false,
	platform: "neutral",
	target: "es2022",
	replaceNodeEnv: true,
});
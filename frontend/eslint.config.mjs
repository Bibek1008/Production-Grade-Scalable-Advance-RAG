import nextPlugin from "@next/eslint-plugin-next";
import tseslint from "typescript-eslint";

export default [
  ...tseslint.configs.recommended,
  nextPlugin.configs["core-web-vitals"],
  {
    ignores: [".next/**", "node_modules/**", "next-env.d.ts"],
  },
];

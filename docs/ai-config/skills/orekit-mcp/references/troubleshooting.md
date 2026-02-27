# Orekit MCP: Troubleshooting

## MCP tools unavailable

If `orekit_docs_*` tools are not callable in the session:

- State the limitation explicitly (tools not exposed, server not healthy).
- If MCP resources are available, use `orekit://*` resources as a fallback for docs lookup.
- Fall back to best-effort local knowledge and mark signature uncertainty.

## Server configuration (local Codex)

Canonical config snippet lives at:

- `docs/ai-config/codex-config-snippets/orekit-docs-mcp.toml`

Key points:

- Prefer direct interpreter launch from `mcp-tools-env` (most reliable).
- If using `conda run`, it must use `--no-capture-output` for stdio MCP.
- Ensure `OREKIT_CACHE_ROOT` points to the local Orekit cache directory.
- Ensure `OREKIT_JAVADOC_VERSION` matches the indexed Javadocs version.

## Cache / version mismatch symptoms

Common signs:

- Search returns empty or unrelated results.
- Class docs exist but member lookups fail unexpectedly.

Actions:

- Verify server env vars: `OREKIT_CACHE_ROOT`, `OREKIT_JAVADOC_VERSION`.
- Rebuild/reindex Javadocs cache if the runtime supports it (see MCP runtime
  repo/tooling).

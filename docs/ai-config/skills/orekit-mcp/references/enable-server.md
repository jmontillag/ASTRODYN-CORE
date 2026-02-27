# Orekit MCP: Enable The Server

If you are configuring the local `orekit_docs` MCP server for Codex, copy one
variant from:

- `docs/ai-config/codex-config-snippets/orekit-docs-mcp.toml`

into:

- `~/.codex/config.toml`

Rules:

- Do not enable both variants under the same server name.
- Prefer the direct `mcp-tools-env` interpreter variant.
- If using `conda run`, include `--no-capture-output`.


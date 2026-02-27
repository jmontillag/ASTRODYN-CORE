# AI Config Canonical Snippets

This folder stores tracked copies/templates of local Codex configuration content
that lives outside the repo (for example `~/.codex/skills/...`) or is reused
across repositories (`AGENTS.md` snippets).

Current contents:

- `skills/orekit-mcp/`
  - canonical copy of the global `orekit-mcp` skill installed in
    `~/.codex/skills/orekit-mcp`
- `skills/astrodyn-core-consumer/`
  - canonical skill template for dependent repositories that should use
    `astrodyn_core` facade APIs for propagation/state/mission workflows
- `agents-snippets/orekit-mcp-usage.md`
  - reusable `AGENTS.md` section for repos that use Orekit + the `orekit_docs`
    MCP server
- `agents-snippets/astrodyn-core-consumer.md`
  - reusable `AGENTS.md` section for repositories that consume
    `/home/astror/Projects/ASTRODYN-CORE` as an editable dependency
- `codex-config-snippets/orekit-docs-mcp.toml`
  - reusable `~/.codex/config.toml` MCP server snippets (direct Python and
    `conda run --no-capture-output` variants)
  - points to a global home-directory runtime install at
    `~/.codex/mcp/orekit-docs/`
- `environments/mcp-tools-env.yml`
  - dedicated Conda environment spec for the Orekit docs MCP runtime
- `data-manifests/orekit-cache-manifest.md`
  - reproducibility manifest for the Orekit docs/tutorials cache in
    `~/.cache/orekit`
- `../scripts/orekit_mcp/export_cache_bundle.sh`
- `../scripts/orekit_mcp/import_cache_bundle.sh`
  - portability helpers to package/restore the Orekit cache bundle

## Sync Policy

This directory is a tracked source of truth for your reusable AI workflow files.

When you update the live global skill in `~/.codex/skills/orekit-mcp`, copy the
changes here (or update here first, then sync out). Keep both in sync manually
until you create a dedicated repo/tooling for these files.

The Orekit docs MCP server code can be installed in a global runtime folder
(recommended: `~/.codex/mcp/orekit-docs/`) and launched from there.

Current recommended source repository:

- `~/Projects/mcp-tools` (contains `tools/orekit_docs_mcp/`)

## Reproducible Setup Helpers

This repo also includes reusable setup scripts:

- `scripts/orekit_mcp/sync_runtime.sh`
  - syncs MCP source code (default `~/Projects/mcp-tools/tools/orekit_docs_mcp`)
    to `~/.codex/mcp/orekit-docs/`
- `scripts/orekit_mcp/bootstrap_env.sh`
  - creates/updates the dedicated `mcp-tools-env` Conda env from the tracked
    environment spec
- `scripts/orekit_mcp/export_cache_bundle.sh`
- `scripts/orekit_mcp/import_cache_bundle.sh`
  - export/import a portable `~/.cache/orekit` bundle for offline reuse

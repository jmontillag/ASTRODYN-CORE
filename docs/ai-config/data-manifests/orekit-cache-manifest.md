# Orekit Cache Manifest (Reproducibility)

This file documents the expected local Orekit docs/tutorial cache used by the
`orekit_docs` MCP server.

## Why A Manifest Instead Of Committing The Full Cache

Committing the entire extracted Javadocs/tutorials cache into this repo is
possible, but usually a poor fit because it is:

- large (many HTML files)
- noisy in diffs
- not source code you actively edit

Recommended approach:

1. Track the MCP source code, env spec, config snippets, and install scripts in
   Git (done in this repo).
2. Track the Orekit cache contents with a manifest (this file).
3. Keep the actual cache in `~/.cache/orekit` or in a separate data repo / LFS
   bundle if you want a portable offline archive.

If you later want an all-in-one portable bundle, create a separate archive or
repo for the cache and keep this manifest as the contract.

## Expected Cache Root

- `~/.cache/orekit`

## Primary Source Artifacts

These are the user-downloaded artifacts that seed the cache:

- `orekit-13.1.4-javadoc.jar`
- `orekit-tutorials-13.1.zip`

Current SHA256 (example local copy):

- `orekit-13.1.4-javadoc.jar`
  - `96d9fc1004655a2360c8885e956dd5d33474ab2d88dd82901c621a8269044a71`
- `orekit-tutorials-13.1.zip`
  - `659e3e54396e658104a703138ee17e9ef43e79a39f92d45ad15b2f4939a2f784`

## Expected Extracted / Generated Layout

- `sources/javadocs/13.1.4/artifact/`
- `sources/javadocs/13.1.4/html/`
- `sources/tutorials/13.1/artifact/`
- `sources/tutorials/13.1/src/`
- `sources/python-wrapper-wiki/` (mirror/index/notes placeholders)
- `index/javadocs/13.1.4/javadocs.sqlite3`
- `mcp/` (runtime cache/config/logs)

## Rebuild Notes

If the SQLite index is missing, rebuild from the extracted Javadocs:

```bash
/home/astror/miniconda3/envs/mcp-tools-env/bin/python \
  /home/astror/.codex/mcp/orekit-docs/tools/orekit_docs_mcp/index_javadocs.py \
  index --rebuild
```

For global runtime usage, the index can also be rebuilt from the global runtime
copy once synced:

```bash
/home/astror/miniconda3/envs/mcp-tools-env/bin/python \
  /home/astror/.codex/mcp/orekit-docs/tools/orekit_docs_mcp/index_javadocs.py \
  index --rebuild
```

## Optional Portable Bundle Strategy

If you want to carry everything together across machines/projects, package the
cache separately and restore it to `~/.cache/orekit`:

```bash
bash scripts/orekit_mcp/export_cache_bundle.sh
bash scripts/orekit_mcp/import_cache_bundle.sh <bundle.tgz>
```

This keeps your code repo clean while preserving full offline reproducibility.

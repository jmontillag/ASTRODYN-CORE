# Heyoka Cache Manifest (Reproducibility)

This file documents the expected local heyoka docs cache used by the
`heyoka_docs` MCP server.

## Why A Manifest Instead Of Committing The Full Cache

Same rationale as the Orekit cache — the mirrored Sphinx HTML docs are large
(~200 MB for Python), noisy in diffs, and not source code you actively edit.

## Expected Cache Root

- `~/.cache/heyoka`

## Primary Source Artifacts

These are wget mirrors of the official Sphinx documentation sites:

- **heyoka.py** (Python): `https://bluescarni.github.io/heyoka.py/`
- **heyoka** (C++): `https://bluescarni.github.io/heyoka/`

Plus `objects.inv` files (Sphinx intersphinx inventories) fetched separately
with `curl` since wget skips binary files.

## Expected Layout

```
~/.cache/heyoka/
├── sources/
│   ├── python/
│   │   └── bluescarni.github.io/
│   │       └── heyoka.py/
│   │           ├── objects.inv              (intersphinx inventory, ~30 KB)
│   │           ├── index.html
│   │           ├── genindex.html
│   │           ├── notebooks/               (tutorial notebooks as HTML)
│   │           ├── autosummary_generated/   (API reference pages)
│   │           ├── _static/
│   │           └── ...
│   └── cpp/
│       └── bluescarni.github.io/
│           └── heyoka/
│               ├── objects.inv
│               ├── index.html
│               ├── api_reference.html
│               └── ...
└── index/
    └── heyoka_docs.sqlite3                  (FTS5 search index)
```

## Typical Sizes

| Component | Approximate Size |
|-----------|-----------------|
| Python docs mirror | ~200 MB |
| C++ docs mirror | ~6 MB |
| SQLite FTS5 index | ~200 KB |

## Rebuild Notes

If the SQLite index is missing, it is auto-built on first MCP server start.

Manual rebuild:

```bash
cd ~/.local/share/mcp-servers/heyoka-docs
python -m heyoka_docs_mcp.sphinx_index index --rebuild
```

Or via MCP tool call: `heyoka_rebuild_index()`

## Download Commands (Full Rebuild)

```bash
# Mirror Python docs
wget -r -np -N -E -k -l inf --reject="*.epub,*.pdf,*.zip" \
  -P ~/.cache/heyoka/sources/python \
  https://bluescarni.github.io/heyoka.py/

# Mirror C++ docs
wget -r -np -N -E -k -l inf --reject="*.epub,*.pdf,*.zip" \
  -P ~/.cache/heyoka/sources/cpp \
  https://bluescarni.github.io/heyoka/

# Fetch objects.inv (wget skips binary files)
curl -sL -o ~/.cache/heyoka/sources/python/bluescarni.github.io/heyoka.py/objects.inv \
  https://bluescarni.github.io/heyoka.py/objects.inv
curl -sL -o ~/.cache/heyoka/sources/cpp/bluescarni.github.io/heyoka/objects.inv \
  https://bluescarni.github.io/heyoka/objects.inv
```

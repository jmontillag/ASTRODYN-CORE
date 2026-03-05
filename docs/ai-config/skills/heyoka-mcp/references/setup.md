# Heyoka Docs MCP — Setup

## Architecture

```
~/.cache/heyoka/
├── sources/
│   ├── python/bluescarni.github.io/heyoka.py/   (wget mirror of heyoka.py Sphinx docs)
│   │   ├── objects.inv                           (Sphinx intersphinx inventory)
│   │   ├── index.html
│   │   ├── notebooks/                            (tutorial notebooks)
│   │   └── autosummary_generated/                (API reference pages)
│   └── cpp/bluescarni.github.io/heyoka/          (wget mirror of heyoka C++ Sphinx docs)
│       ├── objects.inv
│       └── ...
└── index/
    └── heyoka_docs.sqlite3                       (FTS5 search index, auto-built)

~/.local/share/mcp-servers/heyoka-docs/
└── heyoka_docs_mcp/
    ├── __init__.py
    ├── server.py                                 (FastMCP server, 5 tools + resources)
    └── sphinx_index.py                           (index builder + search + HTML extraction)
```

## Dependencies

The server runs in the `mcp-tools-env` Conda environment (shared with the Orekit MCP).

Required packages (already in `mcp-tools-env`):

- `mcp` (MCP Python SDK with FastMCP)
- `anyio` (async runtime for stdio transport)

No additional dependencies beyond the Python standard library are needed.
The `objects.inv` parser uses only `zlib` and `re` (stdlib).

## Claude Code Configuration

Entry in `~/.claude.json` (the main Claude Code config file — **not** `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "heyoka_docs": {
      "type": "stdio",
      "command": "/home/astror/miniconda3/envs/mcp-tools-env/bin/python",
      "args": [
        "-u",
        "/home/astror/.local/share/mcp-servers/heyoka-docs/heyoka_docs_mcp/server.py"
      ],
      "env": {
        "HEYOKA_CACHE_ROOT": "/home/astror/.cache/heyoka"
      }
    }
  }
}
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HEYOKA_CACHE_ROOT` | `~/.cache/heyoka` | Root for downloaded docs and index |
| `HEYOKA_DOCS_DB` | `<cache_root>/index/heyoka_docs.sqlite3` | Override SQLite path |

## Initial Setup (One-Time)

### 1. Download the documentation

```bash
# Python docs
wget -r -np -N -E -k -l inf --reject="*.epub,*.pdf,*.zip" \
  -P ~/.cache/heyoka/sources/python \
  https://bluescarni.github.io/heyoka.py/

# C++ docs
wget -r -np -N -E -k -l inf --reject="*.epub,*.pdf,*.zip" \
  -P ~/.cache/heyoka/sources/cpp \
  https://bluescarni.github.io/heyoka/

# objects.inv (wget skips binary files, fetch separately)
curl -sL -o ~/.cache/heyoka/sources/python/bluescarni.github.io/heyoka.py/objects.inv \
  https://bluescarni.github.io/heyoka.py/objects.inv
curl -sL -o ~/.cache/heyoka/sources/cpp/bluescarni.github.io/heyoka/objects.inv \
  https://bluescarni.github.io/heyoka/objects.inv
```

### 2. Build the index

```bash
cd ~/.local/share/mcp-servers/heyoka-docs
python -m heyoka_docs_mcp.sphinx_index index --rebuild
```

The index is also auto-built on first server start if missing.

### 3. Verify

```bash
# Test search
cd ~/.local/share/mcp-servers/heyoka-docs
python -m heyoka_docs_mcp.sphinx_index search "taylor_adaptive"

# Test doc retrieval
python -m heyoka_docs_mcp.sphinx_index doc "heyoka.taylor_adaptive" --max-chars 500
```

## MCP Tools Exposed

| Tool | Description |
|------|-------------|
| `heyoka_docs_info()` | Server metadata, index stats |
| `heyoka_search(query, doc_set?, kind?, limit?)` | FTS5 symbol search |
| `heyoka_get_symbol_doc(name, doc_set?, max_chars?)` | Documentation for a symbol |
| `heyoka_get_page(doc_set, page_path, max_chars?)` | Read any doc page by path |
| `heyoka_rebuild_index()` | Rebuild index from current cache |

## MCP Resources (Fallback)

| URI Pattern | Description |
|-------------|-------------|
| `heyoka://info` | Server info |
| `heyoka://search/{query}` | Symbol search |
| `heyoka://symbol/{name}` | Symbol documentation |
| `heyoka://page/{doc_set}/{page_path}` | Documentation page |

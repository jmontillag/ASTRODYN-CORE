# Heyoka Docs MCP — Maintenance

## Updating to a New Docs Version

When a new heyoka or heyoka.py version is released:

### 1. Re-download the docs

```bash
# Python docs (incremental, -N only downloads newer files)
wget -r -np -N -E -k -l inf --reject="*.epub,*.pdf,*.zip" \
  -P ~/.cache/heyoka/sources/python \
  https://bluescarni.github.io/heyoka.py/

# C++ docs
wget -r -np -N -E -k -l inf --reject="*.epub,*.pdf,*.zip" \
  -P ~/.cache/heyoka/sources/cpp \
  https://bluescarni.github.io/heyoka/

# Re-fetch objects.inv (always overwrite — small file)
curl -sL -o ~/.cache/heyoka/sources/python/bluescarni.github.io/heyoka.py/objects.inv \
  https://bluescarni.github.io/heyoka.py/objects.inv
curl -sL -o ~/.cache/heyoka/sources/cpp/bluescarni.github.io/heyoka/objects.inv \
  https://bluescarni.github.io/heyoka/objects.inv
```

### 2. Rebuild the index

Either call the MCP tool:

```
heyoka_rebuild_index()
```

Or from the command line:

```bash
cd ~/.local/share/mcp-servers/heyoka-docs
python -m heyoka_docs_mcp.sphinx_index index --rebuild
```

### 3. Restart Claude Code

The MCP server process needs to be restarted to pick up the new index.
Starting a new Claude Code session will do this automatically.

## Troubleshooting

### "Symbol not found" for a known symbol

The symbol may not be in `objects.inv`. Check:

```bash
cd ~/.local/share/mcp-servers/heyoka-docs
python -m heyoka_docs_mcp.sphinx_index search "your_symbol"
```

If missing, the symbol may be documented inline (not indexed by Sphinx).
Use `heyoka_get_page` with the known page path instead.

### Empty or nav-heavy text output

Sphinx themes wrap content differently. The server extracts `<article>` or
`<div role="main">` content and strips navigation. If a specific page returns
poor text, use `heyoka_get_page` with a higher `max_chars` value.

### Index auto-build fails on server start

Check that `objects.inv` files exist:

```bash
ls ~/.cache/heyoka/sources/python/bluescarni.github.io/heyoka.py/objects.inv
ls ~/.cache/heyoka/sources/cpp/bluescarni.github.io/heyoka/objects.inv
```

If missing, re-download with `curl` (see setup instructions).

### MCP server not starting

Verify the server loads:

```bash
conda run -n mcp-tools-env python -c "
import sys; sys.path.insert(0, '/home/astror/.local/share/mcp-servers/heyoka-docs')
from heyoka_docs_mcp.server import _build_server
mcp = _build_server()
print('OK:', len(mcp._tool_manager._tools), 'tools')
"
```

## Cache Sizes (Typical)

| Component | Size |
|-----------|------|
| Python docs mirror | ~200 MB |
| C++ docs mirror | ~6 MB |
| SQLite FTS5 index | ~200 KB |

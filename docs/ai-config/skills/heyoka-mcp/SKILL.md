---
name: heyoka-mcp
description: Use the `heyoka_docs` MCP tools to look up heyoka.py and heyoka C++ documentation — classes, functions, methods, attributes, tutorials, and API references — before writing or reviewing heyoka-related code. Trigger for Taylor integrator construction, expression system, variational equations, event detection, batch/ensemble propagation, extended precision, compiled functions, and whenever exact heyoka API details are uncertain.
---

# Heyoka Docs MCP

## Goal

Use the local `heyoka_docs` MCP server as the primary source for exact heyoka API details. Covers both the Python (`heyoka.py`) and C++ (`heyoka`) documentation sets with 707 indexed symbols and full tutorial/notebook content.

## Use The Tools First

Call these tools directly when available:

1. `heyoka_docs_info` — server metadata and index stats
2. `heyoka_search` — FTS5 search across all symbols (classes, functions, methods, attributes)
3. `heyoka_get_symbol_doc` — fetch documentation for a fully-qualified symbol
4. `heyoka_get_page` — read any documentation page by relative path
5. `heyoka_rebuild_index` — rebuild index after downloading new docs

Do not start with MCP resource discovery when the tools are already available.

## Workflow

1. Call `heyoka_search` with a focused query to find the relevant symbol or page.
2. Call `heyoka_get_symbol_doc` for the exact class/function to get its documentation.
3. For tutorials and usage examples, call `heyoka_get_page` with the doc set and page path.
4. Summarize the API facts (constructor signature, parameters, return types, usage patterns).
5. Write or patch code using those facts.

## Copy/Paste Recipes

### Symbol search and lookup

1. `heyoka_search(query="taylor_adaptive", limit=5)`
2. `heyoka_get_symbol_doc(name="heyoka.taylor_adaptive")`
3. `heyoka_get_symbol_doc(name="heyoka.var_ode_sys")`

### C++-specific lookup

1. `heyoka_search(query="taylor_adaptive", doc_set="cpp")`
2. `heyoka_get_symbol_doc(name="heyoka::taylor_adaptive", doc_set="cpp")`

### Tutorial pages

1. `heyoka_get_page(doc_set="python", page_path="notebooks/The adaptive integrator.html")`
2. `heyoka_get_page(doc_set="python", page_path="notebooks/var_ode_sys.html")`
3. `heyoka_get_page(doc_set="python", page_path="notebooks/Dense output.html")`
4. `heyoka_get_page(doc_set="python", page_path="notebooks/Event detection.html")`
5. `heyoka_get_page(doc_set="cpp", page_path="tut_taylor_method.html")`

### Resources path (fallback)

1. Read `heyoka://info`
2. Read `heyoka://search/taylor_adaptive`
3. Read `heyoka://symbol/heyoka.taylor_adaptive`
4. Read `heyoka://page/python/notebooks/The%20adaptive%20integrator.html`

## Query Guidance

Use short, high-signal queries:

- `taylor_adaptive` — the main integrator class
- `var_ode_sys` — variational equations / STM
- `make_vars` — creating symbolic variables
- `par` — runtime parameters
- `t_event` — terminal events
- `nt_event` — non-terminal events
- `cfunc` — compiled functions
- `model` — built-in models (pendulum, VSOP2013, etc.)
- `expression` — expression system

Filter by doc set when needed:

- `heyoka_search(query="taylor_adaptive", doc_set="python")` — Python API only
- `heyoka_search(query="taylor_adaptive", doc_set="cpp")` — C++ API only

Filter by kind:

- `heyoka_search(query="taylor", kind="class")` — classes only
- `heyoka_search(query="step", kind="method")` — methods only

## Key heyoka.py Patterns

- `hy.make_vars("x", "v")` — create symbolic state variables
- `hy.taylor_adaptive(sys=[(x, v), (v, -9.8*hy.sin(x))], state=[0.05, 0.025])` — build integrator
- `ta.step()` — single adaptive step, returns `(outcome, step_size)`
- `ta.propagate_until(t_final)` — propagate to time
- `ta.propagate_grid(time_grid)` — evaluate at specific times
- `hy.var_ode_sys(sys, args=hy.var_args.vars, order=1)` — variational equations
- `ta.step(max_delta_t=h, write_tc=True)` — step with Taylor coefficient capture
- `ta.tc` — Taylor coefficients for dense output
- `hy.par[i]` — runtime parameters (changeable without recompilation)

## Failure Handling

If `heyoka_docs` tools are unavailable:

1. State that MCP tool calls are not exposed or not healthy.
2. Report the exact tool error when available.
3. Try reading MCP resources if the runtime supports resources but not tools.
4. Fall back to local knowledge and clearly mark uncertainty for signatures.

## References

- `references/setup.md` — server installation and configuration
- `references/maintenance.md` — updating docs and rebuilding the index
- `references/queries.md` — high-signal search queries by domain

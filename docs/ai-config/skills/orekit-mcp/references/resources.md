# Orekit MCP: Resources And Templates

MCP servers can expose two kinds of things:

- **Tools**: RPC-like functions (inputs -> outputs). Great when the runtime can call tools.
- **Resources**: read-only, URI-addressable payloads (docs, JSON blobs, text excerpts). Great as a fallback when the runtime can only read resources, or you want stable “doc addresses”.

Resources come in two flavors:

- **Static resources**: concrete URIs like `orekit://info` that show up in `resources/list`.
- **Resource templates**: parameterized URI patterns like `orekit://class/{fqcn}` that show up in `resources/templates/list`.

## Why Add Resources If Tools Already Exist?

- Some runtimes/clients support **resources** but not **tool calls**.
- Resources give you stable “doc pointers” (URIs) you can reference repeatedly.
- Resources can be made small and bounded (excerpts), avoiding huge doc dumps.

## Current Orekit Docs Resource Surface (Recommended)

Static resources (discoverable via `resources/list`):

- `orekit://info`
- `orekit://root/help-doc.html`
- `orekit://root/overview-summary.html`
- `orekit://root/overview-tree.html`
- `orekit://root/allclasses-index.html`
- `orekit://root/index-all.html`

Templated resources (discoverable via `resources/templates/list`):

- `orekit://root/{page_name}`
- `orekit://search/{query}`
- `orekit://class/{fqcn}`
- `orekit://class/{fqcn}/{max_chars}`
- `orekit://member/{fqcn}/{member_name}`
- `orekit://member/{fqcn}/{member_name}/{max_chars}`
- `orekit://package/{package_name}`
- `orekit://package/{package_name}/{max_chars}`

## URI Encoding Rules (Important)

Template placeholders like `{fqcn}` match a **single path segment**, so they cannot contain `/`.

Guidance:

- Use percent-encoding for special characters in path segments.
- For `search/{query}`, use `+` for spaces (standard “querystring style”), and decode with `unquote_plus` server-side.

Examples:

- `orekit://class/org.orekit.time.AbsoluteDate`
- `orekit://search/FramesFactory+getGCRF`

## Payload Design (What Makes Resources Effective)

For LLM usability, keep responses compact, structured, and bounded:

- Prefer `mime_type="application/json"` and return a dict payload.
- Include “where this came from”: `javadoc_version`, `page_path`, and any `symbol` / `selected` record.
- Include a bounded `text` excerpt plus `max_chars` controls (template variants are fine).
- Keep resources additive: do not remove the existing tools; resources should call the same underlying logic so results match.

## Using Resources In Codex (When Tools Aren’t Available)

If Codex can’t call `orekit_docs_*` tools, you can still try resource reads:

- `list_mcp_resources(server="orekit_docs")`
- `list_mcp_resource_templates(server="orekit_docs")`
- `read_mcp_resource(server="orekit_docs", uri="orekit://info")`

Then use the returned excerpts to pick the right class/member and proceed.


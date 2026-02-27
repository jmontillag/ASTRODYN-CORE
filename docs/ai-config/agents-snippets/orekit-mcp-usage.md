## Orekit MCP usage (Preferred for API tasks)

When a task involves Orekit API usage, class/method selection, signatures,
overloads, or uncertain Orekit behavior, use the `orekit_docs` MCP tools before
writing code.

Preferred workflow:

1. `orekit_docs_info` to confirm the active docs/index.
2. `orekit_search_symbols` to find the relevant classes/methods.
3. `orekit_get_class_doc` and/or `orekit_get_member_doc` to retrieve the docs.
4. Then write code, noting Java-to-Python wrapper differences when relevant.

Do not rely only on memory for exact Orekit signatures if the MCP tools are
available in the current session.

If the current Codex runtime does not expose MCP tool calls (or the server
handshake fails), state that limitation clearly and fall back to local knowledge.

If the runtime supports MCP resources but not tools, use the `orekit://*`
resources as a compatibility layer:

1. `orekit://info` (confirm docs version/paths)
2. `orekit://search/<query>` (find candidate classes/members)
3. `orekit://class/<fqcn>/<max_chars>` and/or `orekit://member/<fqcn>/<member_name>/<max_chars>`
4. Then write code using the retrieved API facts.

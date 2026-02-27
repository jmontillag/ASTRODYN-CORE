---
name: orekit-mcp
description: Use the `orekit_docs` MCP tools to look up Orekit Javadocs classes, methods, overloads, and root pages before writing or reviewing Orekit-related code. Trigger for Orekit propagation, force models, event detectors, frames, time/date APIs, TLE or DSST usage, signature/overload questions, and whenever exact Orekit API details are uncertain (especially when writing Python wrapper code from Java docs).
---

# Orekit MCP

## Goal

Use the local `orekit_docs` MCP server as the primary source for exact Orekit API details. Prefer tool-backed lookup over memory for signatures, overloads, and class selection.

## Use The Tools First

Call these tools directly when available:

1. `orekit_docs_info`
2. `orekit_search_symbols`
3. `orekit_get_class_doc`
4. `orekit_get_member_doc`
5. `orekit_get_root_page` (for help/index/tree pages)

Do not start with MCP resource discovery when the tools are already available.

## Workflow

1. Call `orekit_docs_info` to confirm the active docs version and local index paths.
2. Call `orekit_search_symbols` with a focused query that includes class names, method names, and domain words (for example: `NumericalPropagator addForceModel drag`).
3. Call `orekit_get_class_doc` for the candidate class to confirm role, inheritance, and nearby methods.
4. Call `orekit_get_member_doc` for the exact method/constructor/field, using `overload_hint` when overloads exist.
5. Summarize the API facts (class, method signature, parameters, expected usage).
6. Write or patch code using those facts, then note any Java to Python wrapper adjustments.

## Copy/Paste Recipes

Tools path (preferred):

1. `orekit_docs_info()`
2. `orekit_search_symbols(query="FramesFactory getGCRF", limit=5)`
3. `orekit_get_class_doc("org.orekit.frames.FramesFactory", max_chars=4000)`
4. `orekit_get_member_doc(fqcn="org.orekit.frames.FramesFactory", member_name="getGCRF", overload_hint="()", max_chars=3000)`

Resources path (fallback when tools are unavailable):

1. Read `orekit://info`
2. Read `orekit://search/FramesFactory+getGCRF`
3. Read `orekit://class/org.orekit.frames.FramesFactory/4000`
4. Read `orekit://member/org.orekit.frames.FramesFactory/getGCRF/3000`

## Query Guidance

Use short, high-signal queries:

- `NumericalPropagator addForceModel`
- `DormandPrince853Integrator`
- `FramesFactory EME2000`
- `AbsoluteDate constructor UTC`
- `DragForce IsotropicDrag`

Prefer multiple narrow searches over one long natural-language paragraph.

## Java To Python Wrapper Notes

Treat Javadocs as the source of API semantics, then adapt syntax for the Python wrapper:

- Confirm exact Java method names and parameter types first.
- Expect constructor and overload differences in Python wrapper usage.
- State the Java signature you found before presenting Python code when ambiguity is likely.
- Mention wrapper-specific imports or object creation differences if known.

## Failure Handling

If `orekit_docs` tools are unavailable in the current session:

1. State that MCP tool calls are not exposed or not healthy.
2. Report the exact tool error when available.
3. Try reading MCP resources if the runtime supports resources but not tools (see `references/resources.md`).
4. If resources are available, prefer `orekit://search/...` + `orekit://class/...` / `orekit://member/...` over guessing signatures.
5. Fall back to local knowledge and clearly mark uncertainty for signatures/overloads.

## Output Expectations

When using this skill, explicitly list which `orekit_docs` MCP tools were called before giving the final Orekit code or explanation.

## References

Use bundled references when it saves tool calls or prevents common mistakes:

- `references/queries.md` for high-signal search queries by domain
- `references/overload-hints.md` for `overload_hint` patterns
- `references/resources.md` for MCP resource/template usage and URI surface
- `references/enable-server.md` for Codex MCP server setup pointers
- `references/troubleshooting.md` for common failure modes

# Orekit MCP: Overload Hint Playbook

Use `orekit_get_member_doc` when you already know the class and member name.
If multiple overloads exist, pass `overload_hint` to force the right one.

## Pattern

1. Search to find the member and likely signature:
   - `orekit_search_symbols("AbsoluteDate constructor")`
2. Fetch class doc to confirm context:
   - `orekit_get_class_doc("org.orekit.time.AbsoluteDate")`
3. Fetch member doc with an overload hint:
   - `orekit_get_member_doc(fqcn="org.orekit.time.AbsoluteDate", member_name="AbsoluteDate", overload_hint="(java.lang.String, org.orekit.time.TimeScale)")`

## Choosing an `overload_hint`

- Prefer the exact Java parameter type list in parentheses.
- If you only know one parameter type, keep it partial but distinctive.

## Common member names

- Constructor: `member_name="<ClassName>"` (constructors are indexed by class name)
- Static factory method: `member_name="<methodName>"`
- Instance method: `member_name="<methodName>"`

## Java to Python wrapper note

Always present the Java signature you found, then write Python wrapper code.
If the wrapper differs (constructors, overload resolution), call it out as a
wrapper assumption.

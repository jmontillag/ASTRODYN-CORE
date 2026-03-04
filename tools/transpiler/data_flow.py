"""Data-flow analysis for vector tracking and cross-order dependencies.

Tracks vector element lists (for derivative call size determination)
and analyzes scratch read/write sets across orders to generate
intermediates struct fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tools.transpiler.ast_parser import (
    DerivCallDoOne,
    DerivCallFull,
    ParsedFunction,
    ScratchRead,
    ScratchWrite,
    VectorConstruct,
    VectorElementExtract,
    VectorExtend,
    parse_order_file,
)

# ---------------------------------------------------------------------------
# Vector Tracker
# ---------------------------------------------------------------------------


class VectorTracker:
    """Tracks logical vector element lists throughout a function.

    When Python code does ``r_vector = np.array([r, rp])``, we record
    ``r_vector → ["r", "rp"]``.  When it does ``np.append(r_vector, rpp)``,
    we update to ``r_vector → ["r", "rp", "rpp"]``.

    Elements are stored as C++ expression strings (emitted from AST nodes).
    The ``ast_nodes`` dict stores the raw AST nodes for elements that are
    expressions (non-simple-names), keyed by (vector_name, index).
    """

    def __init__(self) -> None:
        self.vectors: Dict[str, List[str]] = {}
        self.ast_nodes: Dict[Tuple[str, int], "ast.expr"] = {}

    def construct(self, name: str, elements: List[str],
                  element_nodes: Optional[List] = None) -> None:
        self.vectors[name] = list(elements)
        if element_nodes:
            for i, node in enumerate(element_nodes):
                self.ast_nodes[(name, i)] = node

    def extend(self, name: str, element: str,
               element_node=None) -> None:
        if name not in self.vectors:
            raise KeyError(f"VectorTracker: extending unknown vector {name!r}")
        idx = len(self.vectors[name])
        self.vectors[name].append(element)
        if element_node is not None:
            self.ast_nodes[(name, idx)] = element_node

    def get_elements(self, name: str) -> List[str]:
        if name not in self.vectors:
            raise KeyError(f"VectorTracker: unknown vector {name!r}")
        return self.vectors[name]

    def get_length(self, name: str) -> int:
        return len(self.get_elements(name))

    def get_cpp_elements(self, name: str) -> List[str]:
        """Get elements as C++ expression strings.

        For simple Name elements, returns the name directly.
        For expression elements, uses the stored AST node to emit C++.
        """
        from tools.transpiler.cpp_emitter import emit_expr
        elements = self.get_elements(name)
        result = []
        for i, elem in enumerate(elements):
            node = self.ast_nodes.get((name, i))
            if node is not None:
                import ast as ast_mod
                if isinstance(node, ast_mod.Name):
                    result.append(node.id)
                else:
                    result.append(emit_expr(node))
            else:
                result.append(elem)
        return result

    def copy(self) -> "VectorTracker":
        vt = VectorTracker()
        vt.vectors = {k: list(v) for k, v in self.vectors.items()}
        vt.ast_nodes = dict(self.ast_nodes)
        return vt


def build_vector_tracker(
    func: ParsedFunction,
    seed: Optional[VectorTracker] = None,
) -> VectorTracker:
    """Walk a parsed function and build its VectorTracker.

    For orders > 1, vectors may be read from scratch (e.g.
    ``beta_vector = s["beta_vector"]``).  Pass a ``seed`` tracker
    built from the previous order so we know their element lists.
    """
    vt = VectorTracker()
    if seed is not None:
        # Pre-populate with vectors from previous order
        for k, v in seed.vectors.items():
            vt.vectors[k] = list(v)

    for stmt in func.statements:
        if isinstance(stmt, VectorConstruct):
            vt.construct(stmt.target, stmt.elements,
                        getattr(stmt, 'element_nodes', None))
        elif isinstance(stmt, VectorExtend):
            vt.extend(stmt.target, stmt.element,
                     getattr(stmt, 'element_node', None))
        elif isinstance(stmt, ScratchRead) and stmt.key.endswith("_vector"):
            # Vector read from scratch — alias target to the same name
            if stmt.key in vt.vectors and stmt.target not in vt.vectors:
                vt.vectors[stmt.target] = list(vt.vectors[stmt.key])
                # Copy AST nodes for the aliased vector
                for (vname, idx), node in list(vt.ast_nodes.items()):
                    if vname == stmt.key:
                        vt.ast_nodes[(stmt.target, idx)] = node
    return vt


def build_chained_vector_tracker(order: int, base_dir: str | Path) -> VectorTracker:
    """Build a VectorTracker for the given order by chaining from Order 1."""
    base = Path(base_dir)
    seed = None
    for o in range(1, order + 1):
        fp = base / f"taylor_order_{o}.py"
        compute, _ = parse_order_file(fp, o)
        seed = build_vector_tracker(compute, seed)
    return seed


# ---------------------------------------------------------------------------
# Vector-only locals detection
# ---------------------------------------------------------------------------


def find_vector_only_locals(
    func: ParsedFunction,
    vt: VectorTracker,
) -> List[str]:
    """Find vector element names that are NOT in the scratch write set.

    These are Python local variables stored as vector elements (e.g.
    ``r3p = 3*r2*rp`` used in ``r3_vector = np.array([r3, r3p])``).
    They need to be in the intermediates struct so consuming orders
    can access them via derivative calls on the inherited vectors.

    Only considers ``ast.Name`` elements (expression elements like
    ``2 * f2rp`` don't introduce new variables — their sub-names are
    assumed to be available via scratch reads).
    """
    import ast as ast_mod

    # Collect all scratch write keys (scalar only)
    scratch_writes: Set[str] = set()
    for stmt in func.statements:
        if isinstance(stmt, ScratchWrite) and not stmt.key.endswith("_vector"):
            scratch_writes.add(stmt.key)

    # Collect all Name-type vector elements
    needed: Set[str] = set()
    for vec_name, elements in vt.vectors.items():
        for i, elem in enumerate(elements):
            node = vt.ast_nodes.get((vec_name, i))
            if node is not None:
                # Only Name nodes introduce a variable reference
                if isinstance(node, ast_mod.Name):
                    name = node.id
                    if name not in scratch_writes:
                        needed.add(name)
            else:
                # String element — check if it's in scratch writes
                if elem not in scratch_writes:
                    needed.add(elem)

    return sorted(needed)


# ---------------------------------------------------------------------------
# Cross-order dependency analysis
# ---------------------------------------------------------------------------


@dataclass
class CrossOrderDeps:
    """Fields that Order N reads from Order N-1's scratch writes.

    These become the fields of ``Order{N-1}Intermediates`` struct.
    """
    scalar_fields: List[str] = field(default_factory=list)
    vector_fields: Dict[str, int] = field(default_factory=dict)  # name → length


def get_scratch_writes(func: ParsedFunction) -> Tuple[Set[str], Dict[str, int]]:
    """Get all scratch keys written by a compute function.

    Returns (scalar_keys, vector_keys_with_lengths).
    Vector keys are identified by names ending in '_vector'.
    """
    scalar_keys: Set[str] = set()
    vector_keys: Dict[str, int] = {}

    for stmt in func.statements:
        if isinstance(stmt, ScratchWrite):
            if stmt.key.endswith("_vector"):
                # Length will be determined from VectorTracker
                vector_keys[stmt.key] = 0  # placeholder
            else:
                scalar_keys.add(stmt.key)

    return scalar_keys, vector_keys


def get_scratch_reads(func: ParsedFunction) -> Tuple[Set[str], Set[str]]:
    """Get all scratch keys read by a compute function.

    Returns (scalar_keys, vector_keys).
    """
    scalar_keys: Set[str] = set()
    vector_keys: Set[str] = set()

    for stmt in func.statements:
        if isinstance(stmt, ScratchRead):
            if stmt.key.endswith("_vector"):
                vector_keys.add(stmt.key)
            else:
                scalar_keys.add(stmt.key)

    return scalar_keys, vector_keys


def analyze_cross_order(
    prev_order_file: str | Path,
    prev_order: int,
    curr_order_file: str | Path,
    curr_order: int,
) -> CrossOrderDeps:
    """Determine what Order curr reads from Order prev's scratch.

    The intersection of prev's writes and curr's reads gives us the
    fields needed in the intermediates struct.
    """
    prev_compute, _ = parse_order_file(prev_order_file, prev_order)
    curr_compute, _ = parse_order_file(curr_order_file, curr_order)

    prev_writes_scalar, prev_writes_vector = get_scratch_writes(prev_compute)
    curr_reads_scalar, curr_reads_vector = get_scratch_reads(curr_compute)

    # Build vector tracker for previous order to get vector lengths
    prev_vt = build_vector_tracker(prev_compute)

    deps = CrossOrderDeps()

    # Scalar intersection
    shared_scalars = prev_writes_scalar & curr_reads_scalar
    deps.scalar_fields = sorted(shared_scalars)

    # Vector intersection
    shared_vectors = set(prev_writes_vector.keys()) & curr_reads_vector
    for vname in sorted(shared_vectors):
        # Get the vector length at the end of the previous order
        try:
            length = prev_vt.get_length(vname)
        except KeyError:
            length = 0  # fallback
        deps.vector_fields[vname] = length

    return deps


# ---------------------------------------------------------------------------
# Derivative call analysis (for inline helper generation)
# ---------------------------------------------------------------------------


@dataclass
class DerivCallInfo:
    """Info about a derivative function call for inline wrapper generation."""
    func_name: str        # e.g. "derivatives_of_inverse"
    vector_length: int    # length of the input vector
    do_one: bool
    has_param: bool       # whether it's a _wrt_param variant


def collect_deriv_calls(func: ParsedFunction, seed_vt: Optional[VectorTracker] = None) -> List[DerivCallInfo]:
    """Collect all derivative call patterns used in a function.

    Tracks vectors incrementally to get correct lengths at each call point.
    Returns unique (func_name, vector_length, do_one) combinations
    needed for inline wrapper generation.
    """
    vt = VectorTracker()
    if seed_vt is not None:
        vt.vectors = {k: list(v) for k, v in seed_vt.vectors.items()}
        vt.ast_nodes = dict(seed_vt.ast_nodes)

    seen: set = set()
    calls: List[DerivCallInfo] = []

    for stmt in func.statements:
        # Update tracker as we go
        if isinstance(stmt, VectorConstruct):
            vt.construct(stmt.target, stmt.elements,
                        getattr(stmt, 'element_nodes', None))
        elif isinstance(stmt, VectorExtend):
            vt.extend(stmt.target, stmt.element,
                     getattr(stmt, 'element_node', None))
        elif isinstance(stmt, ScratchRead) and stmt.key.endswith("_vector"):
            if stmt.key in vt.vectors and stmt.target not in vt.vectors:
                vt.vectors[stmt.target] = list(vt.vectors[stmt.key])

        if isinstance(stmt, (DerivCallDoOne, DerivCallFull)):
            if stmt.inline_elements is not None:
                vec_len = len(stmt.inline_elements)
            elif stmt.vector_arg is not None:
                vec_len = vt.get_length(stmt.vector_arg)
            else:
                vec_len = 0
            has_param = stmt.param_vector_arg is not None
            do_one = isinstance(stmt, DerivCallDoOne)

            key = (stmt.func_name, vec_len, do_one, has_param)
            if key not in seen:
                seen.add(key)
                calls.append(DerivCallInfo(
                    func_name=stmt.func_name,
                    vector_length=vec_len,
                    do_one=do_one,
                    has_param=has_param,
                ))

    return calls


# ---------------------------------------------------------------------------
# Coefficient struct field extraction
# ---------------------------------------------------------------------------


def extract_coeff_fields(func: ParsedFunction) -> Dict[str, List[str]]:
    """Extract the fields needed in the OrderNCoefficients struct.

    Groups by category:
    - "state_values": nu_0, q1_0, ...
    - "eom_values": q1p_0, q2p_0, ... (or q1pp_0 for order 2)
    - "partials": q1p_nu, q1p_Lr, ...
    - "map_components": map_components_col{N}
    """
    from tools.transpiler.ast_parser import MapAssign

    fields: Dict[str, List[str]] = {
        "state_values": [],
        "eom_values": [],
        "partials": [],
        "map_columns": [],
    }

    # For higher orders, the coefficient struct only needs the NEW coefficients
    # (not those from lower orders). The evaluate function reads them from scratch.
    # We look at what evaluate reads to determine the fields.

    # For now, collect from scratch writes that match the EOM/partial patterns
    for stmt in func.statements:
        if isinstance(stmt, MapAssign):
            fields["map_columns"].append(str(stmt.column))

    return fields

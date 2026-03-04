#!/usr/bin/env python3
"""GEqOE Taylor order transpiler — Python to C++.

Usage:
    python tools/transpiler/cli.py <order> [--output-dir DIR] [--source-dir DIR] [--dry-run]

Example:
    python tools/transpiler/cli.py 2
    python tools/transpiler/cli.py 2 --output-dir src/astrodyn_core/geqoe_cpp/
    python tools/transpiler/cli.py 2 --dry-run   # print to stdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.transpiler.ast_parser import parse_order_file
from tools.transpiler.cpp_emitter import (
    generate_compute_coefficients,
    generate_evaluate_order,
    generate_inline_helpers,
)
from tools.transpiler.data_flow import (
    build_chained_vector_tracker,
    build_vector_tracker,
    collect_deriv_calls,
    analyze_cross_order,
    find_vector_only_locals,
)
from tools.transpiler.struct_generator import (
    extract_eval_coeff_fields,
    extract_intermediates_fields,
    extract_map_column,
    generate_header,
    generate_source,
)


def transpile_order(
    order: int,
    source_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> None:
    """Transpile a single Taylor order from Python to C++."""
    py_file = source_dir / f"taylor_order_{order}.py"
    if not py_file.exists():
        raise FileNotFoundError(f"Python source not found: {py_file}")

    print(f"[transpiler] Parsing taylor_order_{order}.py ...")
    compute, evaluate = parse_order_file(py_file, order)

    # Build vector tracker (seeded from previous orders for cross-order vectors)
    print(f"[transpiler] Building vector tracker (orders 1..{order}) ...")
    if order > 1:
        seed_vt = build_chained_vector_tracker(order - 1, source_dir)
    else:
        seed_vt = None
    # Full tracker through current order (used only for deriv_calls analysis)
    vt = build_chained_vector_tracker(order, source_dir)

    # Collect derivative call patterns (incrementally tracked)
    deriv_calls = collect_deriv_calls(compute, seed_vt)
    print(f"[transpiler] Found {len(deriv_calls)} derivative call patterns")

    # Generate inline helpers
    inline_helpers = generate_inline_helpers(deriv_calls)

    # Extract current order's intermediates (scratch writes → OrderNIntermediates)
    intermediates_fields = extract_intermediates_fields(compute)
    print(f"[transpiler] Order {order} intermediates: {len(intermediates_fields)} fields")

    # Extract coefficient struct fields from evaluate
    eval_fields = extract_eval_coeff_fields(evaluate)
    map_col = extract_map_column(compute)
    print(f"[transpiler] Order {order} coefficients: {len(eval_fields)} fields, map col {map_col}")

    # Detect vector-only locals from previous order
    # These are variables stored in vectors but not in scratch writes.
    # They need synthetic reads from the intermediates struct.
    vec_only_locals: list = []
    if order > 1 and seed_vt is not None:
        prev_py = source_dir / f"taylor_order_{order - 1}.py"
        prev_compute, _ = parse_order_file(prev_py, order - 1)
        prev_vt = build_vector_tracker(prev_compute)
        vec_only_locals = find_vector_only_locals(prev_compute, prev_vt)
        if vec_only_locals:
            print(f"[transpiler] Vector-only locals from order {order - 1}: {vec_only_locals}")

    # Generate compute body
    prev_inter = f"inter{order - 1}" if order > 1 else None
    compute_body = generate_compute_coefficients(
        order, compute, seed_vt, None, prev_inter, vec_only_locals)

    # Generate evaluate body
    evaluate_body = generate_evaluate_order(order, evaluate)

    # Generate header
    prev_header = f"taylor_order_{order - 1}.hpp" if order > 1 else None
    header = generate_header(
        order, eval_fields, map_col, intermediates_fields, prev_header)

    # Generate source
    prev_headers = [f"taylor_order_{i}.hpp" for i in range(1, order)]
    source = generate_source(
        order, inline_helpers, compute_body, evaluate_body, prev_headers,
        eval_coeff_fields=eval_fields, map_col=map_col)

    if dry_run:
        print("\n" + "=" * 70)
        print(f"=== taylor_order_{order}.hpp ===")
        print("=" * 70)
        print(header)
        print("\n" + "=" * 70)
        print(f"=== taylor_order_{order}.cpp ===")
        print("=" * 70)
        print(source)
        return

    # Write files
    include_dir = output_dir / "include"
    src_dir = output_dir / "src"
    include_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)

    hpp_path = include_dir / f"taylor_order_{order}.hpp"
    cpp_path = src_dir / f"taylor_order_{order}.cpp"

    hpp_path.write_text(header)
    print(f"[transpiler] Wrote {hpp_path}")

    cpp_path.write_text(source)
    print(f"[transpiler] Wrote {cpp_path}")

    # Print pipeline wiring hints
    print(f"\n[transpiler] Pipeline wiring for order {order}:")
    print(f"  1. Add '#include \"taylor_order_{order}.hpp\"' to taylor_pipeline.hpp")
    print(f"  2. Add 'Order{order}Coefficients order{order};' to PreparedTaylorCoefficients")
    if intermediates_fields:
        print(f"  3. Add 'Order{order - 1}Intermediates inter{order - 1};' to PreparedTaylorCoefficients")
    print(f"  4. Update prepare/evaluate dispatchers in taylor_pipeline.cpp")
    print(f"  5. Add taylor_order_{order}.cpp to CMakeLists.txt")


def main():
    parser = argparse.ArgumentParser(
        description="GEqOE Taylor order transpiler: Python → C++")
    parser.add_argument("order", type=int, help="Taylor order to transpile (2, 3, or 4)")
    parser.add_argument(
        "--output-dir", type=str,
        default="src/astrodyn_core/geqoe_cpp/",
        help="Output directory for .hpp and .cpp files")
    parser.add_argument(
        "--source-dir", type=str,
        default="src/astrodyn_core/propagation/geqoe/",
        help="Python source directory containing taylor_order_N.py files")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print generated code to stdout instead of writing files")

    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    transpile_order(args.order, source_dir, output_dir, args.dry_run)


if __name__ == "__main__":
    main()

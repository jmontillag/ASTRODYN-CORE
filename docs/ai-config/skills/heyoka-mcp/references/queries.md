# Heyoka Docs MCP — Query Reference

## Core API Symbols (Python)

| Query | What it finds |
|-------|---------------|
| `taylor_adaptive` | Main integrator class |
| `taylor_adaptive_batch` | Batch (SIMD) integrator |
| `var_ode_sys` | Variational ODE system (STM) |
| `var_args` | Variational arguments enum |
| `make_vars` | Create symbolic variables |
| `expression` | Expression class |
| `par` | Runtime parameters |
| `time` | Time expression in non-autonomous systems |
| `t_event` | Terminal event |
| `nt_event` | Non-terminal event |
| `cfunc` | Compiled function |
| `diff` | Symbolic differentiation |

## Built-in Models (Python)

| Query | What it finds |
|-------|---------------|
| `model.sgp4` | SGP4 propagator model |
| `model.vsop2013` | VSOP2013 planetary theory |
| `model.elp2000` | ELP2000 lunar theory |
| `model.egm2008` | EGM2008 geopotential |
| `model.cr3bp` | Circular restricted 3-body problem |
| `model.pendulum` | Simple pendulum |

## Tutorials (Python) — Use with `heyoka_get_page`

| Page path | Topic |
|-----------|-------|
| `notebooks/The adaptive integrator.html` | Integrator basics, step control |
| `notebooks/Customising the adaptive integrator.html` | Tolerance, order, compact mode |
| `notebooks/ODEs with parameters.html` | `hy.par[i]` runtime parameters |
| `notebooks/Non-autonomous systems.html` | Time-dependent systems |
| `notebooks/Dense output.html` | Taylor coefficient dense output |
| `notebooks/Event detection.html` | Terminal and non-terminal events |
| `notebooks/var_ode_sys.html` | Variational equations / STM |
| `notebooks/Computing derivatives.html` | Automatic differentiation |
| `notebooks/Supporting large computational graphs.html` | Compact mode |
| `tut_taylor_method.html` | Taylor method theory |

## C++ API Symbols

| Query | What it finds |
|-------|---------------|
| `taylor_adaptive` (with `doc_set="cpp"`) | C++ integrator class |
| `expression` (with `doc_set="cpp"`) | C++ expression class |
| `make_vars` (with `doc_set="cpp"`) | C++ variable creation |
| `cfunc` (with `doc_set="cpp"`) | C++ compiled function |
| `llvm_state` (with `doc_set="cpp"`) | LLVM JIT state |

## Filtering Tips

- `kind="class"` — classes and enums
- `kind="function"` — top-level functions
- `kind="method"` — class methods
- `kind="property"` — class properties
- `kind="attribute"` — enum members and attributes
- `kind="module"` — Python modules
- `kind="macro"` — C preprocessor macros (C++ only)

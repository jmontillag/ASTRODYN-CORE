"""Compatibility shim for DSST assembly helpers.

Canonical DSST assembly now lives in
``astrodyn_core.propagation.dsst_parts``.
"""

from astrodyn_core.propagation.dsst_parts import assemble_dsst_force_models

__all__ = ["assemble_dsst_force_models"]

"""FDTD Module -- grid/: Yee staggering + material rasterization.

No Maxwell's-equations or time-stepping knowledge -- only geometry and
staggering (docs/fdtd_module_plan.md Section 0.2). Separately testable with
no field ever stepped.
"""
from .rasterize import ComponentMask, rasterize_all, rasterize_component
from .yee import COMPONENT_OFFSETS, E_COMPONENTS, H_COMPONENTS, YeeGrid

__all__ = [
    "YeeGrid",
    "COMPONENT_OFFSETS",
    "E_COMPONENTS",
    "H_COMPONENTS",
    "ComponentMask",
    "rasterize_component",
    "rasterize_all",
]

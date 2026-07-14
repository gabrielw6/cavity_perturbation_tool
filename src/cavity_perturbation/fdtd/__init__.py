"""FDTD Module -- time-domain PerturbationModel sibling.

Predicts (f, Q) for a cavity + sample by time-stepping Maxwell's equations
on a Yee grid and extracting resonance from the recorded ringdown, rather
than by an algebraic/eigenvalue solve. See docs/fdtd_module_plan.md for the
full design.
"""
from .model import FDTDModel

__all__ = ["FDTDModel"]

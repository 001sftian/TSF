"""Weak-form RL/CEM sparse PDE discovery."""
from .data import FieldData, make_3d_reaction_advection_diffusion
from .library import build_scalar_library, CandidateTerm
from .weakform import WeakFormProjector, WeakSystem
from .search import RNSPDEDiscoverer, DiscoveryResult

__all__ = [
    "FieldData",
    "make_3d_reaction_advection_diffusion",
    "build_scalar_library",
    "CandidateTerm",
    "WeakFormProjector",
    "WeakSystem",
    "RNSPDEDiscoverer",
    "DiscoveryResult",
]

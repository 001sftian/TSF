"""Weak-form RL/CEM sparse PDE discovery."""
from .data import (
    FieldData,
    make_3d_allen_cahn,
    make_3d_burgers,
    make_3d_cahn_hilliard,
    make_3d_heat,
    make_3d_kdv,
    make_3d_nls,
    make_3d_ns,
    make_3d_reaction_advection_diffusion,
    make_3d_reaction_diffusion,
)
from .library import build_nls_library, build_ns_library, build_scalar_library, CandidateTerm
from .weakform import WeakFormProjector, WeakSystem
from .search import RNSPDEDiscoverer, DiscoveryResult

__all__ = [
    "FieldData",
    "make_3d_heat",
    "make_3d_burgers",
    "make_3d_ns",
    "make_3d_allen_cahn",
    "make_3d_cahn_hilliard",
    "make_3d_nls",
    "make_3d_kdv",
    "make_3d_reaction_advection_diffusion",
    "make_3d_reaction_diffusion",
    "build_scalar_library",
    "build_nls_library",
    "build_ns_library",
    "CandidateTerm",
    "WeakFormProjector",
    "WeakSystem",
    "RNSPDEDiscoverer",
    "DiscoveryResult",
]

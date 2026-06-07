from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weakpde_rl import (
    RNSPDEDiscoverer,
    WeakFormProjector,
    build_nls_library,
    build_ns_library,
    build_scalar_library,
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


def discover(name, data, terms, max_terms=4):
    system = WeakFormProjector(radius=1, stride=1, sample_fraction=0.06, seed=23).project(data, terms)
    result = RNSPDEDiscoverer(max_terms=max_terms, top_k=min(12, len(terms)), cem_iterations=5, population=45, random_state=5).fit(system)
    print(f"\n{name}")
    print(f"  target: {data.target}_t, weak samples: {len(system.theta)} ({100*len(system.theta)/((data.shape[0]-2)*(data.shape[1]-2)*(data.shape[2]-2)*(data.shape[3]-2)):.1f}% windows)")
    print(f"  residual: {result.relative_residual:.4f}")
    for term, coef in sorted(result.coefficients.items()):
        print(f"  {coef:+.4f} * {term}")


def main() -> None:
    scalar_terms = build_scalar_library(include_decoys=True)
    cases = [
        ("3D heat", make_3d_heat(), scalar_terms, 2),
        ("3D Burgers", make_3d_burgers(), scalar_terms, 3),
        ("3D Allen-Cahn", make_3d_allen_cahn(), scalar_terms, 4),
        ("3D Cahn-Hilliard", make_3d_cahn_hilliard(), scalar_terms, 3),
        ("3D KdV", make_3d_kdv(), scalar_terms, 3),
        ("3D reaction-diffusion", make_3d_reaction_diffusion(), scalar_terms, 4),
        ("3D reaction-advection-diffusion", make_3d_reaction_advection_diffusion(nx=14, ny=12, nz=10, nt=9), scalar_terms, 5),
    ]
    for case in cases:
        discover(*case)

    nls = make_3d_nls()
    discover("3D NLS / GP (real component)", nls, build_nls_library("a"), 2)

    ns = make_3d_ns()
    discover("3D Navier-Stokes / ABC flow (u component)", ns, build_ns_library("u"), 4)


if __name__ == "__main__":
    main()

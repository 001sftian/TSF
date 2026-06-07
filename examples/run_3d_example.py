from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weakpde_rl import (
    RNSPDEDiscoverer,
    WeakFormProjector,
    build_scalar_library,
    make_3d_reaction_advection_diffusion,
)


def main() -> None:
    data = make_3d_reaction_advection_diffusion(noise=0.0)
    terms = build_scalar_library(include_decoys=True)
    system = WeakFormProjector(radius=1, stride=1, sample_fraction=0.06, seed=23).project(data, terms)
    result = RNSPDEDiscoverer(max_terms=7, top_k=10, cem_iterations=3, population=25).fit(system)

    print("3-D weak-form RL/CEM PDE discovery")
    print("True PDE: u_t = -0.15*u -0.80*u_x +0.45*u_y -0.30*u_z +0.06*u_xx +0.06*u_yy +0.06*u_zz")
    print(f"Weak samples: {len(system.theta)}, candidate terms: {len(system.theta[0])}")
    print(f"Relative residual: {result.relative_residual:.4f}")
    print("Discovered PDE coefficients:")
    for name, value in sorted(result.coefficients.items()):
        print(f"  {name:10s} {value:+.4f}")


if __name__ == "__main__":
    main()

# WeakPDE-RL

A compact implementation of the weak-form reinforcement-learning/CEM sparse PDE
identification pipeline sketched in the prompt images.

## Pipeline

1. **Input data**: regular-grid scalar, complex split-field, and vector-pressure
   spatiotemporal fields, with synthetic 3-D datasets for validation.
2. **Fourth-order derivatives**: `u_t`, `u_x`, `u_xx`, `u_xxx`, Laplacian and
   related block operators are estimated with fourth-order finite-difference
   stencils in the discovery library.
3. **6% weak sampling**: examples use deterministic 6% sampling of compact weak
   windows before regression/search, matching sparse data settings.
4. **Weak projection**: local compact polynomial windows integrate the target
   time derivative and every candidate term, producing a weak design system that
   is more robust to noisy samples than pointwise regression.
5. **Adaptive libraries**: scalar atoms, gradients, Laplacian/biharmonic blocks,
   nonlinear advection/reaction terms, NLS split-complex terms, and NS
   vector-pressure terms.
6. **RL/CEM support search**: high-recall correlation action ranking, stochastic
   cross-entropy support-set sampling, ridge refits as environment transitions,
   residual/sparsity/complexity reward, beam refinement, and pruning.
7. **Output**: a sparse discovered PDE with coefficients in the original term
   scaling.

## 3-D PDE examples included

- 3-D heat equation
- 3-D Burgers equation
- 3-D Navier-Stokes / ABC-flow component equation
- 3-D Allen-Cahn equation
- 3-D Cahn-Hilliard-style operator-block equation
- 3-D nonlinear Schrödinger / Gross-Pitaevskii split real component
- 3-D KdV-style dispersive equation
- 3-D reaction-diffusion equation
- 3-D reaction-advection-diffusion equation

Run the single reaction-advection-diffusion example:

```bash
python examples/run_3d_example.py
```

Run the full suite:

```bash
python examples/run_3d_pde_suite.py
```

The reaction-advection-diffusion support is close to:

```text
u_t = -0.15*u -0.80*u_x +0.45*u_y -0.30*u_z +0.06*laplacian
```

## Tests

```bash
python -m pytest -q
```

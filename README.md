# WeakPDE-RL

A compact implementation of the weak-form reinforcement-learning/CEM sparse PDE
identification pipeline sketched in the prompt images.

## Pipeline

1. **Input data**: regular-grid scalar spatiotemporal fields, with a synthetic
   3-D reaction-advection-diffusion generator for validation.
2. **Weak projection**: local compact polynomial windows integrate the target
   `u_t` and every candidate term, producing a weak design system that is more
   robust to noisy samples than pointwise regression.
3. **Adaptive library**: scalar atom, gradient, Laplacian operator block,
   nonlinear, and derivative decoy terms.
4. **RL/CEM support search**: high-recall correlation action ranking, stochastic
   cross-entropy support-set sampling, ridge refits as environment transitions,
   residual/sparsity/complexity reward, beam refinement, and pruning.
5. **Output**: a sparse discovered PDE with coefficients in the original term
   scaling.

## Run the 3-D example

```bash
python examples/run_3d_example.py
```

Expected discovered support is close to:

```text
u_t = -0.15*u -0.80*u_x +0.45*u_y -0.30*u_z +0.06*laplacian
```

## Tests

```bash
python -m pytest -q
```

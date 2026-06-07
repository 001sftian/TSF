from weakpde_rl import RNSPDEDiscoverer, WeakFormProjector, build_scalar_library, make_3d_reaction_advection_diffusion


def test_3d_reaction_advection_diffusion_discovery():
    data = make_3d_reaction_advection_diffusion(noise=0.0)
    system = WeakFormProjector(radius=1, stride=1, sample_fraction=0.06, seed=23).project(data, build_scalar_library(include_decoys=True))
    result = RNSPDEDiscoverer(max_terms=5, top_k=12, cem_iterations=5, population=45, random_state=4).fit(system)

    expected = {"u", "u_x", "u_y", "u_z", "laplacian"}
    assert expected.issubset(result.coefficients)
    assert result.relative_residual < 0.05
    for name, truth in data.true_coefficients.items():
        assert abs(result.coefficients[name] - truth) < 0.15

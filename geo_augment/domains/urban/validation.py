def validate_urban_specs(synthesis, constraints, latent):
    if not (0.01 <= synthesis.perturbation_strength <= 0.5):
        raise ValueError("Invalid perturbation_strength")

    if synthesis.spatial_scale <= 0:
        raise ValueError("spatial_scale must be positive")

    if not (0 < synthesis.density_percentile < 100):
        raise ValueError("density_percentile must be in (0,100)")

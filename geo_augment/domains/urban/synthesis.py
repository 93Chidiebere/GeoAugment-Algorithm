from geo_augment.domains.urban.latent import generate_latent_urban_field
from geo_augment.domains.urban.constraints import apply_urban_constraints
from geo_augment.domains.urban.calibration import calibrate_urban_density


def generate_synthetic_urban_density(
    features,
    synthesis_spec,
    constraints,
    latent_spec,
    seed,
):
    latent = generate_latent_urban_field(
        shape=features.shape[1:],
        perturbation_strength=synthesis_spec.perturbation_strength,
        spatial_scale=synthesis_spec.spatial_scale,
        seed=seed,
        latent_spec=latent_spec,
    )

    constrained = apply_urban_constraints(
        latent,
        features,
        constraints,
    )

    return calibrate_urban_density(
        constrained,
        synthesis_spec.density_percentile,
        synthesis_spec.value_range,
    )

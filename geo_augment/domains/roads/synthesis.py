from geo_augment.domains.roads.latent import generate_latent_road_field
from geo_augment.domains.roads.constraints import apply_road_constraints
from geo_augment.domains.roads.calibration import calibrate_road_connectivity


def generate_synthetic_road_connectivity(
    features,
    synthesis_spec,
    constraints,
    latent_spec,
    seed,
):
    latent = generate_latent_road_field(
        shape=features.shape[1:],
        perturbation_strength=synthesis_spec.perturbation_strength,
        spatial_scale=synthesis_spec.spatial_scale,
        seed=seed,
        latent_spec=latent_spec,
    )

    constrained = apply_road_constraints(
        latent,
        features,
        constraints,
    )

    return calibrate_road_connectivity(
        constrained,
        synthesis_spec.connectivity_percentile,
        synthesis_spec.value_range,
    )

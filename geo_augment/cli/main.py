import argparse
import os

from geo_augment.io.raster import RasterLoader
from geo_augment.domains.floods.api import synthesize_flood_labels
from geo_augment.datasets.tiling import tile_raster
from geo_augment.datasets.export import export_npz, export_torch

from geo_augment.config import (
    load_yaml_config,
    build_flood_specs_from_config,
    summarize_specs,
)
from geo_augment.domains.floods.spec import (
    DEFAULT_FLOOD_SPEC,
    DEFAULT_FLOOD_CONSTRAINTS,
    DEFAULT_LATENT_SPEC,
)


def floods_generate(args):
    # ----------------------------------
    # Load specs (YAML or defaults)
    # ----------------------------------
    if args.config:
        cfg = load_yaml_config(args.config)
        synthesis_spec, constraints, latent_spec = (
            build_flood_specs_from_config(cfg)
        )
    else:
        synthesis_spec = DEFAULT_FLOOD_SPEC
        constraints = DEFAULT_FLOOD_CONSTRAINTS
        latent_spec = DEFAULT_LATENT_SPEC

    # ----------------------------------
    # Dry-run mode (validation only)
    # ----------------------------------
    if args.dry_run:
        print(summarize_specs(synthesis_spec, constraints, latent_spec))
        print("Dry-run successful. No data generated.")
        return

    # ----------------------------------
    # Load DEM
    # ----------------------------------
    print("Loading DEM...")
    dem = RasterLoader(args.dem).read()

    # ----------------------------------
    # Generate synthetic flood labels
    # ----------------------------------
    print("Generating synthetic flood labels...")
    labels = synthesize_flood_labels(
        dem=dem,
        synthesis_spec=synthesis_spec,
        constraints=constraints,
        latent_spec=latent_spec,
        n_samples=1,
    )[0]

    # ----------------------------------
    # Tile dataset
    # ----------------------------------
    print("Tiling dataset...")
    X, y = tile_raster(
        dem,
        labels,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )

    os.makedirs(args.out, exist_ok=True)

    # ----------------------------------
    # Export dataset
    # ----------------------------------
    print(f"Exporting dataset ({args.format})...")
    if args.format == "npz":
        export_npz(
            X,
            y,
            out_dir=args.out,
            name="geoaugment_flood",
            metadata={
                "tile_size": args.tile_size,
                "overlap": args.overlap,
                "spec": synthesis_spec.__dict__,
            },
        )
    elif args.format == "torch":
        export_torch(
            X,
            y,
            out_dir=args.out,
            name="geoaugment_flood",
        )
    else:
        raise ValueError("Unsupported export format")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        prog="geoaugment",
        description="GeoAugment synthetic GeoAI dataset generator",
    )

    subparsers = parser.add_subparsers(dest="domain")

    # -------------------------------
    # Floods domain
    # -------------------------------
    floods = subparsers.add_parser("floods", help="Flood-risk datasets")
    floods_sub = floods.add_subparsers(dest="command")

    generate = floods_sub.add_parser(
        "generate", help="Generate synthetic flood dataset"
    )

    generate.add_argument(
        "--dem",
        help="Path to DEM .tif (required unless --dry-run)",
    )
    generate.add_argument(
        "--out",
        help="Output directory",
        default="./geoaugment_output",
    )
    generate.add_argument("--tile-size", type=int, default=256)
    generate.add_argument("--overlap", type=int, default=64)

    generate.add_argument(
        "--format",
        choices=["npz", "torch"],
        default="npz",
    )

    # -------------------------------
    # New options
    # -------------------------------
    generate.add_argument(
        "--config",
        help="Path to YAML configuration file",
    )
    generate.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and exit without generating data",
    )

    generate.set_defaults(func=floods_generate)

    args = parser.parse_args()

    if hasattr(args, "func"):
        if not args.dry_run and not args.dem:
            parser.error("--dem is required unless --dry-run is specified")
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

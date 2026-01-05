import argparse
import os

from geo_augment.io.raster import RasterLoader
from geo_augment.domains.floods.features import stack_flood_features
from geo_augment.domains.floods.api import synthesize_flood_labels
from geo_augment.datasets.tiling import tile_raster
from geo_augment.datasets.export import export_npz, export_torch


def floods_generate(args):
    print("Loading DEM...")
    dem = RasterLoader(args.dem).read()

    print("Stacking flood features...")
    features = stack_flood_features(dem)

    print("Generating synthetic flood labels...")
    labels = synthesize_flood_labels(
        dem,
        n_samples=1,
        percentile=args.percentile
    )[0]

    print("Tiling dataset...")
    X, y = tile_raster(
        features,
        labels,
        tile_size=args.tile_size,
        overlap=args.overlap
    )

    os.makedirs(args.out, exist_ok=True)

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
                "percentile": args.percentile
            }
        )
    elif args.format == "torch":
        export_torch(
            X,
            y,
            out_dir=args.out,
            name="geoaugment_flood"
        )
    else:
        raise ValueError("Unsupported export format")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        prog="geoaugment",
        description="GeoAugment synthetic GeoAI dataset generator"
    )

    subparsers = parser.add_subparsers(dest="domain")

    # Floods domain
    floods = subparsers.add_parser("floods", help="Flood-risk datasets")
    floods_sub = floods.add_subparsers(dest="command")

    generate = floods_sub.add_parser("generate", help="Generate flood dataset")
    generate.add_argument("--dem", required=True, help="Path to DEM .tif")
    generate.add_argument("--out", required=True, help="Output directory")
    generate.add_argument("--tile-size", type=int, default=256)
    generate.add_argument("--overlap", type=int, default=64)
    generate.add_argument("--percentile", type=float, default=90.0)
    generate.add_argument(
        "--format",
        choices=["npz", "torch"],
        default="npz"
    )

    generate.set_defaults(func=floods_generate)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

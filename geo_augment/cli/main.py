import click
import numpy as np
from geo_augment.io.raster import RasterLoader
from geo_augment.domains.floods.features import stack_flood_features
from geo_augment.domains.floods.generator import FloodSyntheticGenerator


@click.group()
def cli():
    pass


@cli.command()
@click.option("--dem", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
def generate_flood(dem, out):
    loader = RasterLoader(dem)
    dem_data = loader.read()

    features = stack_flood_features(dem_data)

    generator = FloodSyntheticGenerator()
    result = generator.generate(features)

    np.save(out, result["risk"])

    click.echo(f"Synthetic flood risk saved to {out}")

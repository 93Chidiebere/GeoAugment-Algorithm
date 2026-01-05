from setuptools import setup, find_packages

setup(
    name="geoaugment",
    version="0.1.0",
    description="Constraint-aware synthetic geospatial data augmentation engine for GeoAI",
    author="Chidiebere V. Christopher",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "rasterio",
        "torch",
        "torchgeo",
        "click",
        "scikit-learn"
    ],
    entry_points={
        "console_scripts": [
            "geoaugment=geo_augment.cli.main:cli"
        ]
    },
    python_requires=">=3.10"
)
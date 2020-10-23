from setuptools import setup, find_packages

import os

_dir = os.path.dirname(os.path.realpath(__file__))

with open("VERSION", "r") as f:
    VERSION = f.read().strip("\n")

setup(
    name="mds",
    version=VERSION,
    description="Overdamped Langevin Importance Sampling",
    url="https://github.com/eborrell/mds",
    author="Enric Ribera Borrell",
    author_email="ribera.borrell@me.com",
    include_package_data=True,
    packages=find_packages(),
)

#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="nnsysident",
    version="0.0.0",
    description="Identifying and modelling the biological visual system with deep neural networks",
    author="Konstantin-Klemens Lurz",
    author_email="konstantin.lurz@uni-tuebingen.de",
    packages=find_packages(exclude=[]),
    install_requires=["neuralpredictors~=0.0.1"],
)

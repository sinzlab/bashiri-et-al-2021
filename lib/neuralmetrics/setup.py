#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="neuralmetrics",
    version="0.0.0",
    description="Metrics for evaluation and comparison of neural prediction models",
    author="Konstantin-Klemens Lurz and Mohammad Bashiri",
    author_email="sinzlab.tuebingen@gmail.com",
    packages=find_packages(exclude=[]),
    install_requires=[],
)

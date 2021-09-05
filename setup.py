from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="neuraldistributions",
    version="0.0.0",
    author="Mohammad Bashiri",
    author_email="mohammadbashiri93@gmail.com",
    description='Code for models used in Bashiri et al., "A Flow-based latent state generative model of neural population responses to natural images"',
    packages=find_packages(exclude=[]),
)
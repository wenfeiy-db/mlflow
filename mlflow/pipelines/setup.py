from setuptools import setup

with open("requirements.txt") as f:
    reqs = f.read()

setup(
    name="mlx",
    version="0.1.dev0",
    packages=["mlx"],
    scripts=["bin/mlx"],
    install_requires=reqs,
)

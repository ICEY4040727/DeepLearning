from setuptools import setup, find_packages

setup(
    name="deeplearning_pkg",
    version="0.0.1",
    description="Reusable utilities for DeepLearning notebooks",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[],
    python_requires=">=3.8",
)
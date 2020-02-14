#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

requirements = [
    "numpy>=1.16.2",
    "torch>=1.0.1",
    "matplotlib>=3.0.3",
    "scikit-learn>=0.20.3",
    "pandas>=0.24.2",
    "tqdm>=4.31.1",
    "statsmodels",
    "arviz",
]

setup_requirements = [
    "pytest-runner",
]
test_requirements = [
    "pytest",
]

setup(
    description="scVAE",
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords="sbvae",
    name="sbvae",
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    zip_safe=False,
)

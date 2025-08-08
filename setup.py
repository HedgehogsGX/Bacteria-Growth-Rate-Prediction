#!/usr/bin/env python3
"""
MicroCurve ML - Advanced Bacterial Growth Prediction
Setup configuration for package installation
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="microcurve-ml",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced Bacterial Growth Prediction using Deep Learning with Ecological Intelligence",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MicroCurve-ML",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
    },
    entry_points={
        "console_scripts": [
            "microcurve-predict=bacteria_predictor:main",
            "microcurve-train=model_trainer:main",
            "microcurve-evaluate=evaluation_report:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
    keywords=[
        "bacterial growth",
        "deep learning",
        "bioinformatics",
        "machine learning",
        "microbiology",
        "prediction",
        "tensorflow",
        "ecology",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/MicroCurve-ML/issues",
        "Source": "https://github.com/yourusername/MicroCurve-ML",
        "Documentation": "https://github.com/yourusername/MicroCurve-ML/wiki",
    },
)

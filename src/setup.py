#!/usr/bin/env python3
"""
Setup script for Cached Binance Client & Quant Analysis Tools
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements from requirements.txt
def read_requirements():
    requirements_file = this_directory / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

def get_version():
    """Get version from version file or default to 0.1.0"""
    version_file = this_directory / "VERSION"
    if version_file.exists():
        return version_file.read_text().strip()
    return "0.1.0"

setup(
    name="cached-binance-client",
    version=get_version(),
    description="A cached Binance price data retriever with quant analysis tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/cached-binance-client",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=[
        "cached_binance_client",
        "basic_quant_analysis.py",
        "simple_binance_client",
        "basic_analysis"
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "analysis": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "scikit-learn>=1.2.0",
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
        "ml": [
            "scikit-learn>=1.2.0",
            "tensorflow>=2.12.0; platform_system != 'Darwin' or platform_machine != 'arm64'",
            "tensorflow-macos>=2.12.0; platform_system == 'Darwin' and platform_machine == 'arm64'",
            "torch>=2.0.0",
            "xgboost>=1.7.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "cached-binance=cached_binance_client:main",
            "quant-analysis=quant_analysis_example:main",
            "binance-client=simple_binance_client:main",
            "binance-analysis=basic_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.toml", ".env.example"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords=[
        "binance",
        "crypto",
        "trading",
        "quantitative-analysis",
        "machine-learning",
        "bitcoin",
        "cryptocurrency",
        "data-analysis",
        "financial-data",
    ],
    project_urls={
        "Documentation": "https://github.com/yourusername/cached-binance-client",
        "Source": "https://github.com/yourusername/cached-binance-client",
        "Tracker": "https://github.com/yourusername/cached-binance-client/issues",
    },
)


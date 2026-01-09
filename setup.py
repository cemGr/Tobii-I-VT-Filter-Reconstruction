# setup.py
from pathlib import Path
from setuptools import setup, find_packages

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="tobii-ivt-filter",
    version="0.1.0",
    description="From-scratch Python implementation of Tobii's I-VT velocity-threshold filter",
    long_description=README,
    long_description_content_type="text/markdown",
    author="cemGr",
    url="https://github.com/cemGr/Tobii-I-VT-Filter-Reconstruction",
    license="MIT",
    packages=find_packages(exclude=("tests", "examples", "experiments", "notebooks")),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
    ],
    extras_require={
        "plot": ["matplotlib>=3.5"],
        "parallel": ["joblib>=1.1"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import re

from setuptools import find_packages, setup

with io.open("src/face_recognition_cam/__init__.py", "rt", encoding="utf8") as f:
    searched = re.search(r"__version__ = \"(.*?)\"", f.read())
    if searched is None:
        raise ImportError(
            "Could not find __version__ in face_recognition_cam/__init__.py"
        )
    else:
        version = searched.group(1)

setup(
    name="face_recognition_cam",
    version=version,
    author="Andrea Conti",
    author_email="andrea.conti@tutanota.com",
    description="Face recognition camera",
    long_description=open("README.md").read(),
    package_dir={"": "src"},
    packages=find_packages(),
    package_data={"": ["*.dat", "*.csv"]},
    entry_points={"console_scripts": ["facecam = face_recognition_cam.__main__:main"]},
    tests_require=["pytest"],
    install_requires=[
        "argparse",
        "numpy",
        "opencv-python",
        "mxnet",
        "dlib",
        "setuptools",
    ],
    extras_require={
        "dev": ["pytest", "pre-commit", "mypy", "flake8", "black", "isort"],
        "doc": [
            "sphinx",
            "sphinxcontrib.bibtex",
            "sphinx_gallery",
            "sphinx_press_theme",
        ],
    },
)

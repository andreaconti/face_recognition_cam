#!/usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

setup(
    name="face_recognition_cam",
    version="0.1.0",
    author="Andrea Conti",
    author_email="andrea.conti@tutanota.com",
    description="Face recognition camera",
    long_description=open('README.md').read(),
    package_dir={'': 'src'},
    packages=find_packages(),
    package_data={'': ['*.dat', '*.csv']},
    entry_points={
        'console_scripts': [
            'facecam = face_recognition_cam.__main__:main'
        ]
    },
    tests_require=[
        'pytest'
    ],
    install_requires=[
        'argparse',
        'numpy',
        'opencv-python',
        'dlib',
        'scikit-learn',
        'pkg_resources'
    ],
    extras_require={
        'dev': [
            'pytest',
            'sphinx',
            'sphinx_gallery',
            'sphinx_press_theme'
        ]
    }
)

"""
Setup of meshpy python codebase
Author: Jeff Mahler
"""
from setuptools import setup, find_packages

requirements = [
    'numpy',
    'scipy',
    'sklearn',
    'Pillow',
]

setup(name='meshpy',
    version='0.1.0',
    description='MeshPy project code',
    author='Matt Matl',
    author_email='mmatl@berkeley.edu',
    packages=find_packages(),
    install_requires=requirements
)

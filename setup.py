from distutils.core import setup
from setuptools import find_packages

setup(
    name='graspnetAPI',
    version='1.0.0',
    description='graspnet API',
    author='Hao-Shu Fang, Chenxi Wang, Minghao Gou',
    author_email='fhaoshu@gmail.com',
    url='graspnet.net',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'cython',
        'scipy',
        'transforms3d==0.3.1',
        'open3d>=0.8.0.0',
        'trimesh==3.8.4',
        'tqdm',
        'Pillow==7.2.0',
        'opencv-python',
        'pillow',
        'matplotlib',
        'pywavefront',
        'trimesh',
        'scikit-image',
        'autolab_core',
        'autolab-perception'
    ]
)

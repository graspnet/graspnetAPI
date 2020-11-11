from distutils.core import setup
from setuptools import find_packages
from setuptools.command.install import install
import os

class PostInstallCmd(install):
    def run(self):
        os.system('python -m pip install ./meshpy')
        os.system('python -m pip install ./dexnet')
        os.system('cd graspnms\npython -m pip install .')
        install.run(self)

os.system('python -m pip install cython numpy')

setup(
    name='graspnetAPI',
    version='1.1.0',
    description='graspnet API',
    author='Hao-Shu Fang, Chenxi Wang, Minghao Gou',
    author_email='gouminghao@gmail.com',
    url='https://graspnet.net',
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
        'autolab-perception',
    ],
    cmdclass={
        'install': PostInstallCmd
    }
)

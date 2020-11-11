# // Code Written by Minghao Gou

from distutils.core import setup

from Cython.Build import cythonize
import numpy

setup(
    version='1.0.0',
    requires=['cython', 'numpy'],
    name = 'grasp_nms',
    ext_modules=cythonize("grasp_nms.pyx"),
    include_dirs=[numpy.get_include()],
    install_requires=['cython']
)

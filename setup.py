
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys

# setup sequence class
ext_modules = [Extension("seq", ["seq.pyx"])]

setup(
    name = 'seq',
    cmdclass = {'build_ext': build_ext},
    include_dirs=[numpy.get_include(), '.'],
    ext_modules = ext_modules
)

# setup ribohmm
ext_modules = [Extension("ribohmm", sources=["ribohmm.pyx"])]

setup(
    name = 'ribohmm',
    cmdclass = {'build_ext': build_ext},
    include_dirs=[numpy.get_include(), '.'],
    ext_modules = ext_modules
)


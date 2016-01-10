
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys

# to run
# py setup.py build_ext --inplace
# py setup.py install

# setup bed parser
ext_modules = [Extension("ribohmm", sources=["ribohmm.pyx"], libraries=["gsl","gslcblas"],
                extra_compile_args=["-O3"])]
ext_modules = cythonize(ext_modules, compiler_directives={'embedsignature': True})

setup(
    name = 'ribohmm',
    author = 'Anil Raj',
    version = '1.0',
    cmdclass = {'build_ext': build_ext},
    include_dirs=[numpy.get_include(), '.'],
    ext_modules = ext_modules
)

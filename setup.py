from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='3sat',
    ext_modules=cythonize('three_sat/algorithms/*.pyx', language_level=3, include_dirs=[numpy.get_include()]),
    include_dirs=[numpy.get_include()],
    install_requires=[
        'Cython',
        'NumPy',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest>=3.0']
)
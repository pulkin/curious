#!/usr/bin/env python3
from distutils.core import setup, Extension
import numpy

setup(
    name='curious',
    version='0.0.0',
    author='Artem Pulkin',
    description='Feature-driven function sampling',
    ext_modules=[Extension('cutil', sources=['c/cutil.c'], include_dirs=[numpy.get_include()])],
    scripts=['scripts/curious.py'],
)

#!/usr/bin/env python3
from distutils.core import setup, Extension

setup(
    name='curious',
    version='0.0.0',
    author='Artem Pulkin',
    description='Feature-driven function sampling',
    ext_modules=[Extension('cutil', sources=['c/cutil.c'])],
    scripts=['scripts/curious.py'],
    install_requires=['numpy', 'scipy', 'matplotlib', 'imageio'],
)

#!/usr/bin/env python3
from setuptools import setup, find_packages
from setuptools.extension import Extension
import numpy


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='curious',
    version='0.1.0',
    author='curious contributors',
    author_email='gpulkin@gmail.com',
    url='https://github.com/pulkin/curious',
    packages=find_packages(),
    data_files=["requirements.txt"],
    description='Feature-driven function sampling',
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    install_requires=requirements,
    ext_modules=[Extension('cutil', sources=['c/cutil.c'], include_dirs=[numpy.get_include()])],
)

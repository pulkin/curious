[![Build Status](https://dev.azure.com/gpulkin/curious/_apis/build/status/pulkin.curious?branchName=master)](https://dev.azure.com/gpulkin/curious/_build/latest?definitionId=1&branchName=master)

curious - parallel function sampling
====================================

Samples multidimensional scalar functions by observing curvature features.
Uses arbitrary executables as a target function.
Based on `scipy` triangulation `scipy.spatial.Delaunay`.

Inspired by [python-adaptive](https://github.com/python-adaptive/adaptive).

Install
-------

TBD

Custom build
------------

Perform these 3 steps inside the repository folder.

1. Install build system

    ```commandline
    pip install build
    ```

2. Build

    ```commandline
    python -m build
    ```

3. Install

    ```commandline
    pip install dist/*.whl
    ```

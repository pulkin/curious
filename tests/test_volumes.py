from unittest import TestCase

import numpy
from numpy import testing

from cutil import simplex_volumes as sv, simplex_volumes_n as svn, __file__


def simplex_volumes(a, b):
    return sv(numpy.asanyarray(a, dtype=float), numpy.asanyarray(b, dtype=numpy.int32))


def simplex_volumes_n(a, b, c, d):
    return svn(numpy.asanyarray(a, dtype=float), numpy.asanyarray(b, dtype=numpy.int32),
               numpy.asanyarray(c, dtype=numpy.int32), numpy.asanyarray(d, dtype=numpy.int32))


class TestVolumes(TestCase):
    def test_volumes_1(self):
        testing.assert_allclose(simplex_volumes([[0], [1], [3], [5]], [[0, 1], [1, 2], [2, 3]]), [1, 2, 2])

    def test_volumes_2(self):
        testing.assert_allclose(simplex_volumes([[0, 0], [0, 1], [1, 0]], [[0, 1, 2]]), [.5])
        testing.assert_allclose(simplex_volumes([[0, 0], [0, 1], [1, 0], [1, 1]], [[0, 1, 2], [1, 2, 3]]), [.5, .5])

    def test_volumes_3(self):
        testing.assert_allclose(simplex_volumes([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], [[0, 1, 2, 3]]), [1./6])

    def test_volumes_4(self):
        testing.assert_allclose(simplex_volumes([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 0, -1, 0],
            [0, -1, 0, 0],
            [-1, 0, 0, 0],
        ], [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8]]), [1./24, 1./24])

    def test_volumes_n_1(self):
        testing.assert_allclose(
            simplex_volumes_n(
                [[0, 0], [1, 0], [2, 1], [3, 1]],
                [[0, 1], [1, 2], [2, 3]],
                [0, 1, 3, 5, 6],
                [1, 0, 2, 1, 3, 2],
            ),
            [.5, 1, .5],
        )

    def test_volumes_n_2(self):
        testing.assert_allclose(
            simplex_volumes_n(
                [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]],
                [[0, 1, 2], [1, 2, 3]],
                [0, 2, 4, 6, 8],
                [1, 2, 0, 3, 0, 3, 1, 2],
            ),
            [.5 / 3, .5 / 3],
        )

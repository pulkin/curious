from curious.util import load

from unittest import TestCase
import sys
import os
from pathlib import Path
import subprocess
import tempfile
import numpy
import time
from copy import deepcopy


def assert_close_points(ref, test, **kwargs):
    """Checks if point arrays are close"""
    def take_column(data, col):
        return list(i[col] for i in data)

    numpy.testing.assert_almost_equal(take_column(ref, 0), take_column(test, 0), **kwargs)  # x
    numpy.testing.assert_almost_equal(take_column(ref, 1), take_column(test, 1), **kwargs)  # y
    numpy.testing.assert_equal(take_column(ref, 2), take_column(test, 2), **kwargs)  # tag


class TestScript(TestCase):
    def run_curious(self, target, *options, debug=False, ignore_warnings=False):
        this = Path(os.path.dirname(os.path.realpath(__file__)))
        if "--save" in options:
            fl = None
            for k, v in zip(options[:-1], options[1:]):
                if k == "--save":
                    fl = v
            if fl is None:
                raise ValueError("No argument followed by '--save'")
        else:
            _, fl = tempfile.mkstemp(suffix=".json")
        p = subprocess.Popen(
            ("python", "-m", "curious", this / "functions" / target, '--save', fl) + tuple(options),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        out, err = p.communicate()
        out = out.decode("utf-8")
        err = err.decode("utf-8")
        if p.returncode != 0 or debug:
            print(out, file=sys.stdout)
            print(err, file=sys.stderr)
        if p.returncode != 0:
            raise RuntimeError("Curious returned {:d}".format(p.returncode))
        tmp_folder = Path('.curious')
        if tmp_folder.exists():
            raise RuntimeError("Temp folder {} was created".format(tmp_folder))

        if not self.debug:
            self.assertEqual(out, "")
            if not ignore_warnings:
                self.assertEqual(err, "")

        with open(fl, 'r') as f:
            return load(f)

    def test_lim_case_0(self):
        _x1 = 1. / abs(2.**.5 - 0.5 - 0.1j) ** 2
        _x2 = 1. / abs(2.**.5 / 3 - 0.5 - 0.1j) ** 2 - 1./9
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5')
        assert_close_points(data["sampler"]["points"], [
            [[-1.0, -1.0], [_x1 + 1], "✓"], [[-1.0, 1.0], [_x1 - 1], "✓"],
            [[1.0, -1.0], [_x1 - 1], "✓"], [[1.0, 1.0], [_x1 + 1], "✓"],
            [[-1./3, 1./3], [_x2], "✓"],
        ])
        self.assertEqual(data["sampler"]["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)

    def test_lim_case_1(self):
        data = self.run_curious("1_2d_step.py", '-1 1, -1 1', '-l', 'eval:5')
        assert_close_points(data["sampler"]["points"], [
            [[-1.0, -1.0], [-1.0], "✓"], [[-1.0, 1.0], [1.0], "✓"],
            [[1.0, -1.0], [-1.0], "✓"], [[1.0, 1.0], [1.0], "✓"],
            [[-1./3, 1./3], [-3.**.5 / 2 + 1], "✓"],
        ])
        self.assertEqual(data["sampler"]["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)

    def test_lim_case_2(self):
        data = self.run_curious("2_2d_divergence.py", '-1 1, -1 1', '-l', 'eval:5')
        assert_close_points(data["sampler"]["points"], [
            [[-1.0, -1.0], [-2.0], "✓"], [[-1.0, 1.0], [0.0], "✓"],
            [[1.0, -1.0], [0.0], "✓"], [[1.0, 1.0], [2.0], "✓"],
            [[-1./3, 1./3], [0.0], "✓"],
        ])
        self.assertEqual(data["sampler"]["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)

    def test_lim_case_4(self):
        data = self.run_curious("4_1d_gaussian.py", '0 1', '-l', 'eval:5')
        assert_close_points(data["sampler"]["points"], [
            [[0.0], [1.0], "✓"], [[1.0], [numpy.exp(-1)], "✓"],
            [[0.5], [numpy.exp(-0.25)], "✓"], [[0.25], [numpy.exp(-1./16)], "✓"],
            [[0.375], [numpy.exp(-0.375 ** 2)], "✓"],
        ])
        self.assertEqual(data["sampler"]["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)

    def test_lim_eval(self):
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:30')
        self.assertEqual(data["sampler"]["points"].shape, (30,))
        self.assertEqual(data["sampler"]["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)

    def test_lim_time(self):
        t = time.time()
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'time:00:00:05')
        t = time.time() - t
        self.assertEqual(len(data["sampler"]["points"].dtype.names), 4)
        self.assertEqual(data["sampler"]["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)
        self.assertLess(t, 7)
        self.assertGreater(t, 5)

    def test_depth(self):
        _x1 = 1. / abs(2.**.5 - 0.5 - 0.1j) ** 2
        _x2 = 1. / abs(2.**.5 / 3 - 0.5 - 0.1j) ** 2 - 1./9
        _x3 = 1. / abs(26.**.5 / 9 - 0.5 - 0.1j) ** 2 + 5./81
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:6', "--depth", "2")
        assert_close_points(data["sampler"]["points"], [
            [[-1.0, -1.0], [_x1 + 1], "✓"], [[-1.0, 1.0], [_x1 - 1], "✓"],
            [[1.0, -1.0], [_x1 - 1], "✓"], [[1.0, 1.0], [_x1 + 1], "✓"],
            [[-1./3, 1./3], [_x2], "✓"], [[-1./9, -5./9], [_x3], "✓"],
        ])
        self.assertEqual(data["sampler"]["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)

    def test_max_fails(self):
        with self.assertRaises(RuntimeError):
            self.run_curious("3_exit_code.py", '-1 1, -1 1', '-l', 'eval:1')
        data = self.run_curious("3_exit_code.py", '-1 1, -1 1', '-l', 'eval:5', "--fail-limit", "5")
        assert_close_points(data["sampler"]["points"], [
            [[-1.0, -1.0], [float("nan")], "✓"], [[-1.0, 1.0], [float("nan")], "✓"],
            [[1.0, -1.0], [float("nan")], "✓"], [[1.0, 1.0], [float("nan")], "✓"],
            [[-1./3, 1./3], [float("nan")], "✓"],
        ])
        self.assertEqual(data["sampler"]["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)
        with self.assertRaises(RuntimeError):
            self.run_curious("3_exit_code.py", '-1 1, -1 1', '-l', 'eval:5', "--fail-limit", "4")

    def test_tweaks(self):
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', "--snap-threshold", "0")
        self.assertEqual(data["sampler"]["snap_threshold"], 0)
        data = self.run_curious("2_2d_divergence.py", '-1 1, -1 1', '-l', 'eval:5', "--nan-threshold", "0")
        self.assertEqual(data["nan_threshold"], 0)
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', "--volume-ratio", "1")
        self.assertEqual(data["volume_ratio"], 1)

    def test_restart(self):
        fl = tempfile.mkstemp(suffix=".json")[1]
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', "--save", fl)
        self.assertEqual(numpy.array(data["sampler"]["points"]).shape, (5,))
        self.assertEqual(len(numpy.array(data["sampler"]["points"]).dtype.names), 4)
        data_new = self.run_curious("0_nd_circle_feature.py", '-1 2, -1 1', '-l', 'eval:5', '--load', fl, '--save', fl,
                                    "--snap-threshold", "0.1", "--no-rescale", "--nan-threshold", "0.1",
                                    "--volume-ratio", "30")

        self.assertEqual(numpy.array(data_new["sampler"]["points"]).shape, (10,))
        self.assertEqual(len(numpy.array(data_new["sampler"]["points"]).dtype.names), 4)

        numpy.testing.assert_equal(data_new["sampler"]["points"][:5], data["sampler"]["points"])
        del data["sampler"]["points"]
        data_test = deepcopy(data_new)
        del data_test["sampler"]["points"]
        data_test["sampler"]["snap_threshold"] = 0.5
        data_test["sampler"]["rescale"] = True
        data_test["nan_threshold"] = 0.5
        data_test["volume_ratio"] = 10
        self.assertEqual(data, data_test)

        point_coordinates = numpy.array([i[0] for i in data_new["sampler"]["points"]])
        d = numpy.linalg.norm(point_coordinates - [[2, -1]], axis=-1)
        self.assertEqual((d == 0).sum(), 1)

        d = numpy.linalg.norm(point_coordinates - [[2, 1]], axis=-1)
        self.assertEqual((d == 0).sum(), 1)

        self.assertEqual(data_new["sampler"]["rescale"], False)
        self.assertEqual(data_new["sampler"]["snap_threshold"], 0.1)
        self.assertEqual(data_new["nan_threshold"], 0.1)
        self.assertEqual(data_new["volume_ratio"], 30)

    def test_4d(self):
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1, -1 1, -1 1', '-l', 'eval:16')
        self.assertEqual(len(data["sampler"]["points"]), 16)

    def test_two_component(self):
        def f1(x):
            return numpy.exp(-(x - 0.5) ** 2)

        def f2(x):
            return numpy.exp(-(x + 0.5) ** 2) / 2

        data = self.run_curious("5_1d_2d_gaussian.py", '-1 1', '-l', 'eval:5', '-n', '2')
        assert_close_points(data["sampler"]["points"], [
            ([-1], [f1(-1), f2(-1)], "✓"),
            ([1], [f1(1), f2(1)], "✓"),
            ([0], [f1(0), f2(0)], "✓"),
            ([-0.5], [f1(-0.5), f2(-0.5)], "✓"),
            ([-0.25], [f1(-0.25), f2(-0.25)], "✓"),
        ])

    def test_two_parameter_two_component(self):
        self.run_curious("6_nd_2d_two_features.py", '-1 1, -1 1', '-l', 'eval:5', '-n', '2')

    def test_on_plot_update(self):
        result = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:30', '--on-update',
                                  'cat', '-v', debug=True)

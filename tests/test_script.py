from unittest import TestCase
import matplotlib

import sys
import os
from pathlib import Path
import subprocess
import tempfile
import json
import numpy
import time
import imageio


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
            fl = tempfile.mkstemp(suffix=".json")[1]
        executable = "curious.py" if "CURIOUS_TEST_INSTALL" in os.environ else this / ".." / "scripts" / "curious.py"
        p = subprocess.Popen(
            (executable, this / "functions" / target, '--save', fl) + tuple(options),
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
            return json.load(f)

    def test_lim_case_0(self):
        _x1 = 1. / abs(2.**.5 - 0.5 - 0.1j) ** 2
        _x2 = 1. / abs(2.**.5 / 3 - 0.5 - 0.1j) ** 2 - 1./9
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5')
        self.assertEqual(data["dims"], 2)
        numpy.testing.assert_almost_equal(data["points"], [
            [-1.0, -1.0, _x1 + 1, 2.0], [-1.0, 1.0, _x1 - 1, 2.0],
            [1.0, -1.0, _x1 - 1, 2.0], [1.0, 1.0, _x1 + 1, 2.0],
            [-1./3, 1./3, _x2, 2.0],
        ])
        self.assertEqual(data["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)
        meta = numpy.array(data["meta"])
        self.assertEqual(meta.shape, (5, 2))
        # Output stream
        out_num = tuple(float(i[i.index("\n")+1:-1]) for i in meta[:, 0])
        numpy.testing.assert_almost_equal(out_num, (_x1 + 1, _x1 - 1, _x1 - 1, _x1 + 1, _x2))
        # Error stream
        numpy.testing.assert_equal(meta[:, 1], ('',) * 5)

    def test_lim_case_1(self):
        data = self.run_curious("1_2d_step.py", '-1 1, -1 1', '-l', 'eval:5')
        self.assertEqual(data["dims"], 2)
        numpy.testing.assert_almost_equal(data["points"], [
            [-1.0, -1.0, -1.0, 2.0], [-1.0, 1.0, 1.0, 2.0],
            [1.0, -1.0, -1.0, 2.0], [1.0, 1.0, 1.0, 2.0],
            [-1./3, 1./3, -3.**.5 / 2 + 1, 2.0],
        ])
        self.assertEqual(data["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)

    def test_lim_case_2(self):
        data = self.run_curious("2_2d_divergence.py", '-1 1, -1 1', '-l', 'eval:5')
        self.assertEqual(data["dims"], 2)
        numpy.testing.assert_almost_equal(data["points"], [
            [-1.0, -1.0, -2.0, 2.0], [-1.0, 1.0, 0.0, 2.0],
            [1.0, -1.0, 0.0, 2.0], [1.0, 1.0, 2.0, 2.0],
            [-1./3, 1./3, 0.0, 2.0],
        ])
        self.assertEqual(data["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)

    def test_lim_case_4(self):
        data = self.run_curious("4_1d_gaussian.py", '0 1', '-l', 'eval:5')
        self.assertEqual(data["dims"], 1)
        numpy.testing.assert_almost_equal(data["points"], [
            [0.0, 1.0, 2.0], [1.0, numpy.exp(-1), 2.0],
            [0.5, numpy.exp(-0.25), 2.0], [0.25, numpy.exp(-1./16), 2.0],
            [0.375, numpy.exp(-0.375 ** 2), 2.0],
        ])
        self.assertEqual(data["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)

    def test_lim_eval(self):
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:30')
        self.assertEqual(data["dims"], 2)
        self.assertEqual(numpy.array(data["points"]).shape, (30, 4))
        self.assertEqual(data["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)

    def test_lim_time(self):
        t = time.time()
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'time:00:00:05')
        t = time.time() - t
        self.assertEqual(data["dims"], 2)
        self.assertEqual(numpy.array(data["points"]).shape[1], 4)
        self.assertEqual(data["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)
        self.assertLess(t, 7)
        self.assertGreater(t, 5)

    def test_depth(self):
        _x1 = 1. / abs(2.**.5 - 0.5 - 0.1j) ** 2
        _x2 = 1. / abs(2.**.5 / 3 - 0.5 - 0.1j) ** 2 - 1./9
        _x3 = 1. / abs(26.**.5 / 9 - 0.5 - 0.1j) ** 2 + 5./81
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:6', "--depth", "2")
        self.assertEqual(data["dims"], 2)
        numpy.testing.assert_almost_equal(data["points"], [
            [-1.0, -1.0, _x1 + 1, 2.0], [-1.0, 1.0, _x1 - 1, 2.0],
            [1.0, -1.0, _x1 - 1, 2.0], [1.0, 1.0, _x1 + 1, 2.0],
            [-1./3, 1./3, _x2, 2.0], [-1./9, -5./9, _x3, 2.0],
        ])
        self.assertEqual(data["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)

    def test_max_fails(self):
        with self.assertRaises(RuntimeError):
            self.run_curious("3_exit_code.py", '-1 1, -1 1', '-l', 'eval:1')
        data = self.run_curious("3_exit_code.py", '-1 1, -1 1', '-l', 'eval:5', "--max-fails", "5")
        self.assertEqual(data["dims"], 2)
        numpy.testing.assert_almost_equal(data["points"], [
            [-1.0, -1.0, 0, 1.0], [-1.0, 1.0, 0, 1.0],
            [1.0, -1.0, 0, 1.0], [1.0, 1.0, 0, 1.0],
            [-1./3, 1./3, 0, 1.0],
        ])
        self.assertEqual(data["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)
        with self.assertRaises(RuntimeError):
            self.run_curious("3_exit_code.py", '-1 1, -1 1', '-l', 'eval:5', "--max-fails", "4")

    def test_gif(self):
        size_inches = matplotlib.rcParams["figure.figsize"]
        dpi = matplotlib.rcParams["figure.dpi"]
        fl = tempfile.mkstemp(suffix=".gif")[1]
        self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', '--plot', fl, ignore_warnings=True)
        self.assertTrue(os.path.isfile(fl))
        data = imageio.mimread(fl)
        self.assertEqual(len(data), 6)
        for i in data:
            self.assertEqual(i.shape, (int(size_inches[1] * dpi), int(size_inches[0] * dpi), 4))

    def test_tweaks(self):
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', "--snap-threshold", "0")
        self.assertEqual(data["snap_threshold"], 0)
        data = self.run_curious("2_2d_divergence.py", '-1 1, -1 1', '-l', 'eval:5', "--nan-threshold", "0")
        self.assertEqual(data["nan_threshold"], 0)
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', "--volume-ratio", "1")
        self.assertEqual(data["volume_ratio"], 1)

    def test_restart(self):
        fl = tempfile.mkstemp(suffix=".json")[1]
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', "--save", fl)
        self.assertEqual(numpy.array(data["points"]).shape, (5, 4))
        self.assertEqual(numpy.array(data["meta"]).shape, (5, 2))
        data_new = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', '--load', fl, '--save', fl)
        self.assertEqual(numpy.array(data_new["points"]).shape, (10, 4))
        self.assertEqual(numpy.array(data_new["meta"]).shape, (10, 2))
        data_test = data_new.copy()
        data_test["points"] = data_test["points"][:5]
        data_test["meta"] = data_test["meta"][:5]
        self.assertEqual(data, data_test)

    def test_4d(self):
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1, -1 1, -1 1', '-l', 'eval:16')
        self.assertEqual(numpy.array(data["points"]).shape, (16, 6))

    def test_two_component(self):
        def f1(x):
            return numpy.exp(-(x - 0.5) ** 2)

        def f2(x):
            return numpy.exp(-(x + 0.5) ** 2) / 2

        data = self.run_curious("5_1d_2d_gaussian.py", '-1 1', '-l', 'eval:5', '--plot', '-n', '2')
        self.assertEqual(data["dims"], 1)
        self.assertEqual(data["dims_f"], 2)
        numpy.testing.assert_equal(numpy.array(data["points"]), [
            (-1, f1(-1), f2(-1), 2.0),
            (1, f1(1), f2(1), 2.0),
            (0, f1(0), f2(0), 2.0),
            (-0.5, f1(-0.5), f2(-0.5), 2.0),
            (-0.25, f1(-0.25), f2(-0.25), 2.0),
        ])

    def test_two_parameter_two_component(self):
        self.run_curious("6_nd_2d_two_features.py", '-1 1, -1 1', '-l', 'eval:5', '--plot', '-n', '2')

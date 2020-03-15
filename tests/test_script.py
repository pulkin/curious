from unittest import TestCase

import sys
import os
from pathlib import Path
import subprocess
import tempfile
import json
import numpy


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
        _x1 = 1. / abs(2.**.5 - 0.5 - 0.1j) ** 2 + 1
        _x2 = 1. / abs(2.**.5 / 3 - 0.5 - 0.1j) ** 2 + 1
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5')
        self.assertEqual(data["dims"], 2)
        numpy.testing.assert_almost_equal(data["points"], [
            [-1.0, -1.0, _x1, 2.0], [-1.0, 1.0, _x1, 2.0],
            [1.0, -1.0, _x1, 2.0], [1.0, 1.0, _x1, 2.0],
            [-1./3, 1./3, _x2, 2.0],
        ])
        self.assertEqual(data["snap_threshold"], 0.5)
        self.assertEqual(data["nan_threshold"], 0.5)
        self.assertEqual(data["volume_ratio"], 10)

    def test_lim_case_1(self):
        self.run_curious("1_2d_step.py", '-1 1, -1 1', '-l', 'eval:5')

    def test_lim_case_2(self):
        self.run_curious("2_2d_divergence.py", '-1 1, -1 1', '-l', 'eval:5')

    def test_lim_case_4(self):
        self.run_curious("4_1d_gaussian.py", '-1 1', '-l', 'eval:5')

    def test_lim_eval(self):
        self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:30')

    def test_lim_time(self):
        self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'time:00:00:05')

    def test_depth(self):
        self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', "--depth", "2")

    def test_max_fails(self):
        with self.assertRaises(RuntimeError):
            self.run_curious("3_exit_code.py", '-1 1, -1 1', '-l', 'eval:1')
        self.run_curious("3_exit_code.py", '-1 1, -1 1', '-l', 'eval:5', "--max-fails", "5")
        with self.assertRaises(RuntimeError):
            self.run_curious("3_exit_code.py", '-1 1, -1 1', '-l', 'eval:5', "--max-fails", "4")

    def test_plot(self):
        self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', '--plot', ignore_warnings=True)

    def test_gif(self):
        fl = tempfile.mkstemp(suffix=".gif")[1]
        self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', '--plot', fl, ignore_warnings=True)
        self.assertTrue(os.path.isfile(fl))

    def test_tweaks(self):
        self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', "--snap-threshold", "0")
        self.run_curious("2_2d_divergence.py", '-1 1, -1 1', '-l', 'eval:5', "--nan-threshold", "0")
        self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', "--volume-ratio", "1")

    def test_restart(self):
        fl = tempfile.mkstemp(suffix=".json")[1]
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', "--save", fl)
        self.assertTrue(os.path.isfile(fl))
        pts = numpy.array(data["points"])
        self.assertEqual(pts.shape, (5, 4))
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', '--load', fl, '--save', fl)
        pts = numpy.array(data["points"])
        self.assertEqual(pts.shape, (10, 4))

    def test_4d(self):
        data = self.run_curious("0_nd_circle_feature.py", '-1 1, -1 1, -1 1, -1 1', '-l', 'eval:16')
        pts = numpy.array(data["points"])
        self.assertEqual(pts.shape, (16, 6))

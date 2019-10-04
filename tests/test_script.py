from unittest import TestCase

import sys
import os
from pathlib import Path
import subprocess
import tempfile


class TestScript(TestCase):
    def run_curious(self, target, *options, debug=False, ignore_warnings=False):
        this = Path(os.path.dirname(os.path.realpath(__file__)))
        p = subprocess.Popen(
            (this / ".." / "scripts" / "curious.py", this / "functions" / target,) + tuple(options),
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
        self.assertEqual(out, "")
        if not ignore_warnings:
            self.assertEqual(err, "")

    def test_lim_case_0(self):
        self.run_curious("0_2d_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5')

    def test_lim_case_1(self):
        self.run_curious("1_2d_step.py", '-1 1, -1 1', '-l', 'eval:5')

    def test_lim_case_2(self):
        self.run_curious("2_2d_divergence.py", '-1 1, -1 1', '-l', 'eval:5')

    def test_lim_eval(self):
        self.run_curious("0_2d_circle_feature.py", '-1 1, -1 1', '-l', 'eval:30')

    def test_lim_time(self):
        self.run_curious("0_2d_circle_feature.py", '-1 1, -1 1', '-l', 'time:00:00:05')

    def test_depth(self):
        self.run_curious("0_2d_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', "--depth", "2")

    def test_max_fails(self):
        with self.assertRaises(RuntimeError):
            self.run_curious("3_exit_code.py", '-1 1, -1 1', '-l', 'eval:1')
        self.run_curious("3_exit_code.py", '-1 1, -1 1', '-l', 'eval:5', "--max-fails", "5")
        with self.assertRaises(RuntimeError):
            self.run_curious("3_exit_code.py", '-1 1, -1 1', '-l', 'eval:5', "--max-fails", "4")

    def test_plot(self):
        self.run_curious("0_2d_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', '--plot', ignore_warnings=True)

    def test_gif(self):
        fl = tempfile.mkstemp(suffix=".gif")[1]
        self.run_curious("0_2d_circle_feature.py", '-1 1, -1 1', '-l', 'eval:5', '--gif', fl, ignore_warnings=True)
        self.assertTrue(os.path.isfile(fl))

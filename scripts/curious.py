#!/usr/bin/env python3
from cutil import simplex_volumes, simplex_volumes_n

import imageio
from matplotlib import pyplot
from matplotlib.tri import Triangulation
import numpy
from scipy.spatial import Delaunay

from itertools import product
from collections.abc import Iterable
from warnings import warn
import argparse
import inspect
from datetime import timedelta, datetime
import subprocess
import re
import sys
import os
from pathlib import Path
import json


FLAG_PENDING = 0
FLAG_RUNNING = 1
FLAG_DONE = 2


class Guide(Iterable):
    def __init__(self, dims, points=None, snap_threshold=None, default_value=0, default_flag=FLAG_PENDING):
        """
        Guides the sampling.
        Args:
            dims (int): the dimensionality;
            points (Iterable): data points;
            snap_threshold (float): a threshold to snap to triangle/simplex faces;
        """
        self.dims = dims
        if points is None:
            self.__data__ = numpy.empty((0, dims + 1), dtype=float)
        else:
            points = numpy.array(tuple(points))
            self.data = numpy.zeros((len(points), dims+2), dtype=float)
            self.values[:] = default_value
            self.flags[:] = default_flag
            self.data[:, :points.shape[1]] = points
        self.snap_threshold = snap_threshold
        self.tri = self.maybe_triangulate()

    @classmethod
    def from_bounds(cls, bounds, **kwargs):
        """
        Constructs a guide starting from multidimensional box bounds.
        Args:
            bounds (tuple): bounds defining the box;
            **kwargs: keyword arguments to constructor;

        Returns:
            A new guide.
        """
        return cls(len(bounds), product(*bounds), **kwargs)

    @property
    def coordinates(self):
        return self.data[:, :self.dims]

    @property
    def values(self):
        return self.data[:, self.dims]

    @property
    def flags(self):
        return self.data[:, self.dims + 1]

    def __getstate__(self):
        return dict(dims=self.dims, points=self.data.tolist(), snap_threshold=self.snap_threshold)

    def __setstate__(self, state):
        self.__init__(**state)

    def to_json(self):
        """To json representation"""
        return self.__getstate__()

    @classmethod
    def from_json(cls, data):
        """Object from json representation"""
        return cls(**data)

    @property
    def m_pending(self):
        """Pending points to process"""
        return numpy.where(self.flags == FLAG_PENDING)[0]

    @property
    def m_running(self):
        return numpy.where(self.flags == FLAG_RUNNING)[0]

    @property
    def m_done(self):
        """Pending points to process"""
        return numpy.where(self.flags == FLAG_DONE)[0]

    @property
    def m_done_numeric(self):
        """Final points with numeric values"""
        return numpy.where((self.flags == FLAG_DONE) * (numpy.logical_not(numpy.isnan(self.values))))[0]

    @property
    def m_done_nan(self):
        """Final points with numeric values"""
        return numpy.where((self.flags == FLAG_DONE) * (numpy.isnan(self.values)))[0]

    def maybe_triangulate(self):
        """Computes a new triangulation"""
        if len(self.data) >= 2 ** self.dims:
            self.tri = Delaunay(self.coordinates, qhull_options='Qz')
        else:
            self.tri = None
        return self.tri

    def add(self, *p):
        """
        Adds a new point and triangulates.
        Args:
            p (tuple): the point to add;

        Returns:
            The index of the point added.
        """
        if not self.dims <= len(p) < self.dims + 3:
            raise ValueError("Wrong point data length {:d}, expected {:d} <= len(p) < {:d}".format(
                len(p), self.dims, self.dims + 3))
        self.data = numpy.append(self.data, [tuple(p) + (0, FLAG_PENDING)[len(p) - self.dims:]], axis=0)
        self.maybe_triangulate()
        return len(self.data) - 1

    def get_lims(self):
        """Data limits"""
        return tuple(zip(
            numpy.min(self.coordinates, axis=0),
            numpy.max(self.coordinates, axis=0))
        )

    def priority(self):
        """
        Priority for sampling each of the triangles.
        Returns:
            An array with priorities: arbitrary real numbers.
        """
        raise NotImplementedError

    def __center__(self, simplex):
        """
        This function picks a 'center' point in a triangle/simplex. It also prevents triangulation
        from squeezing at borders.
        Args:
            simplex (numpy.array): simplex points;

        Returns:
            A center point in barycentric coordinates.
        """
        center = numpy.mean(simplex, axis=0)
        dst = numpy.linalg.norm(simplex - center[numpy.newaxis, :], axis=-1)
        mean_dst = numpy.mean(dst)
        mask = numpy.asanyarray(dst > self.snap_threshold * mean_dst, dtype=float)
        if mask.sum() < 2:
            mask[numpy.argsort(dst)[:-3:-1]] = .5
        else:
            mask /= mask.sum()
        return mask

    def __next__(self):
        """Picks a next point index to compute based on priority."""
        # First, preset points
        for i in self.m_pending:
            self.flags[i] = FLAG_RUNNING
            return i

        if self.tri is None:
            raise StopIteration

        priority = self.priority()
        # Second, nans
        nans = numpy.where(numpy.isnan(priority))[0]
        if len(nans) > 0:
            simplex = nans[0]
        else:
            # Largest volumes otherwise
            simplex = numpy.argmax(priority)
        simplex_coordinates = self.tri.points[self.tri.simplices[simplex], :]
        center_bcc = self.__center__(simplex_coordinates)
        center_coordinate = center_bcc @ simplex_coordinates
        stub_value = center_bcc @ self.values[self.tri.simplices[simplex], ...]
        return self.add(*center_coordinate, stub_value, FLAG_RUNNING)

    def __iter__(self):
        return self

    def suggest(self, n=None):
        """
        Suggests one or more points to calculate.
        Args:
            n (int): the number of points to calculate;

        Returns:
            A list of points.
        """
        if n is None:
            return next(self)
        else:
            return list(i for _, i in zip(range(n), self))


class UniformGuide(Guide):
    def get_simplex_volumes(self):
        """Calculates simplexes' volumes"""
        return simplex_volumes(self.tri.points, self.tri.simplices)

    def priority(self):
        """Calculates priority for sampling of each triangle"""
        return self.get_simplex_volumes()


class CurvatureGuide(UniformGuide):
    def __init__(self, dims, points, snap_threshold=None, nan_threshold=None, volume_ratio=None):
        """
        Guides the sampling.
        Args:
            dims (int): the dimensionality;
            points (Iterable): data points;
            snap_threshold (float): a threshold to snap to triangle/simplex faces;
            nan_threshold (float): a critical value of NaNs to fallback to uniform sampling;
            volume_ratio (float): the ratio between the largest volume present and the
            smallest volume to sample;
        """
        super().__init__(dims, points, snap_threshold)
        self.nan_threshold = nan_threshold
        self.volume_ratio = volume_ratio

    def __getstate__(self):
        return {**super().__getstate__(), **dict(
            nan_threshold=self.nan_threshold,
            volume_ratio=self.volume_ratio,
        )}

    def get_curvature_measure(self):
        """Calculates the measure of curvature"""
        return simplex_volumes_n(self.data[:, :self.dims+1], self.tri.simplices, *self.tri.vertex_neighbor_vertices)

    def priority(self):
        """Calculates priority for sampling of each triangle"""
        result = self.get_curvature_measure()

        if self.nan_threshold is not None:
            if numpy.isnan(result).sum() > self.nan_threshold * len(result):
                return self.get_simplex_volumes()

        if self.volume_ratio is not None:
            volumes = self.get_simplex_volumes()
            discard_mask = volumes / max(volumes) < 1./self.volume_ratio
            result[discard_mask] = 0

        return result


class PointProcessPool:
    def __init__(self, target, guide, float_regex=r"([-+]?[0-9]+\.?[0-9]*(?:[eEdD][-+]?[0-9]*)?)|(nan)",
                 encoding="utf-8"):
        self.target = target
        self.guide = guide
        self.compiled_float_regex = re.compile(float_regex)
        self.encoding = encoding
        self.processes = []
        self.__failed__ = self.__succeeded__ = 0

    @property
    def failed(self):
        """Failed runs"""
        return self.__failed__

    @property
    def succeeded(self):
        return self.__succeeded__

    @property
    def running(self):
        return len(self.processes)

    @property
    def completed(self):
        return self.failed + self.succeeded

    def new(self, p):
        self.processes.append((p, subprocess.Popen(
            (self.target,) + tuple(map("{:.12e}".format, self.guide.coordinates[p])),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )))

    def sweep(self):
        transaction = []
        for item in self.processes:
            point, process = item
            if process.poll() is not None:
                transaction.append(item)

                out, err = process.communicate()
                out = out.decode(self.encoding)
                err = err.decode(self.encoding)
                if len(err) > 0:
                    print(err, file=sys.stderr)

                if process.returncode:
                    self.__failed__ += 1

                else:
                    self.__succeeded__ += 1
                    match = None
                    for match in self.compiled_float_regex.finditer(out):
                        pass

                    if match is not None:
                        self.guide.values[point] = float(match.group())
                        self.guide.flags[point] = FLAG_DONE

        for i in transaction:
            self.processes.remove(i)

        return len(transaction)


class PlotView2D:
    def __init__(self, guide, window=False, record_animation=False, **kwargs):
        """
        Plots of the 2D sampling.
        Args:
            guide (Guide): the guide to view;
            kwargs: keyword arguments to `pyplot.subplots`;
        """
        if guide.dims != 2:
            raise ValueError("This view is only for 2D guides")
        self.guide = guide
        self.fig, self.ax = pyplot.subplots(**kwargs)
        self.axes_limits = guide.get_lims()
        self.color_bar = True
        pyplot.ion()
        if window:
            pyplot.show()

        if record_animation:
            self.animation_data = []
        else:
            self.animation_data = None
        self.closed_flag = False
        self.fig.canvas.mpl_connect('close_event', self.__listen_close__)

    def __listen_close__(self, event):
        """Listens for closing event."""
        self.closed_flag = True

    def update(self):
        """Updates the figure."""
        tri = self.guide.tri
        values = self.guide.values
        if tri is None:
            raise ValueError("Triangulation is None")

        self.ax.clear()
        color_plot = self.ax.tripcolor(
            Triangulation(*tri.points.T, triangles=tri.simplices),
            values,
            vmin=numpy.nanmin(values),
            vmax=numpy.nanmax(values),
        )

        if self.color_bar is True:
            self.color_bar = pyplot.colorbar(color_plot)
        elif self.color_bar in (False, None):
            pass
        else:
            self.color_bar.on_mappable_changed(color_plot)

        running = self.guide.coordinates[self.guide.m_running, :]
        if len(running) > 0:
            self.ax.scatter(*running.T, facecolors='none', edgecolors="white")

        done = self.guide.coordinates[self.guide.m_done_numeric, :]
        if len(done) > 0:
            self.ax.scatter(*done.T, color="white", s=3)

        nan = self.guide.coordinates[self.guide.m_done_nan, :]
        if len(nan) > 0:
            self.ax.scatter(*nan.T, s=10, color="#FF5555", marker="x")

        self.ax.set_title("Data: {:d} points".format(len(values)))
        self.ax.set_xlim(self.axes_limits[0])
        self.ax.set_ylim(self.axes_limits[1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def __dump_animation__(self):
        """Dumps animation frame."""
        data = numpy.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.animation_data.append(data)

    def save_gif(self, name, **kwargs):
        """Saves the GIF animation."""
        imageio.mimsave(name, self.animation_data, **kwargs)

    def notify_changed(self):
        """Updates if possible."""
        if self.guide.tri is not None:
            self.update()
            if self.animation_data is not None:
                self.__dump_animation__()


def run(target, ranges, verbose=False, depth=1, max_fails=0, limit=None, plot=False,
        gif=None, snap_threshold=0.5, nan_threshold=0.5, volume_ratio=10, save=True,
        load=None):

    stats_start_time = datetime.now()
    status_code = 0

    def v(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if load is not None:
        v("Loading from {} ...".format(load))
        with open(load, 'r') as f:
            guide = CurvatureGuide.from_json(json.load(f))
    else:
        guide = CurvatureGuide.from_bounds(
            ranges, snap_threshold=snap_threshold, nan_threshold=nan_threshold,
            volume_ratio=volume_ratio
        )
    ppp = PointProcessPool(target, guide)

    if limit is None:
        def limit_checker():
            return True

    elif isinstance(limit, int):
        def limit_checker():
            return ppp.completed < limit

    elif isinstance(limit, timedelta):
        def limit_checker():
            return datetime.now() < stats_start_time + limit

    else:
        raise ValueError("Unknown limit object: {}".format(limit))

    if max_fails is not None:
        def fail_checker():
            return ppp.failed <= max_fails

    else:
        def fail_checker():
            return True

    v("Hello")

    if plot:
        if guide.dims == 2:
            plot_view = PlotView2D(guide, window=True, record_animation=gif is not None)

        else:
            warn("Cannot plot {:d}D data".format(guide.dims))
            plot_view = None

    else:
        plot_view = None

    def exit_caught():
        if plot_view is not None and plot_view.closed_flag:
            return True
        return False

    while True:

        # Collect data
        swept = ppp.sweep()

        if exit_caught():
            if len(ppp.processes) == 0:
                break
        else:
            if not (limit_checker() and fail_checker()):
                if not fail_checker():
                    status_code = 1
                break

        # Spawn new
        new_to_spawn = depth - len(ppp.processes)
        if isinstance(limit, int):
            new_to_spawn = min(new_to_spawn, limit - ppp.completed)
        if new_to_spawn > 0 and not exit_caught():
            new_points = guide.suggest(new_to_spawn)
            if len(new_points) == 0 and ppp.running == 0:
                raise RuntimeError("No processes are running and no new points have been suggested")
            if len(new_points) > 0:
                v("Spawn: {:d} success: {:d} fail: {:d}".format(len(new_points), ppp.succeeded, ppp.failed))
                for p in new_points:
                    ppp.new(p)
            spawned = len(new_points)

        if plot_view is not None and (swept > 0 or spawned > 0):
            plot_view.notify_changed()

    v("Done")

    if plot_view and gif is not None:
        v("Saving image {} ...".format(gif))
        plot_view.save_gif(gif, duration=0.5)

    if save not in (None, False):
        if save is True:
            data_folder = Path(".curious")
            save = data_folder / datetime.now().strftime("%Y-%m-%d-%H:%M:%S:%f.json")
        else:
            save = Path(save)
            data_folder = save.parent

        if not data_folder.exists():
            v("Creating data folder {} ...".format(data_folder))
            os.mkdir(data_folder)

        if not data_folder.is_dir():
            print("Not a folder: {}".format(data_folder), file=sys.stderr)
            print("Data: {}".format(json.dumps(guide.to_json())))
            status_code = 1

        else:
            v("Dumping into {} ...".format(save))
            with open(save, 'w') as f:
                json.dump(guide.to_json(), f)

    return status_code


if __name__ == "__main__":
    def l2s(limit):
        """
        Converts limit into string.
        Args:
            limit (str, timedelta, None): the limit;

        Returns:
            A string representing the limit.
        """
        if isinstance(limit, timedelta):
            return "time:{}".format(str(limit))
        elif isinstance(limit, int):
            return "eval:{:d}".format(limit)
        elif limit is None:
            return "none"
        else:
            raise ValueError("Unknown limit object: {}".format(limit))

    def s2l(limit):
        """
        Converts string into limit.
        Args:
            limit (str): a string with the limit;

        Returns:
            A limit.
        """
        if limit.startswith("time:"):
            t = datetime.strptime(limit[5:], "%H:%M:%S")
            return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
        elif limit.startswith("eval:"):
            return int(limit[5:])
        elif limit == 'none':
            return None
        else:
            raise ValueError("Unknown limit: {}".format(limit))

    def s2r(ranges):
        """
        Converts string into ranges.
        Args:
            ranges (str): a string representation of ranges;

        Returns:
            A nested tuple with ranges.
        """
        return tuple(
            tuple(float(j) for j in i.strip().split(" "))
            for i in ranges.split(",")
        )

    spec = inspect.getfullargspec(run)
    defaults = dict(zip(spec.args[-len(spec.defaults):], spec.defaults))

    parser = argparse.ArgumentParser(description="curious: sample parameter spaces efficiently")
    parser.add_argument("target", help="target function", metavar="EXECUTABLE", type=str)
    parser.add_argument("ranges", help="ranges to sample", metavar="X Y, A B, ...", type=str)

    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-d", "--depth", help="the number of simultaneous evaluations", metavar="N", type=int,
                        default=defaults["depth"])
    parser.add_argument("-f", "--max-fails", help="the maximal number of failures before exiting", metavar="N",
                        type=int, default=defaults["max_fails"])
    parser.add_argument("-l", "--limit", help="the stop condition: 'none' (runs infinitely), "
                                              "'time:DD:HH:MM:SS' (time limit), "
                                              "'eval:N' (the total number of evaluations)",
                        metavar="LIMIT", type=str, default=l2s(defaults["limit"]))
    parser.add_argument("-p", "--plot", help="display a window with status plot", action="store_true")
    parser.add_argument("--gif", help="Save visualization with all steps into GIF file", metavar="FILENAME", type=str,
                        default=defaults["gif"])
    parser.add_argument("--snap-threshold", help="a threshold to snap points to faces", metavar="FLOAT", type=float,
                        default=defaults["snap_threshold"])
    parser.add_argument("--nan-threshold", help="a critical ratio of nans to fallback to uniform sampling",
                        metavar="FLOAT", type=float, default=defaults["nan_threshold"])
    parser.add_argument("--volume-ratio", help="the largest-to-smallest volume ratio to keep",
                        metavar="FLOAT", type=float, default=defaults["volume_ratio"])
    parser.add_argument("--no-io", help="do not save progress", action="store_true")
    parser.add_argument("--save", help="save progress to a file (overrides --no-io)", metavar="FILENAME", type=str,
                        default=defaults["save"])
    parser.add_argument("--load", help="loads a previous calculation", metavar="FILENAME", type=str,
                        default=defaults["load"])

    options = parser.parse_args()

    exit(run(
        target=options.target,
        ranges=s2r(options.ranges),
        verbose=options.verbose,
        depth=options.depth,
        max_fails=options.max_fails,
        limit=s2l(options.limit),
        plot=options.plot or options.gif is not None,
        gif=options.gif,
        snap_threshold=options.snap_threshold,
        nan_threshold=options.nan_threshold,
        volume_ratio=options.volume_ratio,
        save=options.save if options.save is not True else not options.no_io,
        load=options.load,
    ))

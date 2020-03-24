#!/usr/bin/env python3
from cutil import simplex_volumes, simplex_volumes_n

import imageio
import matplotlib
from matplotlib import pyplot
from matplotlib.tri import Triangulation
from matplotlib.collections import LineCollection
import numpy
from scipy.spatial import Delaunay

from itertools import product, combinations
from collections.abc import Iterable
import argparse
import inspect
from datetime import timedelta, datetime
import subprocess
import re
import sys
import os
from pathlib import Path
import json
import time
import signal


FLAG_PENDING = 0
FLAG_RUNNING = 1
FLAG_DONE = 2


class DSTub1D:
    def __init__(self, coordinates):
        # Points
        self.points = coordinates

        # Simplices
        order = numpy.argsort(coordinates[:, 0]).astype(numpy.int32)
        self.simplices = numpy.concatenate((order[:-1, numpy.newaxis], order[1:,numpy.newaxis]), axis=-1)

        # Neighbours
        left_right = numpy.full(len(order) + 2, -1, dtype=numpy.int32)
        left_right[1:-1] = order
        order_inv = numpy.argsort(order)
        left = left_right[:-2][order_inv]
        right = left_right[2:][order_inv]
        neighbours = numpy.concatenate((right[:, numpy.newaxis], left[:, numpy.newaxis]), axis=-1).reshape(-1)
        neighbours_ptr = numpy.arange(0, len(neighbours) + 2, 2, dtype=numpy.int32)
        self.vertex_neighbor_vertices = neighbours_ptr, neighbours


class Guide(Iterable):
    def __init__(self, dims, dims_f=1, points=None, snap_threshold=None, default_value=0, default_flag=FLAG_PENDING,
                 priority_tolerance=None, meta=None):
        """
        Guides the sampling.
        Args:
            dims (int): the dimensionality of the parameter space;
            dims_f (int): the dimensionality of the target cost function space;
            points (Iterable): data points;
            snap_threshold (float): a threshold to snap to triangle/simplex faces;
            default_value (float): default value to assign to input points;
            default_flag (float): default flag to assign to input points;
            priority_tolerance (float): expected tolerance of priority values;
            meta (Iterable): additional metadata to store alongside each point;
        """
        self.dims = dims
        self.dims_f = dims_f
        if points is None:
            self.data = numpy.empty((0, dims + dims_f + 1), dtype=float)
            self.data_meta = numpy.empty(0, dtype=object)
        else:
            points = numpy.array(tuple(points))
            self.data = numpy.zeros((len(points), dims + dims_f + 1), dtype=float)
            self.data_meta = numpy.full(len(points), None, dtype=object)
            self.values[:] = default_value
            self.flags[:] = default_flag
            self.data[:, :points.shape[1]] = points
            if meta is not None:
                self.data_meta[:] = meta
        self.snap_threshold = snap_threshold
        self.default_value = default_value
        if priority_tolerance is None:
            self.priority_tolerance = 1e-9
        else:
            self.priority_tolerance = priority_tolerance
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
        return cls(len(bounds), points=product(*bounds), **kwargs)

    @property
    def coordinates(self):
        return self.data[:, :self.dims]

    @property
    def values(self):
        return self.data[:, self.dims:self.dims + self.dims_f]

    @property
    def flags(self):
        return self.data[:, -1]

    def __getstate__(self):
        return dict(dims=self.dims, dims_f=self.dims_f, points=self.data.tolist(), snap_threshold=self.snap_threshold,
                    default_value=self.default_value, priority_tolerance=self.priority_tolerance,
                    meta=self.data_meta.tolist())

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
    def __any_nan_mask__(self):
        return numpy.any(numpy.isnan(self.values), axis=-1)

    @property
    def m_done_numeric(self):
        """Final points with numeric values"""
        return numpy.where((self.flags == FLAG_DONE) * (numpy.logical_not(self.__any_nan_mask__)))[0]

    @property
    def m_done_nan(self):
        """Final points with numeric values"""
        return numpy.where((self.flags == FLAG_DONE) * self.__any_nan_mask__)[0]

    def maybe_triangulate(self):
        """Computes a new triangulation"""
        if len(self.data) >= 2 ** self.dims:
            if self.dims == 1:
                self.tri = DSTub1D(self.coordinates)
            else:
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
        if not self.dims <= len(p) <= self.dims + self.dims_f + 1:
            raise ValueError("Wrong point data length {:d}, expected {:d} <= len(p) <= {:d}".format(
                len(p), self.dims, self.dims + self.dims_f + 1))
        self.data = numpy.append(
            self.data,
            [tuple(p) + ((self.default_value,) * self.dims_f + (FLAG_PENDING,))[len(p) - self.dims:]],
            axis=0)
        self.data_meta = numpy.append(self.data_meta, [None])
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

    def __choose_from_alternatives__(self, alternatives):
        raise NotImplementedError

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
            simplex = self.__choose_from_alternatives__(nans)
        else:
            # Largest volumes otherwise
            max_priority = numpy.max(priority)
            simplex = self.__choose_from_alternatives__(
                numpy.where(priority >= max_priority - self.priority_tolerance)[0])
        simplex_coordinates = self.tri.points[self.tri.simplices[simplex], :]
        center_bcc = self.__center__(simplex_coordinates)
        center_coordinate = center_bcc @ simplex_coordinates
        stub_value = center_bcc @ self.values[self.tri.simplices[simplex], ...]
        return self.add(*center_coordinate, *stub_value, FLAG_RUNNING)

    def __iter__(self):
        return self


class UniformGuide(Guide):
    def __choose_from_alternatives__(self, alternatives):
        return alternatives[0]

    def get_simplex_volumes(self):
        """Calculates simplexes' volumes"""
        return simplex_volumes(self.tri.points, self.tri.simplices)

    def priority(self):
        """Calculates priority for sampling of each triangle"""
        return self.get_simplex_volumes()


class CurvatureGuide(UniformGuide):
    def __init__(self, *args, nan_threshold=None, volume_ratio=None, **kwargs):
        """
        Guides the sampling.
        Args:
            nan_threshold (float): a critical value of NaNs to fallback to uniform sampling;
            volume_ratio (float): the ratio between the largest volume present and the
            smallest volume to sample;
            args, kwargs: passed to Guide;
        """
        super().__init__(*args, **kwargs)
        self.nan_threshold = nan_threshold
        self.volume_ratio = volume_ratio

    def __getstate__(self):
        return {**super().__getstate__(), **dict(
            nan_threshold=self.nan_threshold,
            volume_ratio=self.volume_ratio,
        )}

    def __choose_from_alternatives__(self, alternatives):
        return alternatives[numpy.argmax(self.get_simplex_volumes()[alternatives])]

    def get_curvature_measure(self):
        """Calculates the measure of curvature"""
        fns = []
        for i in self.values.T:
            fns.append(simplex_volumes_n(
                numpy.concatenate((self.coordinates, i[:, numpy.newaxis]), axis=-1),
                self.tri.simplices, *self.tri.vertex_neighbor_vertices,
            ))
        return numpy.max(fns, axis=0)

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
    def __init__(self, target, guide, float_regex=r"([-+]?[0-9]+\.?[0-9]*(?:[eEdD][-+]?[0-9]+)?)|(nan)",
                 encoding="utf-8", limit=None, fail_limit=None):
        self.target = target
        self.guide = guide
        self.compiled_float_regex = re.compile(float_regex)
        self.encoding = encoding
        self.processes = []
        self.__failed__ = self.__succeeded__ = 0
        self.draining = False
        self.limit = limit
        self.fail_limit = fail_limit
        self.start_time = datetime.now()
        self.__check_drain__()

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

    @property
    def projected(self):
        return self.running + self.completed

    def __check_drain__(self):
        if not self.draining:
            if self.fail_limit is not None:
                if self.failed > self.fail_limit:
                    self.draining = True

            if self.limit is None:
                pass
            elif isinstance(self.limit, int):
                if self.projected >= self.limit:
                    self.draining = True
            elif isinstance(self.limit, timedelta):
                if datetime.now() > self.start_time + self.limit:
                    self.draining = True
            else:
                raise ValueError("Unknown limit object: {}".format(self.limit))

    def new(self, p):
        if not self.draining:
            self.processes.append((p, subprocess.Popen(
                (self.target,) + tuple(map("{:.12e}".format, self.guide.coordinates[p])),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)
            )))
            self.__check_drain__()
            return True
        else:
            return False

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
                    matches = tuple(i.group() for i in self.compiled_float_regex.finditer(out))

                    if len(matches) >= self.guide.dims_f:
                        self.guide.values[point] = tuple(map(float, matches[-self.guide.dims_f:]))
                        self.guide.flags[point] = FLAG_DONE
                        self.guide.data_meta[point] = (out, err)

        for i in transaction:
            self.processes.remove(i)

        return len(transaction)

    @property
    def safe_to_exit(self):
        return len(self.processes) == 0

    @property
    def exit_now(self):
        return self.draining and self.safe_to_exit

    def start_drain(self, *args):
        """Listens for closing event."""
        self.draining = True


def plot_matrix(n, **kwargs):
    """
    Creates a plot matrix.
    Args:
        n (int): the number of plots;
        kwargs: keyword arguments passed to `pyplot.subplots`;

    Returns:
        Figure and axes.
    """
    n_cols = int(numpy.ceil(n ** .5))
    n_rows = int(numpy.ceil(n / n_cols))
    if "squeeze" not in kwargs:
        kwargs["squeeze"] = False
    fig, ax = pyplot.subplots(n_rows, n_cols, **kwargs)
    ax = ax.reshape(-1)
    return fig, ax


class PlotView:
    def __init__(self, guide, target=None, window_close_listener=None, **kwargs):
        """
        Plots 1D sampling.
        Args:
            guide (Guide): the guide to view;
            kwargs: keyword arguments to `pyplot.subplots`;
        """
        self.guide = guide
        self.axes_limits = guide.get_lims()
        self.target = target
        if target:
            matplotlib.use('agg')
        self.fig, self.ax = plot_matrix(self.__get_plot_n__(), **kwargs)
        if self.target is None:
            pyplot.ion()
            pyplot.show()
            if window_close_listener is not None:
                self.fig.canvas.mpl_connect('close_event', window_close_listener)

        self.record_animation = self.target is not None and self.target.endswith(".gif")
        if self.record_animation:
            self.animation_data = []
        else:
            self.animation_data = None

    def __get_plot_n__(self):
        return self.guide.dims_f

    def update(self):
        """Updates the figure."""
        if self.guide.tri is None:
            raise ValueError("Triangulation is None")

        for ax in self.ax:
            ax.clear()
        self.__plot_main__()

        self.fig.suptitle("Data: {:d} points".format(len(self.guide.values)))

        self.fig.canvas.draw()
        if self.target is None:
            self.fig.canvas.flush_events()
        else:
            if self.record_animation:
                if len(self.animation_data) > 0:
                    imageio.mimsave(self.target, self.animation_data, duration=1)
            else:
                pyplot.savefig(self.target)

    def __plot_main__(self):
        raise NotImplementedError

    def __dump_animation__(self):
        """Dumps animation frame."""
        data = numpy.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
        data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        self.animation_data.append(data)

    def notify_changed(self):
        """Updates if possible."""
        if self.guide.tri is not None:
            self.update()
            if self.animation_data is not None:
                self.__dump_animation__()


class PlotView1D(PlotView):
    def __init__(self, guide, **kwargs):
        if guide.dims != 1:
            raise ValueError("This view is for 1D->ND functions only")
        super().__init__(guide, **kwargs)

    def __plot_main__(self):
        tri = self.guide.tri
        p = tri.points.squeeze()[tri.simplices]
        coordinates = numpy.squeeze(self.guide.coordinates, axis=-1)
        for values, ax in zip(self.guide.values.T, self.ax):
            v = values[tri.simplices]
            lc = LineCollection(numpy.concatenate((p[..., numpy.newaxis], v[..., numpy.newaxis]), axis=-1))
            ax.add_collection(lc)

            running = self.guide.m_running
            if numpy.any(running):
                ax.scatter(coordinates[running], values[running], facecolors='none', edgecolors="black")

            nan = self.guide.m_done_nan
            if numpy.any(nan):
                ax.scatter(coordinates[nan], values[nan], s=10, color="#FF5555", marker="x")

            ax.set_xlim(self.axes_limits[0])


class PlotView2D(PlotView):
    def __init__(self, guide, **kwargs):
        if guide.dims != 2:
            raise ValueError("This view is for 2D->ND functions only")
        super().__init__(guide, **kwargs)

    def __plot_main__(self):
        for values, ax in zip(self.guide.values.T, self.ax):
            color_plot = ax.tripcolor(
                Triangulation(*self.guide.tri.points.T, triangles=self.guide.tri.simplices),
                values,
                vmin=numpy.nanmin(self.guide.values),
                vmax=numpy.nanmax(self.guide.values),
            )

            # if self.color_bar is True:
            #     self.color_bar = pyplot.colorbar(color_plot)
            # elif self.color_bar in (False, None):
            #     pass
            # else:
            #     self.color_bar.on_mappable_changed(color_plot)

            # valid = self.guide.coordinates[self.guide.m_done_numeric, :]
            # if len(valid) > 0:
            #     ax.scatter(*valid.T, color="white")

            running = self.guide.coordinates[self.guide.m_running, :]
            if len(running) > 0:
                ax.scatter(*running.T, facecolors='none', edgecolors="white")

            nan = self.guide.coordinates[self.guide.m_done_nan, :]
            if len(nan) > 0:
                ax.scatter(*nan.T, s=10, color="#FF5555", marker="x")


            ax.set_xlim(self.axes_limits[0])
            ax.set_ylim(self.axes_limits[1])


class PlotViewProj2D(PlotView):
    def __get_plot_n__(self):
        return self.guide.dims_f * self.guide.dims * (self.guide.dims - 1) / 2

    def __plot_main__(self):
        for ax, ((i_x, i_y), i_v) in zip(self.ax, product(
                combinations(numpy.arange(self.guide.dims), 2),
                numpy.arange(self.guide.dims_f),
        )):
            x = self.guide.coordinates[:, i_x]
            y = self.guide.coordinates[:, i_y]
            v = self.guide.values[:, i_v]
            for mask, kw in (
                (self.guide.m_done_numeric, dict(c=v[self.guide.m_done_numeric])),
                (self.guide.m_running, dict(facecolors='none', edgecolors="black")),
                (self.guide.m_done_nan, dict(color="#FF5555", marker="x")),
            ):
                ax.scatter(x[mask], y[mask], **kw)

            ax.set_xlim(self.axes_limits[i_x])
            ax.set_ylim(self.axes_limits[i_y])
            ax.set_xlabel(f"x[{i_x}]")
            ax.set_ylabel(f"x[{i_y}]")
            ax.set_title(f"f(x)[{i_v}]")


def run(target, ranges, n=1, verbose=False, depth=1, max_fails=0, limit=None, plot=False,
        snap_threshold=0.5, nan_threshold=0.5, volume_ratio=10, save=True, load=None):

    def v(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if load is not None:
        v("Loading from {} ...".format(load))
        with open(load, 'r') as f:
            guide = (UniformGuide if n == 0 else CurvatureGuide).from_json(json.load(f))
    else:
        if n == 0:
            guide = UniformGuide.from_bounds(
                ranges, snap_threshold=snap_threshold, dims_f=n,
            )
        else:
            guide = CurvatureGuide.from_bounds(
                ranges, snap_threshold=snap_threshold, nan_threshold=nan_threshold,
                volume_ratio=volume_ratio, dims_f=n,
            )

    ppp = PointProcessPool(target, guide, limit=limit, fail_limit=max_fails)

    v("Hello")

    if plot is not False:
        plot_view = {1: PlotView1D, 2: PlotView2D}.get(guide.dims, PlotViewProj2D)(
            guide, target=plot, window_close_listener=ppp.start_drain)
        plot_view.notify_changed()

    else:
        plot_view = None

    while not ppp.exit_now:

        try:
            # Collect the data
            swept = ppp.sweep()

            # Spawn new
            spawned = 0
            for _ in range(depth - len(ppp.processes)):
                if ppp.draining:
                    break
                try:
                    spawned += ppp.new(next(guide))
                except StopIteration:
                    break

            if swept or spawned:
                if plot_view is not None:
                    plot_view.notify_changed()
                v("Running: {:d} (+{:d} -{:d}) \tcompleted: {:d}[{:d}] ({:d} fails) \t{}".format(
                    ppp.running, spawned, swept, ppp.completed, len(guide.m_done), ppp.failed, "DRAIN" if ppp.draining else ""))

            time.sleep(0.1)

        except KeyboardInterrupt:
            if ppp.draining:
                raise
            else:
                print("Ctrl-C caught: finalizing ...")
                ppp.start_drain()

    v("Done")

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
            print("Data: {}".format(json.dumps(guide.to_json(), indent=2)))
            return 1

        else:
            v("Dumping into {} ...".format(save))
            with open(save, 'w') as f:
                json.dump(guide.to_json(), f, indent=2)

    return ppp.failed > ppp.fail_limit


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
    parser.add_argument("-n", help="the number of target cost functions", metavar="N", type=int, default=defaults["n"])
    parser.add_argument("-d", "--depth", help="the number of simultaneous evaluations", metavar="N", type=int,
                        default=defaults["depth"])
    parser.add_argument("-f", "--max-fails", help="the maximal number of failures before exiting", metavar="N",
                        type=int, default=defaults["max_fails"])
    parser.add_argument("-l", "--limit", help="the stop condition: 'none' (runs infinitely), "
                                              "'time:DD:HH:MM:SS' (time limit), "
                                              "'eval:N' (the total number of evaluations)",
                        metavar="LIMIT", type=str, default=l2s(defaults["limit"]))
    parser.add_argument("--plot", help="make a status plot", metavar="[FILENAME]", type=str, nargs="?",
                        default=defaults["plot"])
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
        n=options.n,
        verbose=options.verbose,
        depth=options.depth,
        max_fails=options.max_fails,
        limit=s2l(options.limit),
        plot=options.plot,
        snap_threshold=options.snap_threshold,
        nan_threshold=options.nan_threshold,
        volume_ratio=options.volume_ratio,
        save=options.save if options.save is not True else not options.no_io,
        load=options.load,
    ))

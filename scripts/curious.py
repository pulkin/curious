#!/usr/bin/env python3
from cutil import simplex_volumes, simplex_volumes_n

import numpy as np
from scipy.spatial import Delaunay

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from itertools import product
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
import random
import string


FLAG_PENDING = 0
FLAG_RUNNING = 1
FLAG_DONE = 2

MIN_PORT = 9911
MAX_PORT = 9926


class DSTub1D:
    def __init__(self, coordinates):
        """
        A replica of `Delaunay` triangulation in 1D.

        Args:
            coordinates (numpy.ndarray): a plain vector with 1D coordinates.
        """
        # Points
        self.points = coordinates

        # Simplexes
        order = np.argsort(coordinates[:, 0]).astype(np.int32)
        self.simplices = np.concatenate((order[:-1, np.newaxis], order[1:, np.newaxis]), axis=-1)

        # Neighbours
        left_right = np.full(len(order) + 2, -1, dtype=np.int32)
        left_right[1:-1] = order
        order_inv = np.argsort(order)
        left = left_right[:-2][order_inv]
        right = left_right[2:][order_inv]
        neighbours = np.concatenate((right[:, np.newaxis], left[:, np.newaxis]), axis=-1).reshape(-1)
        neighbours_ptr = np.arange(0, len(neighbours) + 2, 2, dtype=np.int32)
        self.vertex_neighbor_vertices = neighbours_ptr, neighbours


class Guide(Iterable):
    def __init__(self, dims, dims_f=1, points=None, snap_threshold=None, default_value=0, default_flag=FLAG_PENDING,
                 priority_tolerance=None, meta=None, rescale=True):
        """
        Guides the sampling.

        Args:
            dims (int): the dimensionality of the parameter space to sample;
            dims_f (int): the dimensionality of the output (target space);
            points (Iterable): data points;
            snap_threshold (float): a threshold to snap to triangle/simplex faces;
            default_value (float): default value to assign to input points;
            default_flag (float): default flag to assign to input points;
            priority_tolerance (float): expected tolerance of priority values;
            meta (Iterable): additional metadata to store alongside each point;
            rescale (bool): if True, rescales parameter ranges to a unit box before triangulating;
        """
        self.dims = dims
        self.dims_f = dims_f
        if points is None:
            self.data = np.empty((0, dims + dims_f + 1), dtype=float)
            self.data_meta = np.empty(0, dtype=object)
        else:
            points = np.array(tuple(points))
            self.data = np.zeros((len(points), dims + dims_f + 1), dtype=float)
            self.data_meta = np.full(len(points), None, dtype=object)
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
        self.rescale = rescale
        self.tri = self.maybe_triangulate()

    def set_bounds(self, bounds):
        """
        Set new bounds.

        Args:
            bounds (tuple): bounds defining the box;
        """
        for i in product(*bounds):
            self.maybe_add(*i, strict=False)

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

    def set_prop(self, name, value):
        """
        Sets a property.

        Args:
            name (str): property name;
            value (str): property value;
        """
        if name == "bounds":
            self.set_bounds(s2r(value))
        elif name == "snap-threshold":
            self.snap_threshold = float(value)
        elif name == "rescale":
            self.rescale = bool(value)
        else:
            raise KeyError(f"Unknown property to set {repr(name)}")

    @property
    def coordinates(self) -> np.ndarray:
        return self.data[:, :self.dims]

    @property
    def values(self) -> np.ndarray:
        return self.data[:, self.dims:self.dims + self.dims_f]

    @property
    def flags(self) -> np.ndarray:
        return self.data[:, -1]

    def __getstate__(self) -> dict:
        return dict(dims=self.dims, dims_f=self.dims_f, points=self.data.tolist(), snap_threshold=self.snap_threshold,
                    default_value=self.default_value, priority_tolerance=self.priority_tolerance,
                    meta=self.data_meta.tolist(), rescale=self.rescale)

    def __setstate__(self, state):
        self.__init__(**state)

    def to_json(self):
        """
        Serializes to JSON-compatible dict.
        Returns:
            A dict with parameters.
        """
        return self.__getstate__()

    def save_to(self, f, **kwargs):
        """
        Serializes into a text file as JSON.

        Args:
            f (file, str): a file to save to;
            **kwargs: arguments to `json.dump`;
        """
        def_kwargs = dict(indent=2)
        def_kwargs.update(kwargs)
        if isinstance(f, (str, Path)):
            with open(f, 'w') as _f:
                json.dump(self.to_json(), _f, **def_kwargs)
        else:
            json.dump(self.to_json(), f, **def_kwargs)

    @classmethod
    def from_json(cls, data):
        """
        Restores from JSON data.

        Args:
            data (dict): the serialized data;

        Returns:
            A new Guide.
        """
        return cls(**data)

    @property
    def m_pending(self) -> np.ndarray:
        """Mask with points having 'pending' status."""
        return np.where(self.flags == FLAG_PENDING)[0]

    @property
    def m_running(self) -> np.ndarray:
        """Indices of points with 'running' status."""
        return np.where(self.flags == FLAG_RUNNING)[0]

    @property
    def m_done(self) -> np.ndarray:
        """Indices of points with 'done' status."""
        return np.where(self.flags == FLAG_DONE)[0]

    @property
    def __any_nan_mask__(self) -> np.ndarray:
        return np.any(np.isnan(self.values), axis=-1)

    @property
    def m_done_numeric(self) -> np.ndarray:
        """Indices of points with numeric results."""
        return np.where((self.flags == FLAG_DONE) * (np.logical_not(self.__any_nan_mask__)))[0]

    @property
    def m_done_nan(self) -> np.ndarray:
        """Indices ofpoints with non-numeric results."""
        return np.where((self.flags == FLAG_DONE) * self.__any_nan_mask__)[0]

    def maybe_triangulate(self):
        """
        Prepares a new triangulation.

        Returns:
            Tbe new triangulation stored in `self.tri` or None if no
            triangulation possible.
        """
        if len(self.data) >= 2 ** self.dims:

            if self.rescale:
                coordinates_min = self.coordinates.min(axis=0)
                coordinates_max = self.coordinates.max(axis=0)
                coordinates = self.coordinates / ((coordinates_max - coordinates_min)[np.newaxis, :])
            else:
                coordinates = self.coordinates

            if self.dims == 1:
                self.tri = DSTub1D(coordinates)
            else:
                self.tri = Delaunay(coordinates, qhull_options='Qz')

        else:
            self.tri = None
        return self.tri

    def maybe_add(self, *p, strict=False):
        """
        Adds a new point and triangulates.
        Args:
            p: the point to add;
            strict (bool): raises an exception if the point is already present;

        Returns:
            The index of the point added.
        """
        if not self.dims <= len(p) <= self.dims + self.dims_f + 1:
            raise ValueError("Wrong point data length {:d}, expected {:d} <= len(p) <= {:d}".format(
                len(p), self.dims, self.dims + self.dims_f + 1))
        point_location = np.array(p)[:self.dims]
        delta = np.prod(self.coordinates == point_location[np.newaxis, :], axis=1)
        if np.any(delta):
            if strict:
                raise ValueError(f"The point is already present in the guide.")
            else:
                return
        new_value = (self.default_value,) * self.dims_f
        new_flag = FLAG_PENDING,
        new_data_row = new_value + new_flag
        new_data_row = p + new_data_row[len(p) - self.dims:]
        self.data = np.append(
            self.data,
            [new_data_row],
            axis=0)
        self.data_meta = np.append(self.data_meta, [None])
        self.maybe_triangulate()
        return len(self.data) - 1

    def get_data_limits(self) -> tuple:
        """Computes data limits."""
        return tuple(zip(
            np.min(self.coordinates, axis=0),
            np.max(self.coordinates, axis=0))
        )

    def priority(self) -> np.ndarray:
        """
        Priority for sampling each of the triangles.
        Returns:
            An array with priorities: arbitrary real numbers.
        """
        raise NotImplementedError

    def __simplex_middle__(self, simplex) -> np.ndarray:
        """
        Pick a middle point inside a multidimensional simplex according to
        the rules provided by this object.

        Args:
            simplex (numpy.array): simplex points;

        Returns:
            A center point in barycentric coordinates.
        """
        center = np.mean(simplex, axis=0)
        dst = np.linalg.norm(simplex - center[np.newaxis, :], axis=-1)
        max_dst = dst.max()
        mask = np.asanyarray(dst > self.snap_threshold * max_dst, dtype=float)
        if mask.sum() < 2:
            mask[np.argsort(dst)[:-3:-1]] = .5
        else:
            mask /= mask.sum()
        return mask

    def __choose_from_alternatives__(self, alternatives) -> int:
        """
        Choose from indices of alternative simplexes to sample.

        Return:
            A single index of the simplex to sample.
        """
        raise NotImplementedError

    def __next__(self):
        """Sample another point."""
        # First, pick points that are pending.
        for i in self.m_pending:
            self.flags[i] = FLAG_RUNNING
            return i

        # Do nothing if not triangulated
        if self.tri is None:
            raise StopIteration

        priority = self.priority()

        # Second, nans
        nans = np.where(np.isnan(priority))[0]
        if len(nans) > 0:
            simplex = self.__choose_from_alternatives__(nans)
        else:
            # Largest volumes otherwise
            max_priority = np.max(priority)
            simplex = self.__choose_from_alternatives__(
                np.where(priority >= max_priority - self.priority_tolerance)[0])
        simplex_coordinates = self.coordinates[self.tri.simplices[simplex], :]
        center_bcc = self.__simplex_middle__(simplex_coordinates)
        center_coordinate = center_bcc @ simplex_coordinates
        stub_value = center_bcc @ self.values[self.tri.simplices[simplex], ...]
        return self.maybe_add(*center_coordinate, *stub_value, FLAG_RUNNING, strict=True)

    def __iter__(self):
        return self


class UniformGuide(Guide):
    def __choose_from_alternatives__(self, alternatives) -> int:
        """
        Choose from indices of alternative simplexes to sample.
        This implementation simply picks the first index.

        Return:
            A single index of the simplex to sample.
        """
        return alternatives[0]

    def get_simplex_volumes(self) -> np.ndarray:
        """
        Computes simplex volumes.

        Returns:
            A plain array with volumes.
        """
        return simplex_volumes(self.coordinates, self.tri.simplices)

    def priority(self) -> np.ndarray:
        """
        Priority for sampling each of the triangles.
        Priorities are simplex volumes in this implementation.

        Returns:
            An array with priorities: arbitrary real numbers.
        """
        return self.get_simplex_volumes()


class CurvatureGuide(UniformGuide):
    def __init__(self, *args, nan_threshold=None, volume_ratio=None, **kwargs):
        """
        Guides the sampling.

        Args:
            nan_threshold (float): a critical value of NaNs to fallback to uniform sampling;
            volume_ratio: the maximal ratio between the largest and the smallest simplex volume;
            args, kwargs: passed to Guide;
        """
        super().__init__(*args, **kwargs)
        self.nan_threshold = nan_threshold
        self.volume_ratio = volume_ratio

    def set_prop(self, name, value):
        """
        Sets a property.

        Args:
            name (str): property name;
            value (str): property value;
        """
        if name == "nan-threshold":
            self.nan_threshold = float(value)
        elif name == "volume-ratio":
            self.volume_ratio = float(value)
        else:
            super().set_prop(name, value)

    def __getstate__(self) -> dict:
        return {**super().__getstate__(), **dict(
            nan_threshold=self.nan_threshold,
            volume_ratio=self.volume_ratio,
        )}

    def __choose_from_alternatives__(self, alternatives) -> int:
        """
        Choose from indices of alternative simplexes to sample.
        This implementation picks the largest-volume alternative.

        Return:
            A single index of the simplex to sample.
        """
        return alternatives[np.argmax(self.get_simplex_volumes()[alternatives])]

    def get_curvature(self) -> np.ndarray:
        """
        Computes the multidimensional curvature as a simplex
        volume.

        Returns:
            The curvature measure as a plain array.
        """
        fns = []
        for i in self.values.T:
            fns.append(simplex_volumes_n(
                np.concatenate((self.coordinates, i[:, np.newaxis]), axis=-1),
                self.tri.simplices, *self.tri.vertex_neighbor_vertices,
            ))
        return np.max(fns, axis=0)

    def priority(self) -> np.ndarray:
        """
        Priority for sampling each of the triangles.
        This implementation takes into account local curvature,
        simplex volumes and the amount of NaNs in the data.

        Returns:
            An array with priorities.
        """
        result = self.get_curvature()

        if self.nan_threshold is not None:
            if np.isnan(result).sum() > self.nan_threshold * len(result):
                return self.get_simplex_volumes()

        if self.volume_ratio is not None:
            volumes = self.get_simplex_volumes()
            discard_mask = volumes / max(volumes) < 1./self.volume_ratio
            result[discard_mask] = 0

        return result


class PointProcessPool:
    def __init__(self, target, guide, float_regex=r"([-+]?[0-9]+\.?[0-9]*(?:[eEdD][-+]?[0-9]+)?)|(nan)",
                 encoding="utf-8", limit=None, fail_limit=None):
        """
        A pool of running processes sampling the parameter space.

        Args:
            target (str): executable;
            guide (Guide): the guide suggesting new points to sample;
            float_regex (str): a regular expression to find float numbers;
            encoding (str): encoding of the target output stream;
            limit (int, timedelta, None): a limit to the total count of processes spawned:
            either total time or total process count;
            fail_limit (int, None): the total count of processes with non-zero
            exit codes to accept;
        """
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
    def failed(self) -> int:
        """The total count of failed processes with non-zero exit codes."""
        return self.__failed__

    @property
    def succeeded(self) -> int:
        """The total count of succeeded processes with zero exit codes."""
        return self.__succeeded__

    @property
    def running(self):
        """The total count of running processes."""
        return len(self.processes)

    @property
    def completed(self):
        """The total count of completed processes."""
        return self.failed + self.succeeded

    @property
    def projected(self):
        """The total count of completed plus running processes."""
        return self.running + self.completed

    def __check_drain__(self):
        """
        Check conditions to enter drain mode.
        Set the corresponding flag if so.
        """
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

    def new(self, p) -> bool:
        """
        Spawn a new process computing the given point.

        Args:
            p (numpy.ndarray): the point to compute;

        Returns:
            True if spawned successfully. False if in drain mode.
        """
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

    def sweep(self) -> int:
        """
        Collects completed processes.

        Returns:
            The number of points collected.
        """
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
                    matches = tuple(i.group() for i in self.compiled_float_regex.finditer(out))

                    if len(matches) >= self.guide.dims_f:
                        if self.guide.dims_f > 0:
                            self.guide.values[point] = tuple(map(float, matches[-self.guide.dims_f:]))
                        self.guide.flags[point] = FLAG_DONE
                        self.guide.data_meta[point] = (out, err)
                        self.__succeeded__ += 1
                    else:
                        print("Missing {:d} numbers in the output".format(self.guide.dims_f))

        for i in transaction:
            self.processes.remove(i)

        return len(transaction)

    @property
    def is_safe_to_exit(self) -> bool:
        """Checks whether it is safe to exit now: no pending processes are present."""
        return self.running == 0

    @property
    def is_finalized(self) -> bool:
        """Checks whether this run is over: in drain mode with no process left."""
        return self.draining and self.is_safe_to_exit

    def start_drain(self, *args):
        """Switch into drain mode."""
        self.draining = True


def timestamp():
    """Current timestep."""
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")


def run(target, bounds, dims_f=1, verbose=False, depth=1, fail_limit=0, limit=None, snap_threshold=0.5, nan_threshold=0.5,
        volume_ratio=10, rescale=True, save=True, load=None, on_update=None, listen=True):
    """
    Run curious.
    
    Args:
        target (str): executable;

        bounds (tuple): bounds defining the box;

        dims_f (int): the dimensionality of the output (target space);

        verbose (bool): if True, prints verbose output;

        depth (int): the count of parallel runs when sampling;

        fail_limit (int):  the total count of processes with non-zero exit codes to accept;

        limit (int, timedelta, None): a limit to the total count of processes spawned:
        either total run time or total process count;

        snap_threshold (float): a threshold to snap to triangle/simplex faces;

        nan_threshold (int): the total count of NaNs to accept before entering drain mode;

        volume_ratio: the maximal ratio between the largest and the smallest simplex volume;

        rescale (bool): if True, rescales parameter ranges to a unit box before triangulating;
        
        save (str, bool): the location of the JSON data file to save progress to. If True,
        prepares a new file name out of the timestamp;
        
        load (str): the location of the JSON data file to load progress from;
        
        on_update (str): executable to run on data updates;

        listen (bool): creates a file to listen to interact through;
    """

    def v(text, important=False):
        if verbose or important:
            print(f"[{timestamp()}] {text}", flush=True)

    v("hello")

    if on_update is not None:
        def on_plot_update_fun(guide):
            try:
                p = subprocess.Popen(on_update, stdin=subprocess.PIPE, shell=True, encoding="utf-8")
                p.stdin.write(json.dumps(guide.to_json()))
                p.stdin.close()
            except BrokenPipeError:
                v("WARN on_update process refused to receive JSON data", important=True)
    else:
        def on_plot_update_fun(guide):
            pass

    if load is not None:
        v(f"Loading from '{load}' ...")
        with open(load, 'r') as f:
            guide = (UniformGuide if dims_f == 0 else CurvatureGuide).from_json(json.load(f))
        # Update the guide with the new parameters
        guide.set_bounds(bounds)
        guide.snap_threshold = snap_threshold
        if guide.dims_f != dims_f:
            v(f"WARN Ignoring dims_f={dims_f}, using restored value {guide.dims_f}")
        guide.rescale = rescale
        if isinstance(guide, CurvatureGuide):
            guide.nan_threshold = nan_threshold
            guide.volume_ratio = volume_ratio
    else:
        if dims_f == 0:
            guide = UniformGuide.from_bounds(bounds, snap_threshold=snap_threshold, dims_f=dims_f, rescale=rescale)
        else:
            guide = CurvatureGuide.from_bounds(bounds, snap_threshold=snap_threshold, nan_threshold=nan_threshold,
                                               volume_ratio=volume_ratio, dims_f=dims_f, rescale=rescale)

    if listen:

        class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                nonlocal depth

                try:
                    url = urlparse(self.path)
                    print(url)
                    if url.path != f"/{secret}":
                        self.send_response(401)
                        self.end_headers()
                        self.wfile.write(f"Not authorized".encode("utf-8"))
                        return

                    params = parse_qs(url.query)
                    if len(params) != 1:
                        raise ValueError(f"Only a single parameter accepted: {params}")

                    (key, value), = params.items()
                    if len(value) != 1:
                        raise ValueError(f"Only a single parameter value accepted: {value}")
                    value = value[0]

                    if key == "depth":
                        depth = int(value)
                    else:
                        guide.set_prop(key, value)

                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(f"{key}={value}".encode("utf-8"))

                except Exception as _e:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(str(_e).encode('utf-8'))

        secret = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

        for port in range(MIN_PORT, MAX_PORT + 1):
            try:
                httpd = HTTPServer(('localhost', port), SimpleHTTPRequestHandler)
                httpd.timeout = 0
            except OSError as e:
                if e.errno == 98:  # already in use
                    continue
            else:
                break
        v(f"Listening for localhost:{port}/{secret}", important=True)

    if save not in (None, False):
        if save is True:
            data_folder = Path(".curious")
            save = data_folder / timestamp() / ".json"
        else:
            save = Path(save)
            data_folder = save.parent

        if not data_folder.exists():
            v(f"Creating data folder '{data_folder}' ...")
            os.mkdir(data_folder)

        if not data_folder.is_dir():
            raise RuntimeError(f"Not a folder: {data_folder}")

    ppp = PointProcessPool(target, guide, limit=limit, fail_limit=fail_limit)

    on_plot_update_fun(guide)

    while not ppp.is_finalized:

        try:
            if listen:
                httpd.handle_request()
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
                v(f"Running: {ppp.running} (+{spawned} -{swept}) \tcompleted: {ppp.completed}[{len(guide.m_done)}] "
                  f"({ppp.failed} fails) \t{'DRAIN' if ppp.draining else ''}")
                on_plot_update_fun(guide)
                if save:
                    guide.save_to(save)

            time.sleep(1)

        except KeyboardInterrupt:
            if ppp.draining:
                raise
            else:
                v("Ctrl-C caught: finalizing ...", important=True)
                ppp.start_drain()

    v("bye")

    return ppp.failed > ppp.fail_limit


if __name__ == "__main__":
    def l2s(limit) -> str:
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
    parser.add_argument("-n", help="the number of target cost functions", metavar="N", type=int, default=defaults["dims_f"])
    parser.add_argument("-d", "--depth", help="the number of simultaneous evaluations", metavar="N", type=int,
                        default=defaults["depth"])
    parser.add_argument("-f", "--fail-limit", help="the maximal number of failures before exiting", metavar="N",
                        type=int, default=defaults["fail_limit"])
    parser.add_argument("-l", "--limit", help="the stop condition: 'none' (runs infinitely), "
                                              "'time:DD:HH:MM:SS' (time limit), "
                                              "'eval:N' (the total number of evaluations)",
                        metavar="LIMIT", type=str, default=l2s(defaults["limit"]))
    parser.add_argument("--snap-threshold", help="a threshold to snap points to faces", metavar="FLOAT", type=float,
                        default=defaults["snap_threshold"])
    parser.add_argument("--nan-threshold", help="a critical ratio of nans to fallback to uniform sampling",
                        metavar="FLOAT", type=float, default=defaults["nan_threshold"])
    parser.add_argument("--volume-ratio", help="the largest-to-smallest volume ratio to keep",
                        metavar="FLOAT", type=float, default=defaults["volume_ratio"])
    parser.add_argument("--no-rescale", help="triangulate as-is without rescaling axes", action="store_true")
    parser.add_argument("--no-io", help="do not save progress", action="store_true")
    parser.add_argument("--save", help="save progress to a file (overrides --no-io)", metavar="FILENAME", type=str,
                        default=defaults["save"])
    parser.add_argument("--load", help="loads a previous calculation", metavar="FILENAME", type=str,
                        default=defaults["load"])
    parser.add_argument("--on-update", help="update hook", metavar="COMMAND", type=str, default=defaults["on_update"])
    options = parser.parse_args()

    exit(run(
        target=options.target,
        bounds=s2r(options.ranges),
        dims_f=options.n,
        verbose=options.verbose,
        depth=options.depth,
        fail_limit=options.fail_limit,
        limit=s2l(options.limit),
        snap_threshold=options.snap_threshold,
        nan_threshold=options.nan_threshold,
        volume_ratio=options.volume_ratio,
        rescale=not options.no_rescale,
        save=options.save if options.save is not True else not options.no_io,
        load=options.load,
        on_update=options.on_update,
    ))

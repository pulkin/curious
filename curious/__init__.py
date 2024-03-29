#!/usr/bin/env python3
from .cutil import simplex_volumes, simplex_volumes_n
from .util import str2ranges, dump, dumps, load as json_load
from .attrs import check_point_data
from .data import point_record_dtype

import numpy as np
from scipy.spatial import Delaunay

from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from itertools import product
from collections.abc import Iterable
from datetime import timedelta, datetime
import subprocess
import re
import sys
import os
from pathlib import Path
import time
import signal
import random
import string
import logging
from attr import define, field, asdict, fields, filters, fields_dict


MIN_PORT = 9911
MAX_PORT = 9926
DATA_LOG = 15
logging.addLevelName(DATA_LOG, "DATA")


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


@define
class Sampling:
    """
    Sampling of multidimensional space.

    Args:
        points (np.ndarray): initial list of points;
    """
    points: np.ndarray = field(validator=lambda instance, attribute, value: check_point_data(value))
    logger: logging.Logger = None

    def __log_init_args__(self):
        self.logger.log(DATA_LOG, f"  points={self.points.tolist()}")
        self.logger.log(DATA_LOG, f"  dims={self.dims}")
        self.logger.log(DATA_LOG, f"  dims_f={self.dims_f}")
        other = asdict(self)
        del other["points"]
        del other["logger"]
        for k, v in other.items():
            self.logger.info(f"  {k}={v}")

    def __attrs_post_init__(self):
        if self.logger is None:
            self.logger = logging.getLogger(f"curious.{self.__class__.__name__}.0x{id(self):08x}")
        self.logger.log(DATA_LOG, f"init")
        self.__log_init_args__()
        self.logger.log(DATA_LOG, "(end of init)")

    @property
    def dims(self):
        return self.points.dtype.fields["x"][0].shape[0]

    @property
    def dims_f(self):
        return self.points.dtype.fields["y"][0].shape[0]

    @property
    def coordinates(self) -> np.ndarray:
        return self.points["x"]

    @property
    def values(self) -> np.ndarray:
        return self.points["y"]

    @property
    def tags(self) -> np.ndarray:
        return self.points["tag"]

    @property
    def meta(self):
        return self.points["meta"]

    @property
    def size(self):
        return len(self.points)

    def __len__(self):
        return self.size

    def get_volumes(self) -> np.ndarray:
        """Volumes of sampling units as a plain array."""
        raise NotImplementedError

    def get_curvature(self) -> np.ndarray:
        """Curvature measure as a plain array."""
        raise NotImplementedError

    def get_limits(self) -> tuple:
        """Computes a box with coordinate limits: min and max."""
        coordinates = self.coordinates
        return tuple(zip(
            np.min(coordinates, axis=0),
            np.max(coordinates, axis=0))
        )

    def closest(self, x) -> float:
        """
        Distance to the closest point in sampling.

        Args:
            x (np.ndarray): the point to compute distance to;

        Returns:
            The distance.
        """
        if self.size == 0:
            return float("+inf")
        x = np.array(x, dtype=float)
        return np.linalg.norm(self.coordinates - x[None, :], axis=1).min()

    def append(self, x, y=None, tag="⏸", meta=None):
        """
        Adds a point to sampling.

        Args:
            x (np.ndarray): point coordinate;
            y (np.ndarray): point value;
            tag (str): point tag;
            meta: point metadata;
        """
        x = np.asanyarray(x, dtype=float)
        if y is None:
            y = np.full(self.dims_f, float("nan"), dtype=float)
        p = np.asanyarray([(x, y, tag, meta)], self.points.dtype)
        self.logger.log(DATA_LOG, f"append: x={x.tolist()} y={y.tolist()} tag={tag} meta={meta}")
        self.points = np.concatenate([self.points, p])

    def set(self, i, **kwargs):
        """
        Sets point values.

        Args:
            i (int): point index;
            **kwargs: values to set;
        """
        for k, v in kwargs.items():
            self.points[k][i] = v
        kwargs_repr = ' '.join(f'{k}={v}' for k, v in kwargs.items())
        self.logger.log(DATA_LOG, f"set {i}: {kwargs_repr}")

    def set_prop(self, name, value):
        """
        Sets a runtime-friendly property.

        Args:
            name (str): property name;
            value (str): property value;
        """
        name = name.replace("-", "_")
        f = fields_dict(type(self))[name]
        if f.metadata.get("flexible", True) is not True:
            raise ValueError(f"cannot set property {name}")
        setattr(self, f.name, value)
        self.logger.log(DATA_LOG, f"set_prop {name}={value}")

    def sample(self, i) -> list:
        """
        Sample a unit.

        Args:
            i (int): unit index;

        Returns:
            An integer array of new point indices.
        """
        raise NotImplementedError


@define
class SimplexSampling(Sampling):
    @property
    def simplices(self):
        raise NotImplementedError

    def get_volumes(self) -> np.ndarray:
        return simplex_volumes(self.coordinates, self.simplices)

    def get_curvature(self) -> np.ndarray:
        fns = []
        tri = self.tri
        for i in self.values.T:
            fns.append(simplex_volumes_n(
                np.concatenate((self.coordinates, i[:, np.newaxis]), axis=-1),
                tri.simplices, *tri.vertex_neighbor_vertices,
            ))
        # TODO: do not take max here
        return np.max(fns, axis=0)

    def sample(self, i) -> list:
        raise NotImplementedError


@define
class DelaunaySampling(SimplexSampling):
    rescale: bool = field(default=True, metadata={"flexible": True}, converter=bool)
    snap_threshold: float = field(default=None, metadata={"flexible": True}, converter=float)
    proximity_epsilon: float = field(default=1e-8, metadata={"flexible": True}, converter=float)

    @property
    def tri(self):
        """
        Prepares a new triangulation.

        Returns:
            The new triangulation stored in `self._tri` or None if no
            triangulation possible.
        """
        if self.size >= 2 ** self.dims:

            if self.rescale:
                coordinates_min = self.coordinates.min(axis=0)
                coordinates_max = self.coordinates.max(axis=0)
                coordinates = self.coordinates / ((coordinates_max - coordinates_min)[np.newaxis, :])
            else:
                coordinates = self.coordinates

            if self.dims == 1:
                return DSTub1D(coordinates)
            else:
                return Delaunay(coordinates, qhull_options='Qz')

    @property
    def simplices(self):
        tri = self.tri
        if tri is None:
            raise RuntimeError("No triangulation available yet")
        else:
            return tri.simplices

    def append(self, x, **kwargs):
        ignore_duplicates = kwargs.pop("ignore_duplicates", False)
        if self.closest(x) < self.proximity_epsilon:
            if ignore_duplicates:
                return
            else:
                raise ValueError(f"point {x} duplicates another point")

        point = super().append(x, **kwargs)
        return point

    def sample(self, i):
        simplex = self.simplices[i]
        coordinates = self.coordinates[simplex, :]
        center = np.mean(coordinates, axis=0)
        dst = np.linalg.norm(coordinates - center[np.newaxis, :], axis=-1)
        max_dst = dst.max()
        mask = np.asanyarray(dst > self.snap_threshold * max_dst, dtype=float)
        if mask.sum() < 2:
            mask[np.argsort(dst)[:-3:-1]] = .5
        else:
            mask /= mask.sum()
        self.append(mask @ coordinates)
        return [self.size - 1]


class Guide(Iterable):
    def __init__(self, sampler, priority_tolerance=1e-9):
        """
        Guides the sampling.

        Args:
            sampler (Sampling): sampling to use;
            priority_tolerance (float): expected tolerance of priority values;
        """
        self.sampler = sampler
        self.priority_tolerance = priority_tolerance

    def set_bounds(self, bounds):
        """
        Set new bounds.

        Args:
            bounds (tuple): bounds defining the box;
        """
        for i in product(*bounds):
            self.sampler.append(i, ignore_duplicates=True)

    def set_prop(self, name, value):
        """
        Sets a property.

        Args:
            name (str): property name;
            value (str): property value;
        """
        if name == "bounds":
            self.set_bounds(str2ranges(value))
        else:
            self.sampler.set_prop(name, value)

    def __getstate__(self) -> dict:
        return dict(
            sampler=asdict(
                self.sampler,
                filter=filters.exclude(fields(Sampling).logger)
            ),
            priority_tolerance=self.priority_tolerance,
        )

    def __setstate__(self, state):
        state["sampler"] = DelaunaySampling(**state["sampler"])
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
                dump(self.to_json(), _f, **def_kwargs)
        else:
            dump(self.to_json(), f, **def_kwargs)

    @classmethod
    def from_json(cls, data):
        """
        Restores from JSON data.

        Args:
            data (dict): the serialized data;

        Returns:
            A new Guide.
        """
        data["sampler"] = DelaunaySampling(**data["sampler"])
        return cls(**data)

    @property
    def m_pending(self) -> np.ndarray:
        """Mask with points having 'pending' status."""
        return np.where(self.sampler.tags == "⏸")[0]

    @property
    def m_running(self) -> np.ndarray:
        """Indices of points with 'running' status."""
        return np.where(self.sampler.tags == "▶")[0]

    @property
    def m_done(self) -> np.ndarray:
        """Indices of points with 'done' status."""
        return np.where(self.sampler.tags == "✓")[0]

    @property
    def __any_nan_mask__(self) -> np.ndarray:
        return np.any(np.isnan(self.sampler.values), axis=-1)

    @property
    def m_done_numeric(self) -> np.ndarray:
        """Indices of points with numeric results."""
        return np.where((self.sampler.tags == "✓") * (np.logical_not(self.__any_nan_mask__)))[0]

    @property
    def m_done_nan(self) -> np.ndarray:
        """Indices ofpoints with non-numeric results."""
        return np.where((self.sampler.tags == "✓") * self.__any_nan_mask__)[0]

    def priority(self) -> np.ndarray:
        """
        Priority for sampling each of the triangles.
        Returns:
            An array with priorities: arbitrary real numbers.
        """
        raise NotImplementedError

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
        new_points = self.m_pending

        if len(new_points) == 0:
            priority = self.priority()

            # nans in priority
            nans = np.where(np.isnan(priority))[0]
            if len(nans) > 0:
                simplex = self.__choose_from_alternatives__(nans)
            else:
                # Largest volumes otherwise
                max_priority = np.max(priority)
                simplex = self.__choose_from_alternatives__(
                    np.where(priority >= max_priority - self.priority_tolerance)[0])
            new_points = self.sampler.sample(simplex)

        new_point = new_points[0]
        self.sampler.tags[new_point] = "▶"
        return new_point

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

    def priority(self) -> np.ndarray:
        """
        Priority for sampling each of the triangles.
        Priorities are simplex volumes in this implementation.

        Returns:
            An array with priorities: arbitrary real numbers.
        """
        return self.sampler.get_volumes()


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
        return alternatives[np.argmax(self.sampler.get_volumes()[alternatives])]

    def priority(self) -> np.ndarray:
        """
        Priority for sampling each of the triangles.
        This implementation takes into account local curvature,
        simplex volumes and the amount of NaNs in the data.

        Returns:
            An array with priorities.
        """
        result = self.sampler.get_curvature()

        if self.nan_threshold is not None:
            if np.isnan(result).sum() > self.nan_threshold * len(result):
                return self.sampler.get_volumes()

        if self.volume_ratio is not None:
            volumes = self.sampler.get_volumes()
            discard_mask = volumes / max(volumes) < 1. / self.volume_ratio
            result[discard_mask] = 0

        return result


class PointProcessPool:
    def __init__(self, target, guide, float_regex=r"([-+]?[0-9]+\.?[0-9]*(?:[eEdD][-+]?[0-9]+)?)|(nan)",
                 encoding="utf-8", limit=None, fail_limit=None, force_collect=False):
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
            force_collect (bool): if True, collects data for non-zero exit codes;
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
        self.force_collect = force_collect
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

    @property
    def sampler(self):
        return self.guide.sampler

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
                (self.target,) + tuple(map("{:.12e}".format, self.guide.sampler.coordinates[p])),
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

                # Floats in the output
                matches = tuple(i.group() for i in self.compiled_float_regex.finditer(out))

                if len(matches) >= self.sampler.dims_f:
                    kwargs = {}
                    if self.sampler.dims_f > 0:
                        kwargs["y"] = tuple(map(float, matches[-self.sampler.dims_f:]))
                    if not process.returncode or self.force_collect:
                        kwargs["meta"] = (out, err)
                    self.sampler.set(point, **kwargs)
                    if process.returncode:
                        self.__failed__ += 1
                    else:
                        self.__succeeded__ += 1
                else:
                    print("Missing {:d} numbers in the output".format(self.sampler.dims_f))
                    self.__failed__ += 1

                self.sampler.tags[point] = "✓"

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
        volume_ratio=10, rescale=True, save=True, load=None, on_update=None, web_server=False, force_collect=False):
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

        web_server (bool): creates a web server for runtime interaction;

        force_collect (bool): if True, collects data for non-zero exit codes;
    """
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG if verbose else logging.WARNING,
        format="[%(levelname)s] %(name)s %(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("hello")

    if on_update is not None:
        def on_plot_update_fun(guide):
            try:
                p = subprocess.Popen(on_update, stdin=subprocess.PIPE, shell=True, encoding="utf-8")
                p.stdin.write(dumps(guide.to_json()))
                p.stdin.close()
            except BrokenPipeError:
                logging.warning("on_update process refused to receive JSON data")
    else:
        def on_plot_update_fun(guide):
            pass

    if load is not None:
        logging.info(f"loading from '{load}' ...")
        with open(load, 'r') as f:
            guide = (UniformGuide if dims_f == 0 else CurvatureGuide).from_json(json_load(f))
        # Update the guide with the new parameters
        guide.set_bounds(bounds)
        guide.sampler.snap_threshold = snap_threshold
        if guide.sampler.dims_f != dims_f:
            logging.warning(f"ignoring dims_f={dims_f}, using the restored value {guide.dims_f} instead")
        guide.sampler.rescale = rescale
        if isinstance(guide, CurvatureGuide):
            guide.nan_threshold = nan_threshold
            guide.volume_ratio = volume_ratio
    else:
        sampler = DelaunaySampling(
            np.empty((0,), dtype=point_record_dtype(len(bounds), dims_f)),
            rescale=rescale, snap_threshold=snap_threshold)
        if dims_f == 0:
            guide = UniformGuide(sampler)
        else:
            guide = CurvatureGuide(sampler, nan_threshold=nan_threshold, volume_ratio=volume_ratio)
        guide.set_bounds(bounds)

    if web_server:

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
        logging.critical(f"listening for localhost:{port}/{secret}")

    if save not in (None, False):
        if save is True:
            data_folder = Path(".curious")
            save = data_folder / timestamp() / ".json"
        else:
            save = Path(save)
            data_folder = save.parent

        if not data_folder.exists():
            logging.info(f"creating data folder '{data_folder}' ...")
            os.mkdir(data_folder)

        if not data_folder.is_dir():
            raise RuntimeError(f"Not a folder: {data_folder}")

    ppp = PointProcessPool(target, guide, limit=limit, fail_limit=fail_limit, force_collect=force_collect)

    on_plot_update_fun(guide)

    while not ppp.is_finalized:

        try:
            if web_server:
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
                logging.info(f"running: {ppp.running} (+{spawned} -{swept}) \tcompleted: {ppp.completed}[{len(guide.m_done)}] "
                  f"({ppp.failed} fails) \t{'DRAIN' if ppp.draining else ''}")
                on_plot_update_fun(guide)
                if save:
                    guide.save_to(save)

            time.sleep(0.1)

        except KeyboardInterrupt:
            if ppp.draining:
                raise
            else:
                logging.critical("ctrl-c caught: finalizing ... (press ctrl-c once more to abort now)")
                ppp.start_drain()

    logging.info("bye")

    return ppp.failed > ppp.fail_limit

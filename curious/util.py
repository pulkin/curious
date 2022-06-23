from datetime import datetime, timedelta
import json
import numpy as np
from functools import partial


def limit2str(limit) -> str:
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


def str2limit(limit):
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


def str2ranges(ranges):
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


def np_tolist(a):
    """
    Transforms numpy into a nested list.

    Args:
        a: the array to transform;

    Returns:
        The resulting nested list.
    """
    return a.tolist()


def np_array_to_json(a):
    """
    Transforms numpy array into json-friendly dict.

    Args:
        a (np.ndarray): array to transform;

    Returns:
        The resulting json-friendly dict.
    """
    return {
        "_type": "numpy",
        "dtype": a.dtype.descr,
        "shape": a.shape,
        "data": np_tolist(a),
    }


def json_to_np_array(j):
    """
    Transforms json into a numpy array.

    Args:
        j: json with array data;

    Returns:
        The resulting array.
    """
    def _make_tuples(obj, level):
        if level == 0:
            return tuple(obj) if isinstance(obj, list) else obj
        else:
            return list(_make_tuples(i, level - 1) for i in obj)

    return np.array(_make_tuples(j["data"], len(j["shape"])), dtype=np.dtype(list(map(tuple, j["dtype"]))))


class ArrayEncoder(json.JSONEncoder):
    """Serialize numpy arrays."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return np_array_to_json(o)
        else:
            return super(ArrayEncoder, self).default(o)


def ndarray_object_hook(d):
    """Deserialize numpy arrays"""
    if d.get("_type", None) == "numpy":
        return json_to_np_array(d)
    else:
        return d


dump = partial(json.dump, cls=ArrayEncoder)
dumps = partial(json.dumps, cls=ArrayEncoder)
load = partial(json.load, object_hook=ndarray_object_hook)
loads = partial(json.loads, object_hook=ndarray_object_hook)

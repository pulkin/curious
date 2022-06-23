import numpy as np


def check_point_data(data, names=("x", "y", "tag", "meta"), ):
    """
    Verifies the correctness of point data format.

    Args:
        data (np.ndarray): the data to check;
        names (tuple): expected record field names;
    """
    if not isinstance(data, np.ndarray):
        raise ValueError(f"not an array: {data}")
    dtype = data.dtype
    if dtype.names is None:
        raise ValueError(f"not an array of records with dtype={dtype}")
    if dtype.names != names:
        raise ValueError(f"dtype.names = {dtype.names} != {names}")

    # x
    dtype_x, _ = dtype.fields["x"]
    subdtype_x, shape_x = dtype_x.subdtype
    if len(shape_x) != 1:
        raise ValueError(f"dtype(x).shape = {shape_x} is not a plain 1D")
    if subdtype_x != np.float_:
        raise ValueError(f"subdtype(x) = {subdtype_x} is not float")

    # y
    dtype_y, _ = dtype.fields["y"]
    subdtype_y, shape_y = dtype_y.subdtype
    if len(shape_y) != 1:
        raise ValueError(f"dtype(y).shape = {shape_y} is not a plain 1D")
    if subdtype_y != np.float_:
        raise ValueError(f"subdtype(y) = {subdtype_y} is not float")

    # tag
    dtype_tag, _ = dtype.fields["tag"]
    if dtype_tag != np.dtype('U1'):
        raise ValueError(f"dtype(tag) = {dtype_tag} is not a string")

    # meta
    dtype_meta, _ = dtype.fields["meta"]
    if dtype_meta != object:
        raise ValueError(f"dtype(meta) = {dtype_meta} is not object type")

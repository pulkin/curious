import numpy as np


def point_record_dtype(dims: int, dims_f: int) -> np.dtype:
    """
    Creates a numpy record type to store point data.

    Args:
        dims: the dimensionality of the parameter space to sample;
        dims_f: the dimensionality of the output (target space);

    Returns:
        The resulting type.
    """
    return np.dtype([
            ('x', np.float_, (dims,)),
            ('y', np.float_, (dims_f,)),
            ('tag', np.unicode_, 1),
            ('meta', object, 1),
        ])


def get_point_record_dtype_dims(dtype: np.dtype) -> (int, int):
    """
    Retrieves dimensions from point record data type.
    Inverse of `point_record_dtype`.

    Args:
        dtype: numpy record data type;

    Returns:
        Two integers: sizes of input and output dimensions.
    """
    if dtype.names is None:
        raise ValueError(f"not an array of records with dtype={dtype}")

    names = "x", "y", "tag", "meta"
    if dtype.names != names:
        raise ValueError(f"dtype.names = {dtype.names} != {names}")

    dtype_x, _ = dtype.fields["x"]
    subdtype_x, shape_x = dtype_x.subdtype
    if len(shape_x) != 1:
        raise ValueError(f"dtype(x).shape = {shape_x} is not a plain 1D")
    if subdtype_x != np.float_:
        raise ValueError(f"subdtype(x) = {subdtype_x} is not float")
    dims, = shape_x

    dtype_y, _ = dtype.fields["y"]
    subdtype_y, shape_y = dtype_y.subdtype
    if len(shape_y) != 1:
        raise ValueError(f"dtype(y).shape = {shape_y} is not a plain 1D")
    if subdtype_y != np.float_:
        raise ValueError(f"subdtype(y) = {subdtype_y} is not float")
    dims_f, = shape_y

    dtype_tag, _ = dtype.fields["tag"]
    if dtype_tag != np.dtype('U1'):
        raise ValueError(f"dtype(tag) = {dtype_tag} is not a string")

    dtype_meta, _ = dtype.fields["meta"]
    if dtype_meta != object:
        raise ValueError(f"dtype(meta) = {dtype_meta} is not object type")

    return dims, dims_f

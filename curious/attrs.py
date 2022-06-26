from .data import get_point_record_dtype_dims

import numpy as np


def check_point_data(data):
    """
    Verifies the correctness of point data format.

    Args:
        data (np.ndarray): the data to check;
    """
    if not isinstance(data, np.ndarray):
        raise ValueError(f"not an array: {data}")

    get_point_record_dtype_dims(data.dtype)

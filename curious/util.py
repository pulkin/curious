from datetime import datetime, timedelta


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

from . import run
from .util import limit2str, str2ranges, str2limit

import argparse
import inspect


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
                    metavar="LIMIT", type=str, default=limit2str(defaults["limit"]))
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
parser.add_argument("--web-server", help="creates a web server to listen during runtime", action="store_true")
parser.add_argument("--force-collect", help="collect data from non-zero exit codes", action="store_true")

options = parser.parse_args()

exit(run(
    target=options.target,
    bounds=str2ranges(options.ranges),
    dims_f=options.n,
    verbose=options.verbose,
    depth=options.depth,
    fail_limit=options.fail_limit,
    limit=str2limit(options.limit),
    snap_threshold=options.snap_threshold,
    nan_threshold=options.nan_threshold,
    volume_ratio=options.volume_ratio,
    rescale=not options.no_rescale,
    save=options.save if options.save is not True else not options.no_io,
    load=options.load,
    on_update=options.on_update,
    web_server=options.web_server,
    force_collect=options.force_collect,
))

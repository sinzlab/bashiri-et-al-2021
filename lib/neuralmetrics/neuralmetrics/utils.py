import os, sys
from warnings import warn
import cProfile, pstats, io
import numpy as np


def fitted_zig_mean(theta, k, loc, q, approximate=False):
    uniform_mean = 0
    if not approximate:
        uniform_mean += (1 - q) * 0.5 * loc
    gamma_mean = q * (k * theta + loc)
    return uniform_mean + gamma_mean


def fitted_zig_variance(theta, k, loc, q, approximate=False):
    expectation_of_variances = q * k * theta ** 2
    if not approximate:
        expectation_of_variances += (1 - q) * loc ** 2 / 12

    variance_of_expectations = q * (1 - q) * (k * theta + loc / 2) ** 2
    return expectation_of_variances + variance_of_expectations


def normalize(aa):
    bb = aa - aa.min()
    return bb / bb.max()


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_slope(x, y):
    """Returns the slope of a linear regressor with intercept zero."""
    if (sum(x < 0) > 0) or (sum(y < 0) > 0):
        warn("x and/or y contain negative values. Slope may not reflect the desired quantity.")
    return np.linalg.lstsq(np.vstack([x, np.zeros(len(x))]).T, y, rcond=None)[0][0]


def bits_per_image(nits, n_images):
    """

    Args:
        nits:     array containing entropy in nits (natural logarithm, base e)
        n_images: number of images to normalize by

    Returns:      array containing entropy per image in bits (logarithm with base 2)

    """
    return nits / (np.log(2) * n_images)

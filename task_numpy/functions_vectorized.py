import numpy as np


def prod_non_zero_diag(x: np.array):
    x_diag = x.diagonal()
    x_filter = x_diag != 0
    x_filtred = x_diag[x_filter]

    return np.prod(x_filtred)

prod_non_zero_diag
def are_multisets_equal(x:np.array, y:np.array):
    return np.array_equal(np.sort(x.flat), np.sort(y.flat))


def max_after_zero(x:np.array):
    x_filter = (x==0)
    x_shift_filter = np.insert(x_filter[:-1], 0, False)
    return np.max(x[x_shift_filter])
    


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """

    pass


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """

    pass


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """

    pass
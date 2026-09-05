import numpy as np

def prod_non_zero_diag(x:list):
    multiplayed_x = 1
    for i in range(len(x)):
        if x[i][i] != 0:
            multiplayed_x *= x[i][i]
    return multiplayed_x


def are_multisets_equal(x:list, y:list):
    return sorted(x) == sorted(y)
    


def max_after_zero(x:list):
    after_zero_n = min(x)
    for i in range(1, len(x)):
        if x[i-1] == 0 and x[i] > after_zero_n:
            after_zero_n = x[i]

    return after_zero_n
    


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """

    pass


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """

    pass


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """

    pass

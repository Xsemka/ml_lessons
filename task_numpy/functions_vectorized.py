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
    


def convert_image(img: np.array, coefs:np.array):
    return np.sum(np.multiply(img, coefs), axis=1)


def run_length_encoding(x: np.array):
    changes = x[1:] != x[:-1]
    change_indexes = np.flatnonzero(changes)
    borders = np.concat([[-1], change_indexes, [len(x) - 1]])
    differences = np.diff(borders)
    value_indexes = np.concat([[0], change_indexes+1])
    return x[value_indexes], differences

# x = np.array([1,1,1,2,2,3,3,3,3])
# print(run_length_encoding(x))


def pairwise_distance(x: np.array, y: np.array):
    return np.sqrt(np.sum((x - y)**2))
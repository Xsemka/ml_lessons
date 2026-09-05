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
    for i in range(len(img)):
        for j in range(len(img[i])):
            for k in range(len(img[i][j])):
                img[i][j][k] *= coefs[k]

            img[i][j] = sum(img[i][j])

    return img

    


def run_length_encoding(x):
    counter = 1
    length_number = []
    values = [x[0]]
    for i in range(1, len(x)):
        if x[i] != x[i-1]:
            values.append(x[i])
            length_number.append(counter)
            counter = 0

        counter += 1

    length_number.append(counter)

    return values, length_number

# x = np.array([1,1,1,2,2,3,3,3,3]).tolist()
# print(run_length_encoding(x))


def pairwise_distance(x, y):
    return sum((x[i] - y[i])**2 for i in range(len(x))) ** 0.5

import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
import torch
from torch import nn
from sklearn.linear_model import SGDRegressor, QuantileRegressor
from sklearn.metrics import mean_absolute_error
torch.set_default_device("cuda")

def task_a1():
    for n in range(100, 0, -1):
        power = 2 ** n
        if 123456123456123456 % power == 0:
            x = (123456123456123456// power) - 1
            if x >= 1:
                return x
            





def task_b1():
    #т.к. для данных входных значений на выходе нет 0 и 255 значит функция clip не применятась. Т.к. на входе 2 значения, а на выходе 3 => матрица имеет формат 2х3, при а так как нет слоёв аквтивации, то вся функция линейна и веса матрицы можно найти системой уравнений и вычислить ответ вообще без программирования, а потом применить на ответ clip
    return torch.tensor([0, 70, 220])



def task_c1(total_sim, batch_size):
    max_sub = -1

    for batch in range(total_sim):
        tnsr = torch.rand(batch_size, 10)
        tnsr, _ = torch.sort(tnsr)
        valid_mask = (torch.max(torch.diff(tnsr, dim=1), dim=1).values <= 0.25) & (tnsr[:,0] <= 0.25) & (tnsr[:,-1] >= 0.75) 
        tnsr = tnsr[valid_mask]

        local_max = torch.max(torch.abs(torch.mean(tnsr, dim=1) - torch.median(tnsr, dim=1).values), dim=0).values.item()
        if local_max > max_sub:
            max_sub = local_max

        if batch % 500 == 0:
            print(f"batch {batch} is completed with max sub {max_sub}")


    return max_sub



def task_d1():
    points = [(0, 0),(0, 2),(0, 7),(0, 8),
              (4, 4),(4, 7),
              (6, 1),(6, 5),
              (9, 0),(9, 1),(9, 4),(9, 7),
              (11, 2),(11, 6)]

    x = np.array([p[0] for p in points]).reshape(-1, 1)
    y = np.array([p[1] for p in points])

    model = QuantileRegressor(alpha=0.0)
    model.fit(x, y)
    pred = model.predict(x)
    score = mean_absolute_error(y, pred)
    return float(f"{score:.6f}")

def task_e1():
    pass

def task_f1(): #решаем по формуле N = (I - Q)^-1 где I это еденичная матрица, а Q матрица переходов с вероятностями 
    сhar = {"A":0, "D":1, "E":2, "I":3, "L":4, "M":5, "N":6, "S":7, "T":8}
    #     0A, 1D, 2E, 3I, 4L, 5M, 6N, 7S, 8T 
    #  0A
    #  1D
    #  2E
    #  3I
    #  4L
    #  5M
    #  6N
    #  7S
    #  8T

    Q = torch.zeros((9, 9), dtype=torch.double)
    Q[0, 1] = 1/2; Q[0, 5] = 1/2
    Q[1, 0] = 1/2; Q[1, 3] = 1/2
    Q[2, 5] = 1/3; Q[2, 7] = 1/3; Q[2, 8] = 1/3
    Q[3, 5] = 1/3; Q[3, 7] = 1/3; Q[3, 8] = 1/3
    Q[4, 1] = 1/3; Q[4, 2] = 1/3; Q[4, 3] = 1/3
    Q[5, 4] = 0; Q[5, 6] = 1/2
    Q[6, 0] = 1/2; Q[6, 3] = 1/2
    Q[7, 1] = 1/3; Q[7, 2] = 1/3; Q[7, 3] = 1/3
    Q[8, 4] = 1/2; Q[8, 6] = 1/2

    I = torch.eye(9, dtype=torch.double)
    N = torch.inverse(I - Q)
    return torch.sum(N[7]).item()





print(task_f1())
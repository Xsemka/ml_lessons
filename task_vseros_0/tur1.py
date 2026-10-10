import numpy as np
import pandas as pd
import itertools
from scipy.optimize import minimize, linprog
import torch
import math
from torch import nn
from sklearn.linear_model import SGDRegressor, QuantileRegressor
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm
from torch.optim import Adam
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
        valid_mask = (torch.max(torch.diff(tnsr, dim=1), dim=1).values <= 0.25) & (tnsr[:,0] <= 0.25) & (tnsr[:,-1] >= 0.75) & (torch.min(torch.diff(tnsr, dim=1), dim=1).values >= 0.01)
        tnsr = tnsr[valid_mask]
        if tnsr.shape[0] == 0:
            continue

        local_max = torch.max(torch.abs(torch.mean(tnsr, dim=1) - (tnsr[:, 4] + tnsr[:, 5])/2), dim=0).values.item()
        if local_max > max_sub:
            max_sub = local_max

        if batch % 500 == 0:
            print(f"batch {batch} is completed with max sub {max_sub}")


    return max_sub

def task_c1_grad(epochs, k = 100, lr = 10**-3):
    x = torch.rand(10, requires_grad=True)
    optimizer = Adam([x], lr=lr)

    max_diff = 0.0
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_sorted, _ = torch.sort(x)

        mean = torch.mean(x_sorted)
        median = (x_sorted[4] + x_sorted[5]) / 2.0
        diffs = torch.diff(x_sorted)

        target_loss = -torch.abs(mean - median)

        pentality = 0.0

        pentality += k * torch.sum(torch.relu(-x_sorted))
        pentality += k * torch.sum(torch.relu(x_sorted - 1.0))
        pentality += k * torch.sum(torch.relu(0.01 - diffs))
        pentality += k * torch.sum(torch.relu(diffs - 0.25))
        pentality += k * torch.relu(x_sorted[0] - 0.25)
        pentality += k * torch.relu(0.75 - x_sorted[9])

        loss = target_loss + pentality

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            with torch.no_grad():
                x_check, _ = torch.sort(x)
                c_diff = torch.diff(x_check)

                valid = (x_check[0] <= 0.25 and x_check[9] >= 0.75 and torch.all(x_check >= 0) and torch.all(x_check <= 1) and torch.all(c_diff >= 0.01) and torch.all(c_diff <= 0.25))
                if valid:
                    c_mean = torch.mean(x_check)
                    c_median = (x_check[4] + x_check[5]) / 2.0
                    c_diff = torch.abs(c_mean - c_median)

                    if max_diff < c_diff:
                        max_diff = c_diff
            print(f"эпоха {epoch} завершилась с максимальной разностью {max_diff}")

    return max_diff



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

def task_e1(n = 4, ar = 0.1):
    max_accuracy = torch.tensor([0.0, 0.0, 0.0])

    field1 = torch.tensor([
        [1,1,1,1,0,1,1,0,1,1,1,1],
        [1,0,0,1,0,1,1,0,1,0,0,0],
        [1,1,1,1,0,1,1,0,1,0,0,0],
        [1,0,0,1,0,1,1,0,1,0,0,0],
        [1,0,0,1,0,0,0,0,1,1,1,1]
    ])

    field2 = torch.tensor([
        [1,0,0,0,1,0,0,0,1,0,0,1,1],
        [1,0,0,0,1,0,0,0,1,1,0,1,1],
        [1,0,0,0,1,0,0,0,1,0,1,0,1],
        [1,0,0,1,1,0,0,0,1,0,0,0,1],
        [1,1,1,0,1,1,1,0,1,0,0,0,1]
    ])

    field3 = torch.tensor([
        [1,1,1,0,0,1,0,0,0],
        [1,0,1,1,0,1,0,0,0],
        [1,0,0,1,0,1,0,0,0],
        [1,0,0,1,0,1,0,0,0],
        [1,1,1,1,0,1,1,1,0],
    ])

    fields = [field1, field2, field3]

    thetas_deg = torch.arange(-90, 90, ar)
    thetas_rad = thetas_deg * (torch.pi / 180)
    k = torch.tan(thetas_rad)

    for f_indx, field in enumerate(tqdm(fields, desc="Обработка полей")):
        h, w = field.shape
        x = (torch.rand(10**n) - 0.5) * w
        y = (torch.rand(10**n) - 0.5) * h
        cols = torch.floor(x + w/2).int()
        rows = torch.floor(y + h/2).int()

        targets = field[rows, cols].unsqueeze(1).float()

        side = y.unsqueeze(1) - k.unsqueeze(0) * x.unsqueeze(1)

        pred_up = (side > 0).float()
        pred_down = (side <= 0).float()

        acc_up = (pred_up == targets).float().mean(dim=0)
        acc_down = (pred_down == targets).float().mean(dim=0)

        vertical_side = x.unsqueeze(1)
        pred1 = (vertical_side > 0).float()
        pred2 = (vertical_side <= 0).float()

        acc_v1 = (pred1 == targets).float().mean(dim=0)
        acc_v2 = (pred2 == targets).float().mean(dim=0)
        maxv = torch.max(acc_v1, acc_v2)

        best_angle_acc = torch.max(torch.max(acc_up), torch.max(acc_down))
        max_accuracy[f_indx] = torch.max(best_angle_acc, maxv)

    return max_accuracy

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

def task_g1():
    solve = []
    fibs = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597]
    r = 1
    while 2**r <= 2025:
        for combination in itertools.combinations_with_replacement(fibs, r):
            prod = math.prod(combination)
            if 1500 <= prod <= 2025:
                solve.append(prod - 1)
    return sorted(solve)






def task_h1():
    pass


print(task_c1_grad(10**6, 100, lr=10**-4))




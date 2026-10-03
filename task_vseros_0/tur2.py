import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
import torch
from torch import nn
from sklearn.linear_model import SGDRegressor, QuantileRegressor
from sklearn.metrics import mean_absolute_error
torch.set_default_device("cuda")
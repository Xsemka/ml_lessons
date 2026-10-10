import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
import torch
import networkx as nx
from torch import nn
from sklearn.linear_model import SGDRegressor, QuantileRegressor
from sklearn.metrics import mean_absolute_error
torch.set_default_device("cuda")

def task_a2():
    ans = {}
    #a1
    df = pd.read_csv("task_vseros_0/items_A.csv", index_col="item_id")
    ans.update({"a1" : df[(df["category"] == "phones") & (df["rating"] >= 4.5) & (df["in_stock"] == 1)].shape[0]})

    #a2
    df = pd.read_csv("task_vseros_0/items_A.csv", index_col="item_id")
    ans.update({"a2" : df[df["category"] == "laptops"].groupby("brand")["price"].mean().idxmax()})

    #a3
    df = pd.read_csv("task_vseros_0/items_A.csv", index_col="item_id")
    ans.update({"a3" : df[(df["rating"] >= 4.5) & (df["price"] >= 50000) & (df["in_stock"] == 1)].shape[0]})

    #a4
    df = pd.read_csv("task_vseros_0/items_A.csv", index_col="item_id")
    av = pd.read_csv("task_vseros_0/also_viewed_A.csv")
    a4_df = df.iloc[pd.unique(av["item_to"])].groupby("category").size()
    a4_df = a4_df.to_frame(name = "cnt")
    a4_df.index.name = "category"
    a4_df.to_csv("task_vseros_0/answer4.csv")
    ans.update({"a4" : "task_vseros_0/answer4.csv"})

    #a5
    df = pd.read_csv("task_vseros_0/items_A.csv", index_col="item_id")
    av = pd.read_csv("task_vseros_0/also_viewed_A.csv")
    G = nx.Graph()
    edges = zip(av["item_from"], av["item_to"])
    G.add_edges_from(edges)
    item_cat = df["category"]

    islands = list(nx.connected_components(G))

    matching_islands_count = 0

    for island in islands:
        cat_islands = {item_cat.get(item) for item in island if item in item_cat}

        if "phones" in cat_islands and "accessories" in cat_islands:
            matching_islands_count += 1

    ans.update({"a5" : matching_islands_count})

    return ans

def task_b2():
    pass
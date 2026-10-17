import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize, linprog
import torch
import networkx as nx
from torch import nn
from sklearn.linear_model import SGDRegressor, QuantileRegressor
from sklearn.metrics import mean_absolute_error, adjusted_rand_score
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
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
    data_raw = pd.read_csv("task_vseros_0/data_B.csv")
    true_labels = pd.read_csv("task_vseros_0/ytest_B.csv")
    true_labels = true_labels["cluster"]

    sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)

    def objective(trial):
        scaler = trial.suggest_categorical("scaler", ["standart", "robust"])
        match scaler:
            case "standart":
                scaler = StandardScaler()

            case "robust":
                scaler = RobustScaler()

        data_scaled = scaler.fit_transform(data_raw)

        contamination = trial.suggest_float("contamination" ,0, 0.50)
        iso_mask = IsolationForest(contamination=contamination, n_jobs=-1).fit_predict(data_scaled)
        data = data_scaled[iso_mask == 1]

        model_clust = trial.suggest_categorical("model", ["kmeans", "gmm"])
        n_clusters = trial.suggest_int("n_clusters", 2, 4)

        match model_clust:
            case "gmm":
                model = GaussianMixture(n_components=n_clusters).fit(data)

            case "kmeans":
                model = KMeans(n_clusters=n_clusters).fit(data)

        pred = model.predict(data_scaled)
        score = adjusted_rand_score(pred, true_labels)
        return score

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    study.optimize(objective, n_trials= 500, n_jobs= -1)
    return study.best_value


def task_c2():
    scaler = StandardScaler()
    ore = OrdinalEncoder()

    data = pd.read_csv("task_vseros_0/train_C.csv")
    data.columns = data.columns.str.lower()
    data.drop("json", inplace=True)
    data.drop(inplace=True)
    iso_mask = IsolationForest(contamination=0.05, n_jobs=-1).fit_predict(data)
    data = data[iso_mask == 1]
    data_encoded = ore.fit_transform(data[["apartment_type", "metro_station", "region", "Renovation"]])
    data_scaled = scaler.fit_transform(data_encoded)

    y = data_scaled["price"]
    data_scaled.drop("price", inplace = True)
    x = data_scaled.copy()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7)

    model = RandomForestRegressor()

#хуй

        

task_b2()
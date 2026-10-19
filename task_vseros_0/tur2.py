import numpy as np
import pandas as pd
import optuna
from scipy.optimize import minimize, linprog
import torch
import networkx as nx
from torch import nn
from sklearn.linear_model import SGDRegressor, QuantileRegressor
from sklearn.metrics import mean_absolute_error, adjusted_rand_score, mean_squared_error, root_mean_squared_error
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from model_lib import getModel
import matplotlib.pyplot as plt
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
        scaler = trial.suggest_categorical("scaler", ["standard", "robust"])
        match scaler:
            case "standard":
                scaler = StandardScaler()

            case "robust":
                scaler = RobustScaler()

        data_scaled = scaler.fit_transform(data_raw)

        contamination = trial.suggest_float("contamination" ,0, 0.50)
        iso_mask = IsolationForest(contamination=contamination).fit_predict(data_scaled)
        data = data_scaled[iso_mask == 1]

        model_clust = trial.suggest_categorical("model", ["kmeans", "gmm", "dbscan", "spectral"])
        

        match model_clust:
            case "gmm":
                n_clusters = trial.suggest_int("n_clusters", 2, 4)
                model = GaussianMixture(n_components=n_clusters).fit(data)
                pred = model.predict(data_scaled)

            case "kmeans":
                n_clusters = trial.suggest_int("n_clusters", 2, 4)
                model = KMeans(n_clusters=n_clusters).fit(data)
                pred = model.predict(data_scaled)

            case "dbscan":
                epsilon = trial.suggest_float("epsilon" ,0, 1)
                p = trial.suggest_int("p", 3, 10)
                metric = trial.suggest_categorical("metric", ["euclidean", "minkowski", "manhattan"])
                model = DBSCAN(eps=epsilon, metric=metric, p=p).fit(data)
                knn = KNeighborsClassifier(n_neighbors=1).fit(data, model.labels_)
                pred = knn.predict(data_scaled)

            case "spectral":
                n_clusters = trial.suggest_int("n_clusters", 2, 4)
                model = SpectralClustering(n_clusters=n_clusters, eigen_solver="arpack").fit(data)
                knn = KNeighborsClassifier(n_neighbors=1).fit(data, model.labels_)
                pred = knn.predict(data_scaled)

        
        score = adjusted_rand_score(pred, true_labels)
        return score

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    study.optimize(objective, n_trials= 500, n_jobs= 1)
    return study.best_value


def task_c2():
    models = getModel(100)
    sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)

    scaler = StandardScaler()
    ore = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    pca = PCA(n_components=4)

    

    data = pd.read_csv("task_vseros_0/train_C.csv")
    data.columns = data.columns.str.lower()
    data.drop("json", inplace=True, axis=1)
    data.dropna(inplace=True)
    data.fillna(0, inplace=True)
    data.replace(",", ".", regex=True, inplace=True)

    cat_cols = ["apartment_type", "metro_station", "region", "renovation"]
    num_cols = data.drop("price", axis=1).select_dtypes(include=['number']).columns
    
    data[cat_cols] = ore.fit_transform(data[cat_cols])

    iso_mask = IsolationForest(contamination=0.05, n_jobs=-1).fit_predict(data)
    data = data[iso_mask == 1]

    y_train = data["price"]

    data.drop("price", inplace = True, axis = 1)
    x_train = data.copy()
    train_median = x_train[num_cols].mean().round()
    x_train[num_cols] = x_train[num_cols].fillna(train_median)
    x_train[cat_cols] = x_train[cat_cols].fillna(-1)
    x_train = scaler.fit_transform(x_train)
    x_train = pca.fit_transform(x_train)
    
    y_test = pd.read_csv("task_vseros_0/ytest_C.csv", index_col="ID")

    x_test = pd.read_csv("task_vseros_0/test_C.csv", index_col="ID")
    x_test.columns = x_test.columns.str.lower()
    x_test.drop("json", inplace=True, axis=1)
    x_test.replace(",", ".", regex=True, inplace=True)
    x_test[cat_cols] = ore.transform(x_test[cat_cols])
    x_test[num_cols] = x_test[num_cols].fillna(train_median)
    x_test[cat_cols] = x_test[cat_cols].fillna(-1)
    x_test = scaler.transform(x_test)
    x_test = pca.transform(x_test)

    def objective(trial):
        model_type = trial.suggest_categorical("model", ["linear", "svm", "knn", "tree", "forest", "catboost"])
        match model_type:
            case "linear":
                model = models.linear_regression(x_train, x_test, y_train, y_test)
            case "svm":
                model = models.svm_regressor(x_train, x_test, y_train, y_test)
            case "knn":
                model = models.KNN_regressor(x_train, x_test, y_train, y_test)
            case "tree":
                model = models.tree_regressor(x_train, x_test, y_train, y_test)
            case "forest":
                model = models.random_forest_regressor(x_train, x_test, y_train, y_test) 
            case "catboost":
                model = models.cat_boost_regression(x_train, x_test, y_train, y_test)

        pred = model.predict(x_test)
        score = root_mean_squared_error(pred, y_test)
        return score

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="minimize")
    study.optimize(objective, n_trials=100)
    return study.best_value


print(task_c2())
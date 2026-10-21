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
from torchmetrics import Accuracy
from model_lib import getModel
import matplotlib.pyplot as plt



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
    models = getModel(300)
    sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)

    scaler = StandardScaler()
    ore = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    pca = PCA(n_components=3)

    apartment_type_sinonims = {"Вторичка" : "Вторичка",
                               "Вторичный рынок" : "Вторичка",
                               "Вторичное жилье" : "Вторичка",
                               "Новостройка" : "Новостройка",
                               "Первичка" : "Новостройка",
                               "От застройщика" : "Новостройка",
                               "Старый фонд" : "Вторичка",
                               "New building" : "Новостройка",
                               "Secondary housing" : "Вторичка",
                               "Новая" : "Новостройка",
                               "Б/У квартира" : "Вторичка",
                               "Новое строительство" : "Новостройка",
                               "Свежая постройка" : "Новостройка",
                               "Старая квартира" : "Вторичка",
                               "Не новостройка" : "Вторичка",
                               "Новая квартира" : "Новостройка",
                               "Современный комплекс" : "Новостройка",
                               "Вторичное" : "Вторичка",
                               "Вторичная квартира" : "Вторичка"
                               }

    renovation_sinonims = {"Косметика" : "Косметика",
                           "Космтичский ремонт" : "Косметика",
                           "Евроремонт" : "Евроремонт",
                           "Евро" : "Евроремонт",
                           "Без ремонта" : "Без ремонта",
                           "Нет Ремонта" : "Без ремонта",
                           "Дизайнерский" : "Дизайнерский",
                           "Дизайн" : "Дизайнерский",}

    data = pd.read_csv("task_vseros_0/train_C.csv")
    data.drop("json", inplace=True, axis=1)
    data.columns = data.columns.str.lower()
    nan_cols = ["area", "living_area", "kitchen_area", "notliving_area"]
    data[nan_cols] = data[nan_cols].fillna(data[nan_cols].median())
    data.dropna(inplace=True)
    data["apartment_type"] = data["apartment_type"].map(lambda x: apartment_type_sinonims[x])
    data["renovation"] = data["renovation"].map(lambda x: renovation_sinonims[x])
    data["metro_station"] = data["metro_station"].str.lower()
    data["metro_station"] = data["metro_station"].str.replace(" ", "", regex=True)
    for trash in ["домна", "москва:", "метро", "станция"]:
        data["metro_station"] = data["metro_station"].str.replace(trash, "", regex=True)
    data.replace(",", ".", regex=True, inplace=True)

    x_train = data.drop("price", axis=1)
    y_train = data["price"]

    y_test = pd.read_csv("task_vseros_0/ytest_C.csv", index_col="ID")

    x_test = pd.read_csv("task_vseros_0/test_C.csv", index_col="ID")
    x_test.drop("json", inplace=True, axis=1)
    x_test.columns = x_test.columns.str.lower()
    nan_cols = ["area", "living_area", "kitchen_area", "notliving_area"]
    x_test[nan_cols] = x_test[nan_cols].fillna(x_test[nan_cols].median())
    x_test.dropna(inplace=True)
    x_test["apartment_type"] = x_test["apartment_type"].map(lambda x: apartment_type_sinonims[x])
    x_test["renovation"] = x_test["renovation"].map(lambda x: renovation_sinonims[x])
    x_test["metro_station"] = x_test["metro_station"].str.lower()
    x_test["metro_station"] = x_test["metro_station"].str.replace(" ", "", regex=True)
    for trash in ["домна", "москва:", "метро", "станция"]:
        x_test["metro_station"] = x_test["metro_station"].str.replace(trash, "", regex=True)
    x_test.replace(",", ".", regex=True, inplace=True)

    x_train[["apartment_type", "renovation", "region", "metro_station"]] = ore.fit_transform(x_train[["apartment_type", "renovation", "region", "metro_station"]])
    x_test[["apartment_type", "renovation", "region", "metro_station"]] = ore.transform(x_test[["apartment_type", "renovation", "region", "metro_station"]])
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = models.cat_boost_regression(x_train, x_test, y_train, y_test)
    pred = model.predict(x_test)
    score = root_mean_squared_error(y_test, pred)
    return score



def task_d2(epochs = 150, batch_size = 64, lr = 0.001):
    torch.set_default_device("cuda")
    metric = Accuracy(task="multiclass", num_classes=25)
    criterion = nn.CrossEntropyLoss()

    data = pd.read_csv("task_vseros_0/train_D.csv", index_col="id", sep=",")
    x_train = data.drop("labels", axis=1).to_numpy(dtype=np.float32)

    labels = data["labels"].str.get_dummies(sep=" ")
    labels = labels.reindex(columns=[str(i) for i in range(25)], fill_value=0).to_numpy(dtype=np.float32)
    y_train = labels / labels.sum(axis = 1, keepdims=True)

    x_test = pd.read_csv("task_vseros_0/test_D.csv", index_col="id", sep=",").to_numpy(dtype=np.float32)
    y_test = pd.read_csv("task_vseros_0/ytest_D.csv", index_col="id", sep=",").to_numpy(dtype=np.float32)

    x_train = torch.from_numpy(x_train).to("cuda").float()
    x_test = torch.from_numpy(x_test).to("cuda").float()
    y_train = torch.from_numpy(y_train).to("cuda").float()
    y_test = torch.from_numpy(y_test).to("cuda").float()

    

    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.BatchNorm1d(64),
        nn.Mish(),
        nn.Dropout(p = 0.2),
        nn.Linear(64, 64),
        nn.BatchNorm1d(64),
        nn.Mish(),
        nn.Dropout(p = 0.2),
        nn.Linear(64, 64),
        nn.BatchNorm1d(64),
        nn.Mish(),
        nn.Dropout(p = 0.2),
        nn.Linear(64, 25),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(x_train.size(0))
        epoch_loss = 0.0
        for i in range(0, x_train.size(0), batch_size):
            optimizer.zero_grad()
            idxs = perm[i:i+batch_size]
            batch_x = x_train[idxs]
            batch_y = y_train[idxs]

            forw_pass = model(batch_x)
            loss = criterion(forw_pass, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()


    model.eval()
    with torch.no_grad():
        pred = torch.argmax(model(x_test), dim = 1)
        score = metric(pred, y_test.view(-1).long())

    return score

print(100 + ((float(task_d2(150, 64, 0.001)) - 0.27)/(0.8 - 0.27) * 100) + ((21.0 - task_c2())/(21.0 - 13.8) * 100) + task_b2()*100)
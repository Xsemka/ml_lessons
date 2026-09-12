import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import f1_score, root_mean_squared_error

import optuna


class getModel:
    def __init__(self, n_trials):
        self.n_trials = n_trials
        self.sampler = optuna.samplers.TPESampler(multivariate=True, warn_independent_sampling=False)
        self.pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=100, reduction_factor=3)

    def linear_regression(self, x_train, x_test, y_train, y_test, direction="minimize"):
        def optim(trial):
            model = LinearRegression(n_jobs=-1)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = root_mean_squared_error(y_test, pred)
            return score
        
        study = optuna.create_study(sampler=self.sampler, pruner=self.pruner, direction=direction)
        study.optimize(optim, n_trials=self.n_trials)
        params = study.best_params
        model = SVR(C=10**params["c_defree"], kernel=params["kernel"], degree=params["degree"])
        model.fit(x_train, y_train)
        return model

    def svm_classifier(self, x_train, x_test, y_train, y_test, direction="maximize"):
        def optim(trial):
            c_degree = trial.suggest_int("c_defree", -2, 2)
            kernel = trial.suggest_categorical("kernel", ["rbf", "sigmoid", "poly"])
            degree = trial.suggest_int("degree", 1, 5)
            model = SVC(C=10**c_degree, kernel=kernel, degree=degree, class_weight="balanced")
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = f1_score(y_test, pred, average="weighted")
            return score
        
        study = optuna.create_study(sampler=self.sampler, pruner=self.pruner, direction=direction)
        study.optimize(optim, n_trials=self.n_trials)
        params = study.best_params
        model = SVC(C=10**params["c_defree"], kernel=params["kernel"], degree=params["degree"])
        model.fit(x_train, y_train)
        return model

    def svm_regressor(self, x_train, x_test, y_train, y_test, direction="minimize"):
        def optim(trial):
            c_degree = trial.suggest_int("c_defree", -2, 2)
            kernel = trial.suggest_categorical("kernel", ["rbf", "sigmoid", "poly"])
            degree = trial.suggest_int("degree", 1, 5)
            model = SVR(C=10**c_degree, kernel=kernel, degree=degree)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = root_mean_squared_error(y_test, pred)
            return score
        
        study = optuna.create_study(sampler=self.sampler, pruner=self.pruner, direction=direction)
        study.optimize(optim, n_trials=self.n_trials)
        params = study.best_params
        model = SVR(C=10**params["c_defree"], kernel=params["kernel"], degree=params["degree"])
        model.fit(x_train, y_train)
        return model

    def KNN_classifier(self, x_train, x_test, y_train, y_test, direction="maximize"):
        def optim(trial):
            n_neighbors = trial.suggest_int("n_neighbors", 3, 20)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = f1_score(y_test, pred, average="weighted")
            return score
        study = optuna.create_study(sampler=self.sampler, pruner=self.pruner, direction=direction)
        study.optimize(optim, n_trials=self.n_trials)
        params = study.best_params
        model = KNeighborsClassifier(n_neighbors=params["n_neighbors"])
        model.fit(x_train, y_train)
        return model

    def KNN_regressor(self, x_train, x_test, y_train, y_test, direction="minimize"):
        def optim(trial):
            n_neighbors = trial.suggest_int("n_neighbors", 3, 20)
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = root_mean_squared_error(y_test, pred)
            return score
        study = optuna.create_study(sampler=self.sampler, pruner=self.pruner, direction=direction, n_jobs=-1)
        study.optimize(optim, n_trials=self.n_trials)
        params = study.best_params
        model = KNeighborsRegressor(n_neighbors=params["n_neighbors"], n_jobs=-1)
        model.fit(x_train, y_train)
        return model

    def naive_bayes(self, x_train, x_test, y_train, y_test, direction="minimize"):
        model = GaussianNB()
        model.fit(x_train, y_train)
        return model

    def tree_classifier(self, x_train, x_test, y_train, y_test, direction="maximize"):
        def optim(trial):
            max_depth = trial.suggest_int("max_depth", 3, 30)
            criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
            splitter = trial.suggest_categorical("splitter", ["best", "random"])
            ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.1)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 30)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 30)
            model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, splitter=splitter, ccp_alpha=ccp_alpha, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf ,class_weight="balanced")
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = f1_score(y_test, pred, average="weighted")
            return score
        study = optuna.create_study(sampler=self.sampler, pruner=self.pruner, direction=direction)
        study.optimize(optim, n_trials=self.n_trials)
        params = study.best_params
        model = DecisionTreeClassifier(max_depth=params["max_depth"], criterion=params["criterion"], splitter=params["splitter"], ccp_alpha=params["ccp_alpha"], min_samples_split=params["min_samples_split"], min_samples_leaf=params["min_samples_leaf"] ,class_weight="balanced")
        model.fit(x_train, y_train)
        return model

    def tree_regressor(self, x_train, x_test, y_train, y_test, direction="minimize"):
        def optim(trial):
            max_depth = trial.suggest_int("max_depth", 3, 30)
            criterion = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])
            splitter = trial.suggest_categorical("splitter", ["best", "random"])
            ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.1)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 30)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 30)
            model = DecisionTreeRegressor(max_depth=max_depth, criterion=criterion, splitter=splitter, ccp_alpha=ccp_alpha, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = root_mean_squared_error(y_test, pred)
            return score
        study = optuna.create_study(sampler=self.sampler, pruner=self.pruner, direction=direction)
        study.optimize(optim, n_trials=self.n_trials)
        params = study.best_params
        model = DecisionTreeRegressor(max_depth=params["max_depth"], criterion=params["criterion"], splitter=params["splitter"], ccp_alpha=params["ccp_alpha"], min_samples_split=params["min_samples_split"], min_samples_leaf=params["min_samples_leaf"] ,class_weight="balanced")
        model.fit(x_train, y_train)
        return model

    
    def random_forest_classifier(self, x_train, x_test, y_train, y_test, direction="maximize"):
        def optim(trial):
            max_depth = trial.suggest_int("max_depth", 3, 30)
            criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
            ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.1)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 30)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 30)
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            model = RandomForestClassifier(max_depth=max_depth, criterion=criterion, ccp_alpha=ccp_alpha, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, class_weight="balanced", n_jobs=-1)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = f1_score(y_test, pred, average="weighted")
            return score
        study = optuna.create_study(sampler=self.sampler, pruner=self.pruner, direction=direction)
        study.optimize(optim, n_trials=self.n_trials)
        params = study.best_params
        model = RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"], criterion=params["criterion"], ccp_alpha=params["ccp_alpha"], min_samples_split=params["min_samples_split"], min_samples_leaf=params["min_samples_leaf"] ,class_weight="balanced", n_jobs=-1)
        model.fit(x_train, y_train)
        return model
    
    def random_forest_regressor(self, x_train, x_test, y_train, y_test, direction="minimize"):
        def optim(trial):
            max_depth = trial.suggest_int("max_depth", 3, 30)
            criterion = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])
            ccp_alpha = trial.suggest_float('ccp_alpha', 0.0, 0.1)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 30)
            min_samples_leaf = trial.suggest_int("min_samples_leaf", 2, 30)
            n_estimators = trial.suggest_int("n_estimators", 50, 500)
            model = RandomForestRegressor(max_depth=max_depth, criterion=criterion, ccp_alpha=ccp_alpha, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, n_estimators=n_estimators, n_jobs=-1)
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            score = root_mean_squared_error(y_test, pred)
            return score
        study = optuna.create_study(sampler=self.sampler, pruner=self.pruner, direction=direction)
        study.optimize(optim, n_trials=self.n_trials)
        params = study.best_params
        model = RandomForestRegressor(n_estimators=params["n_estimators"], max_depth=params["max_depth"], criterion=params["criterion"], ccp_alpha=params["ccp_alpha"], min_samples_split=params["min_samples_split"], min_samples_leaf=params["min_samples_leaf"], n_jobs=-1)
        model.fit(x_train, y_train)
        return model
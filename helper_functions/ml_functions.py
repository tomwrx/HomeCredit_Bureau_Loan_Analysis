import numpy as np
import pandas as pd
from collections import defaultdict
import time
import matplotlib.pyplot as plt
import optuna
import shap
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import BayesianRidge, LogisticRegression, SGDClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from imblearn.over_sampling import SMOTE, SMOTENC, ADASYN
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import make_pipeline as imb_make_pipeline
from sklearn.cluster import KMeans
from hdbscan import HDBSCAN
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    StratifiedKFold,
)
from lightgbm import LGBMRegressor, LGBMClassifier, early_stopping
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_curve,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    mean_squared_error,
    r2_score,
    brier_score_loss,
    log_loss,
)
from helper_functions.general_functions import get_state_region

from typing import Tuple, List, Dict, Callable


def adj_threshold_to_labels(model_probs: np.ndarray, threshold: float) -> np.ndarray:
    """From input of positive class probabilites applies
    threshold to positive probabilities to create labels"""

    return (model_probs >= threshold).astype("int")


def reversed_feature(feature: np.ndarray) -> np.ndarray:
    "From input creates a reversed feature"
    return np.power((feature.astype(float)), -1)


def log_feature(feature: np.ndarray) -> np.ndarray:
    "From input creates a log feature"
    return np.log(((feature.astype(float)) + 1e-4))


def squared_feature(feature: np.ndarray) -> np.ndarray:
    "From input creates a squared feature"
    return np.power((feature.astype(float)), 2)


def cubic_feature(feature: np.ndarray) -> np.ndarray:
    "From input creates a cubic feature"
    return np.power((feature.astype(float)), 3)


def qbinned_feature(feature: np.ndarray) -> np.ndarray:
    "From input creates a binned feature"
    quartile_list = [0, 0.25, 0.5, 0.75, 1.0]
    bins = np.quantile(feature, quartile_list)
    return np.digitize(feature, bins)


def numeric_imputation_search(
    X: pd.DataFrame,
    y: pd.DataFrame,
    eval_model: BaseEstimator,
    random_state: int,
    scoring: str,
) -> plt.figure:
    "Searches for best numeric imputation method and plots results"

    cv = 5

    baseline_scores = pd.DataFrame()
    X_baseline = X.copy()
    y_baseline = y.copy()
    X_baseline = X_baseline.dropna()
    y_baseline = y_baseline.reindex(X_baseline.index)
    baseline_scores["No imputation, instances dropped"] = cross_val_score(
        eval_model, X_baseline, y_baseline, scoring=scoring, cv=cv,
    )

    si_scores = pd.DataFrame()
    for strategy in ["mean", "median"]:
        pipe = make_pipeline(SimpleImputer(strategy=strategy), eval_model)
        si_scores[f"SimpleImputer (strategy = {strategy})"] = cross_val_score(
            pipe, X, y, scoring=scoring, cv=cv
        )

    ii_scores = pd.DataFrame()
    for estimator in [BayesianRidge(), ExtraTreesRegressor(random_state=random_state)]:
        pipe = make_pipeline(
            IterativeImputer(estimator=estimator, random_state=random_state), eval_model
        )
        ii_scores[estimator.__class__.__name__] = cross_val_score(
            pipe, X, y, scoring=scoring, cv=cv
        )

    knn_scores = pd.DataFrame()
    n_neighbors = [2, 3, 5, 7, 9]
    for k in n_neighbors:
        pipe = make_pipeline(KNNImputer(n_neighbors=k), eval_model)
        knn_scores[f"KNN(k = {k})"] = cross_val_score(
            pipe, X, y, scoring=scoring, cv=cv
        )

    final_scores = pd.concat(
        [baseline_scores, si_scores, ii_scores, knn_scores],
        axis=1,
        keys=["baseline_score", "simple_imputer", "iterative_imputer", "knn_imputer"],
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    means, errors = final_scores.mean().sort_values(ascending=False), final_scores.std()
    means.plot.barh(xerr=errors, ax=ax)

    ax.set_title(
        f"Different Imputation Methods results with {eval_model.__class__.__name__} as eval model"
    )
    ax.set_xlabel(f"{scoring.capitalize()} score")
    ax.set_yticks(np.arange(means.shape[0]))

    plt.show()
    

def model_based_cat_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Takes a dataframe and imputes missing values of categorical features
    based on SGD model classification results"""

    df = df.copy()
    eval_model = SGDClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
    for col in df.select_dtypes(exclude="number"):
        if df[col].isnull().sum() > 0:
            train = df[df[col].isna() == False]
            test = df[df[col].isna() == True]
            train_data = train.drop(col, axis=1)
            train_target = train[col].copy()
            test_data = test.drop(col, axis=1)
            cat_cols = []
            for cat_col in train_data.columns:
                if (
                    train_data[cat_col].nunique() < 3
                    or train_data[cat_col].dtype == "object"
                ) and (
                    cat_col.split("_")[-1]
                    not in ["min", "max", "mean", "count", "unique", "norm",]
                ):
                    cat_cols.append(cat_col)
            num_cols = [
                num_col
                for num_col in train_data.select_dtypes("number")
                if num_col not in cat_cols
            ]
            num_pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            cat_pipe = Pipeline(
                [("categorical_enc", OneHotEncoder(handle_unknown="ignore")),]
            )
            full_pipeline = Pipeline(
                [
                    (
                        "union",
                        ColumnTransformer(
                            [
                                ("numeric", num_pipe, num_cols),
                                ("categorical", cat_pipe, cat_cols),
                            ]
                        ),
                    ),
                ],
            )
            train_data_tr = full_pipeline.fit_transform(train_data)
            test_data_tr = full_pipeline.transform(test_data)
            eval_model.fit(train_data_tr, train_target)
            imputed_feature = pd.Series(eval_model.predict(test_data_tr), test.index)
            df[col] = df[col].fillna(imputed_feature)

    return df


def categorical_imputation_search(
    X: pd.DataFrame, y: pd.DataFrame, eval_model: BaseEstimator, scoring: str,
) -> plt.figure:
    "Searches for best categorical imputation method and plots results"

    cv = 5

    baseline_scores = pd.DataFrame()
    X_baseline = X.copy()
    y_baseline = y.copy()
    X_baseline = X_baseline.dropna()
    y_baseline = y_baseline.reindex(X_baseline.index)
    cat_cols = []
    for cat_col in X_baseline.columns:
        if (
            X_baseline[cat_col].nunique() < 3 or X_baseline[cat_col].dtype == "object"
        ) and (
            cat_col.split("_")[-1]
            not in ["min", "max", "mean", "count", "unique", "norm",]
        ):
            cat_cols.append(cat_col)
    num_cols = [
        num_col
        for num_col in X_baseline.select_dtypes("number")
        if num_col not in cat_cols
    ]
    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler()),]
    )
    cat_pipe = Pipeline(
        [
            ("cat_imputer", SimpleImputer(strategy="most_frequent")),
            ("categorical_enc", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    full_pipeline = Pipeline(
        [
            (
                "union",
                ColumnTransformer(
                    [
                        ("numeric", num_pipe, num_cols),
                        ("categorical", cat_pipe, cat_cols),
                    ]
                ),
            ),
        ],
    )
    X_baseline_tr = full_pipeline.fit_transform(X_baseline)
    baseline_scores["No imputation, instances dropped"] = cross_val_score(
        eval_model, X_baseline_tr, y_baseline, scoring=scoring, cv=cv,
    )

    si_scores = pd.DataFrame()
    X_imputer_tr = full_pipeline.fit_transform(X)
    si_scores[f"SimpleImputer (strategy = most_frequent)"] = cross_val_score(
        eval_model, X_imputer_tr, y, scoring=scoring, cv=cv
    )

    model_based_scores = pd.DataFrame()
    model_based_train_data = model_based_cat_imputation(X)
    model_based_train_data_tr = full_pipeline.fit_transform(model_based_train_data)
    model_based_scores[eval_model.__class__.__name__] = cross_val_score(
        eval_model, model_based_train_data_tr, y, scoring=scoring, cv=cv
    )

    final_scores = pd.concat(
        [baseline_scores, si_scores, model_based_scores],
        axis=1,
        keys=["baseline_score", "simple_imputer", "SGD_model_based_imputation"],
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    means, errors = final_scores.mean().sort_values(ascending=False), final_scores.std()
    means.plot.barh(xerr=errors, ax=ax)

    ax.set_title(
        f"Different Imputation Methods results with {eval_model.__class__.__name__} as eval model"
    )
    ax.set_xlabel(f"{scoring.capitalize()} score")
    ax.set_yticks(np.arange(means.shape[0]))

    plt.show()


def baseline_clfmodels_eval_cv(
    clf_list: List[BaseEstimator],
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler: Callable,
    kf: StratifiedKFold,
    num_columns_idx: List[int],
    multi_class: bool = None,
) -> pd.DataFrame:
    """Takes a list of models, training set, 
    training labels, numerical columns list and returns different classification scores in
    a DataFrame"""

    scores = defaultdict(list)

    for clf in clf_list:
        start = time.time()
        scores["Classifier"].append(clf.__class__.__name__)
        pipe = make_pipeline(
            ColumnTransformer(
                [("numeric", scaler, num_columns_idx)], remainder="passthrough",
            ),
            clf,
        )

        for metric in [
            "balanced_accuracy",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f1_macro",
            "average_precision",
            "roc_auc",
        ]:
            if multi_class and metric in [
                "precision",
                "recall",
                "f1",
                "average_precision",
                "roc_auc",
            ]:
                continue
            elif not multi_class and metric == 'f1_macro':
                continue

            cross_val_metric = cross_val_score(
                pipe, X_train, y_train, cv=kf, scoring=metric
            )
            score_name = " ".join(metric.split("_")).capitalize()
            scores[score_name].append(np.mean(cross_val_metric))
        end = time.time()
        scores["Total time in sec"].append((end - start))

    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df = score_df.round(3)

    return score_df


def baseline_regmodels_eval_cv(
    reg_list: List[BaseEstimator],
    X_train: np.ndarray,
    y_train: np.ndarray,
    scaler: Callable,
    kf: KFold,
    num_columns_idx: List[int],
) -> pd.DataFrame:
    """Takes a list of models, training set, 
    training labels, numerical columns list and returns different regression scores in
    a DataFrame"""

    scores = defaultdict(list)

    for reg in reg_list:
        start = time.time()
        scores["Regressor"].append(reg.__class__.__name__)
        pipe = make_pipeline(
            ColumnTransformer(
                [("numeric", scaler, num_columns_idx)], remainder="passthrough",
            ),
            reg,
        )

        for metric in [
            "neg_mean_squared_error",
            "neg_root_mean_squared_error",
        ]:

            cross_val_metric = cross_val_score(
                pipe, X_train, y_train, cv=kf, scoring=metric
            )
            score_name = " ".join(metric.split("_")[1:]).capitalize()
            scores[score_name].append(-np.mean(cross_val_metric))
        end = time.time()
        scores["Total time in sec"].append((end - start))

    score_df = pd.DataFrame(scores).set_index("Regressor")
    score_df = score_df.round(3)

    return score_df


class KmeansClustering(BaseEstimator, TransformerMixin):
    """Performs unsupervised clustering of training data numerical features
    and returns a new array with numerical features and clusters labels"""

    def __init__(self, n_clusters: int, scaler: BaseEstimator):
        self.n_clusters = n_clusters
        self.col_label = ["Kmeans_clusters"]
        self.kmeans = None
        self.scaler = scaler

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.kmeans:
            X_scaled = self.scaler.transform(X)
            clusters = self.kmeans.predict(X_scaled)
            return np.c_[X, clusters]
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=50, random_state=42)
        clusters = self.kmeans.fit_predict(X_scaled)
        return np.c_[X, clusters]


class KmeansClusterDistance(BaseEstimator, TransformerMixin):
    """Performs training data numerical features distance calculation
    to the cluster centroids and returns a new array with numerical features
    and distance to every centroid"""

    def __init__(self, n_clusters: int, scaler: BaseEstimator):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.col_labels = None
        self.centroids = None
        self.scaler = scaler

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is not None:
            X_scaled = self.scaler.transform(X)
            test_centroids = self.kmeans.transform(X_scaled)
            return np.c_[X, test_centroids]
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=50, random_state=42)
        self.centroids = self.kmeans.fit_transform(X_scaled)
        self.col_labels = [f"Centroid_{i}" for i in range(self.centroids.shape[1])]
        return np.c_[X, self.centroids]


class HDBSCANClustering(BaseEstimator, TransformerMixin):
    """Performs unsupervised clustering of training data numerical features
    and returns a new array with numerical features and clusters labels"""

    def __init__(
        self,
        min_cluster_size: int,
        min_samples: int,
        cluster_selection_epsilon: float,
        scaler: BaseEstimator,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.col_label = ["HDBSCAN_clusters"]
        self.hdbscan = None
        self.scaler = scaler

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.hdbscan:
            X_scaled = self.scaler.transform(X)
            clusters = self.hdbscan.predict(X_scaled)
            return np.c_[X, clusters]
        X_scaled = self.scaler.fit_transform(X)
        self.hdbscan = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
        )
        clusters = self.hdbscan.fit_predict(X_scaled)
        return np.c_[X, clusters]


class NumericFeatureTransformation(BaseEstimator, TransformerMixin):
    """Performs training data numerical features transformations with the 
    passed list of functions and leaves only ones with stronger correlation
    then original feature. Returns a new array with original numerical features
    and new features"""

    def __init__(
        self,
        num_col_labels: List[str],
        num_col_idx: List[int],
        func_list: List[Callable],
        y: np.ndarray,
    ):
        self.num_col_labels = num_col_labels
        self.num_col_idx = num_col_idx
        self.col_labels = []
        self.func_list = func_list
        self.y = y
        self.test_check = False

    def check_if_better(self, feature: np.ndarray, new_feature: np.ndarray):

        if new_feature.shape[0] == self.y.shape[0]:
            return (
                True
                if abs(round(np.corrcoef(feature, self.y)[0, 1], 3))
                < abs(round(np.corrcoef(new_feature, self.y)[0, 1], 3))
                else False
            )
        else:
            self.test_check = True

    def fit(self, X: np.ndarray, y=None):
        return self

    def transform(self, X: np.ndarray):
        for label, idx in zip(self.num_col_labels, self.num_col_idx):
            for func in self.func_list:
                new_feature = func(X[:, idx])
                new_feature_label = f"{label}_{func.__name__.split('_')[0]}"
                if (
                    self.check_if_better(X[:, idx], new_feature)
                    and new_feature_label not in self.col_labels
                    and not self.test_check
                ):
                    self.col_labels.append(new_feature_label)
                    X = np.c_[X, new_feature]
                else:
                    if self.test_check and new_feature_label in self.col_labels:
                        X = np.c_[X, new_feature]
        return X


def xgboost_objective(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    num_columns_idx: List[int],
    objective: str,
    metric,
) -> float:
    "XGBoost objective function to tune hyper parameters."

    grid_params = {
        "objective": objective,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
        "use_label_encoder": False,
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300, step=10),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "gama": trial.suggest_int("gama", 0, 10),
        "lambda": trial.suggest_int("lambda", 0, 100, step=5),
        "alpha": trial.suggest_int("alpha", 0, 100, step=5),
    }

    cv_scores = np.empty(cv.n_splits)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        numeric_scaler = ColumnTransformer(
            [("numeric", StandardScaler(), num_columns_idx)], remainder="passthrough"
        )

        X_train_tr = numeric_scaler.fit_transform(X_train)
        X_val_tr = numeric_scaler.transform(X_val)

        model = XGBClassifier(**grid_params)
        model.fit(
            X_train_tr,
            y_train,
            eval_set=[(X_val_tr, y_val)],
            eval_metric="logloss",
            early_stopping_rounds=100,
            callbacks=[XGBoostPruningCallback(trial, "validation_0-logloss"),],
            verbose=False,
        )
        preds = model.predict(X_val_tr)
        cv_scores[idx] = metric(y_val, preds)

    return -np.mean(cv_scores)


def xgboost_objective_reg(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: KFold,
    num_columns_idx: List[int],
    objective: str,
    metric,
) -> float:
    "XGBoost Regressor objective function to tune hyper parameters."

    grid_params = {
        "objective": objective,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300, step=10),
        "min_child_weight": trial.suggest_int("min_child_weight", 0, 10),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "gama": trial.suggest_int("gama", 0, 10),
        "lambda": trial.suggest_int("lambda", 0, 100, step=5),
        "alpha": trial.suggest_int("alpha", 0, 100, step=5),
    }

    cv_scores = np.empty(cv.n_splits)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        numeric_scaler = ColumnTransformer(
            [("numeric", StandardScaler(), num_columns_idx)], remainder="passthrough"
        )

        X_train_tr = numeric_scaler.fit_transform(X_train)
        X_val_tr = numeric_scaler.transform(X_val)

        model = XGBRegressor(**grid_params)
        model.fit(
            X_train_tr,
            y_train,
            eval_set=[(X_val_tr, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=100,
            callbacks=[XGBoostPruningCallback(trial, "validation_0-rmse"),],
            verbose=False,
        )
        preds = model.predict(X_val_tr)
        cv_scores[idx] = metric(y_val, preds, squared=False)

    return np.mean(cv_scores)


def light_gbm_objective(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Callable,
    cv: StratifiedKFold,
    num_columns_idx: List[int],
    objective: str,
    metric,
) -> float:
    "LightGBM objective function to tune hyper parameters."

    grid_params = {
        "objective": objective,
        "metric": "average_precision",
        # "metric": "multi_logloss",
        # "num_class": 35,
        "random_state": 42,
        "verbosity": -1,
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100, step=1),
        #"scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 300, step=10),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=1),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=1),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 10),
    }

    call_back_metric = "average_precision"

    cv_scores = np.empty(cv.n_splits)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        numeric_scaler = ColumnTransformer(
            [("numeric", scaler, num_columns_idx)], remainder="passthrough"
        )
        X_train_tr = numeric_scaler.fit_transform(X_train)
        X_val_tr = numeric_scaler.transform(X_val)

        model = LGBMClassifier(**grid_params)
        model.fit(
            X_train_tr,
            y_train,
            eval_set=[(X_val_tr, y_val)],
            callbacks=[
                early_stopping(100),
                LightGBMPruningCallback(trial, call_back_metric),
            ],
        )
        preds = model.predict(X_val_tr)
        cv_scores[idx] = metric(y_val, preds)

    return np.mean(cv_scores)


def light_gbm_objective_reg(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Callable,
    cv: KFold,
    num_columns_idx: List[int],
    objective: str,
    metric,
) -> float:
    "LightGBM Regressor objective function to tune hyper parameters."

    grid_params = {
        "objective": objective,
        "metric": "rmse",
        "random_state": 42,
        "verbosity": -1,
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100, step=5),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
    }

    cv_scores = np.empty(cv.n_splits)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        numeric_scaler = ColumnTransformer(
            [("numeric", scaler, num_columns_idx)], remainder="passthrough"
        )

        X_train_tr = numeric_scaler.fit_transform(X_train)
        X_val_tr = numeric_scaler.transform(X_val)

        model = LGBMRegressor(**grid_params)
        model.fit(
            X_train_tr,
            y_train,
            eval_set=[(X_val_tr, y_val)],
            callbacks=[early_stopping(100), LightGBMPruningCallback(trial, "rmse"),],
        )
        preds = model.predict(X_val_tr)
        cv_scores[idx] = metric(y_val, preds, squared=False)

    return np.mean(cv_scores)


def cat_boost_objective(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    num_columns_idx: List[int],
    objective: str,
    metric,
) -> float:
    "Cat Boost objective function to tune hyper parameters."

    grid_params = {
        "objective": objective,
        "eval_metric": objective,
        "grow_policy": "Lossguide",
        "random_state": 42,
        "verbose": 0,
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300, step=20),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100, step=5),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 0, 100, step=5),
    }

    cv_scores = np.empty(cv.n_splits)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        numeric_scaler = ColumnTransformer(
            [("numeric", StandardScaler(), num_columns_idx)], remainder="passthrough"
        )

        X_train_tr = numeric_scaler.fit_transform(X_train)
        X_val_tr = numeric_scaler.transform(X_val)

        model = CatBoostClassifier(**grid_params)
        model.fit(
            X_train_tr,
            y_train,
            eval_set=[(X_val_tr, y_val)],
            early_stopping_rounds=100,
            verbose=False,
        )
        preds = model.predict(X_val_tr)
        cv_scores[idx] = metric(y_val, preds)

    return -np.mean(cv_scores)


def cat_boost_objective_reg(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    cv: KFold,
    num_columns_idx: List[int],
    objective: str,
    metric,
) -> float:
    "Cat Boost Regressor objective function to tune hyper parameters."

    grid_params = {
        "objective": objective,
        "eval_metric": objective,
        "grow_policy": "Lossguide",
        "random_state": 42,
        "verbose": 0,
        "n_estimators": trial.suggest_categorical("n_estimators", [10000]),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 300, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100, step=5),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 0, 100, step=5),
    }

    cv_scores = np.empty(cv.n_splits)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        numeric_scaler = ColumnTransformer(
            [("numeric", StandardScaler(), num_columns_idx)], remainder="passthrough"
        )

        X_train_tr = numeric_scaler.fit_transform(X_train)
        X_val_tr = numeric_scaler.transform(X_val)

        model = CatBoostRegressor(**grid_params)
        model.fit(
            X_train_tr,
            y_train,
            eval_set=[(X_val_tr, y_val)],
            early_stopping_rounds=100,
            verbose=False,
        )
        preds = model.predict(X_val_tr)
        cv_scores[idx] = metric(y_val, preds, squared=False)

    return np.mean(cv_scores)


def sgd_clf_objective(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Callable,
    cv: StratifiedKFold,
    num_columns_idx: List[int],
    objective: str,
    metric,
) -> float:
    "SGD Classifier objective function to tune hyper parameters."

    grid_params = {
        "random_state": 42,
        "verbose": 0,
        "early_stopping": True,
        "validation_fraction": 0.25,
        "n_iter_no_change": 10,
        "n_jobs": -1,
        "eta0": 0.1,
        "loss": trial.suggest_categorical("loss", ["hinge", "log"]),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", ["optimal", "adaptive"]
        ),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }

    cv_scores = np.empty(cv.n_splits)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        numeric_scaler = ColumnTransformer(
            [("numeric", scaler, num_columns_idx)], remainder="passthrough"
        )
        X_train_tr = numeric_scaler.fit_transform(X_train)
        X_val_tr = numeric_scaler.transform(X_val)

        model = SGDClassifier(**grid_params)
        model.fit(X_train_tr, y_train)
        preds = model.predict(X_val_tr)
        cv_scores[idx] = metric(y_val, preds)

    return -np.mean(cv_scores)


def logr_clf_objective(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Callable,
    cv: StratifiedKFold,
    num_columns_idx: List[int],
    objective: str,
    metric,
) -> float:
    "Logistic Regression Classifier objective function to tune hyper parameters."

    grid_params = {
        "random_state": 42,
        "verbose": 0,
        "n_jobs": -1,
        "solver": "saga",
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2", 'none']),
        "C": trial.suggest_float("C", 0, 10),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
    }

    cv_scores = np.empty(cv.n_splits)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        numeric_scaler = ColumnTransformer(
            [("numeric", scaler, num_columns_idx)], remainder="passthrough"
        )
        X_train_tr = numeric_scaler.fit_transform(X_train)
        X_val_tr = numeric_scaler.transform(X_val)

        model = LogisticRegression(**grid_params)
        model.fit(X_train_tr, y_train)
        preds = model.predict(X_val_tr)
        cv_scores[idx] = metric(y_val, preds, average="macro")

    return -np.mean(cv_scores)


def dt_clf_objective(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Callable,
    cv: StratifiedKFold,
    num_columns_idx: List[int],
    objective: str,
    metric,
) -> float:
    "DT Classifier objective function to tune hyper parameters."

    grid_params = {
        "random_state": 42,
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
        "splitter": trial.suggest_categorical("splitter", ["best", "random"]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 300),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50),
        "min_weight_fraction_leaf": trial.suggest_float(
            "min_weight_fraction_leaf", 0, 0.5
        ),
        "max_features": trial.suggest_categorical(
            "max_features", ["auto", "sqrt", "log2"]
        ),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
    }

    cv_scores = np.empty(cv.n_splits)

    for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        numeric_scaler = ColumnTransformer(
            [("numeric", scaler, num_columns_idx)], remainder="passthrough"
        )
        X_train_tr = numeric_scaler.fit_transform(X_train)
        X_val_tr = numeric_scaler.transform(X_val)

        model = DecisionTreeClassifier(**grid_params)
        model.fit(X_train_tr, y_train)
        preds = model.predict(X_val_tr)
        cv_scores[idx] = metric(y_val, preds, average="macro")

    return -np.mean(cv_scores)


def tune_model(
    objective_func: Callable,
    direction: str,
    n_trials: int,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Callable,
    cv: StratifiedKFold,
    num_columns_idx: List[int],
    objective: str,
    metric,
) -> Tuple[float, dict]:
    "Funtion to tune a model. Returns best value and tuned model hyper-parameters"

    study = optuna.create_study(direction=direction)
    func = lambda trial: objective_func(
        trial, X, y, scaler, cv, num_columns_idx, objective, metric
    )
    study.optimize(func, n_trials=n_trials)

    return round(study.best_value, 3), study.best_params


def pipeline_objective(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    eval_model: BaseEstimator,
    num_col_idx: List[int],
    cat_col_idx: List[int],
    cv: StratifiedKFold,
    metric: str,
) -> float:
    "Pre-processing pipeline optimization objective function"

    smote_num = trial.suggest_categorical(
        "num_smote_method", ["smote", "adasyn", "smoteenn", "smotetomek"]
    )
    if smote_num == "smote":
        smote = SMOTE(random_state=42)
    elif smote_num == "adasyn":
        smote = ADASYN(random_state=42)
    elif smote_num == "smoteenn":
        smote = SMOTEENN(random_state=42)
    else:
        smote = SMOTETomek(random_state=42)

    scalers = trial.suggest_categorical("scalers", ["minmax", "standard", "robust"])
    if scalers == "minmax":
        scaler = MinMaxScaler()
    elif scalers == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    cat_encoders = trial.suggest_categorical("cat_encoders", ["ordinal", "one_hot"])
    if cat_encoders == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="ignore")
    else:
        encoder = OneHotEncoder(handle_unknown="ignore")

    clustering = trial.suggest_categorical("clustering", ["KMeans", "HDB", None])
    if clustering == "KMeans":
        num_of_clusters = trial.suggest_int("num_of_clusters", 2, 15)
        clust_algo = KmeansClustering(n_clusters=num_of_clusters, scaler=scaler)
    elif clustering == "HDB":
        min_cluster_size = trial.suggest_int("min_cluster_size", 5, 100, step=5)
        min_samples = trial.suggest_int("min_samples", 10, 100, step=10)
        cluster_selection_epsilon = trial.suggest_float("epsilon", 0.1, 0.5)
        clust_algo = HDBSCANClustering(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            scaler=scaler,
        )
    else:
        clust_algo = "passthrough"

    centroids = trial.suggest_categorical("kmeans_centroids", ["KMeansCentroids", None])
    if centroids == "KMeansCentroids":
        num_of_clusters = trial.suggest_int("num_of_centroids", 2, 15)
        centroids_algo = KmeansClusterDistance(
            n_clusters=num_of_clusters, scaler=scaler
        )
    else:
        centroids_algo = "passthrough"

    class_w = trial.suggest_categorical("class_weights", ["balanced", None])
    estimator = eval_model(random_state=42, class_weight=class_w, n_jobs=-1)

    num_cat_transf = ColumnTransformer(
        [
            ("numeric", scaler, num_col_idx),
            ("cat_enc", encoder, [cat_col_idx[0]]),
            ("one_hot", OneHotEncoder(handle_unknown="ignore"), [cat_col_idx[1]]),
        ],
    )

    pipeline = imb_make_pipeline(
        num_cat_transf, smote, clust_algo, centroids_algo, estimator
    )

    score = cross_val_score(pipeline, X, y, cv=cv, scoring=metric)
    return np.mean(score)


def pipeline_objective_2(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    eval_model: BaseEstimator,
    num_col_idx: List[int],
    cat_col_idx: List[int],
    cv: StratifiedKFold,
    metric: str,
) -> float:
    "Pre-processing pipeline optimization objective function"

    scalers = trial.suggest_categorical("scalers", ["minmax", "standard", "robust"])
    if scalers == "minmax":
        scaler = MinMaxScaler()
    elif scalers == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    cat_encoders = trial.suggest_categorical("cat_encoders", ["ordinal", "one_hot"])
    if cat_encoders == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="ignore")
    else:
        encoder = OneHotEncoder(handle_unknown="ignore")

    clustering = trial.suggest_categorical("clustering", ["KMeans", None])
    if clustering == "KMeans":
        num_of_clusters = trial.suggest_int("num_of_clusters", 2, 15)
        clust_algo = KmeansClustering(n_clusters=num_of_clusters, scaler=scaler)
    else:
        clust_algo = "passthrough"

    centroids = trial.suggest_categorical("kmeans_centroids", ["KMeansCentroids", None])
    if centroids == "KMeansCentroids":
        num_of_clusters = trial.suggest_int("num_of_centroids", 2, 15)
        centroids_algo = KmeansClusterDistance(
            n_clusters=num_of_clusters, scaler=scaler
        )
    else:
        centroids_algo = "passthrough"

    class_w = trial.suggest_categorical("class_weights", ["balanced", None])
    estimator = eval_model(class_weight=class_w, random_state=42, n_jobs=-1)

    num_cat_transf = ColumnTransformer(
        [("numeric", scaler, num_col_idx), ("cat_enc", encoder, cat_col_idx),]
    )

    pipeline = make_pipeline(num_cat_transf, clust_algo, centroids_algo, estimator)

    score = cross_val_score(pipeline, X, y, cv=cv, scoring=metric)
    return np.mean(score)


def pipeline_objective_3(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    eval_model: BaseEstimator,
    num_col_idx: List[int],
    cat_col_idx: List[int],
    cv: StratifiedKFold,
    metric: str,
) -> float:
    "Pre-processing pipeline optimization objective function"

    scalers = trial.suggest_categorical("scalers", ["minmax", "standard", "robust"])
    if scalers == "minmax":
        scaler = MinMaxScaler()
    elif scalers == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    cat_encoders = trial.suggest_categorical("cat_encoders", ["ordinal", "one_hot"])
    if cat_encoders == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="ignore")
    else:
        encoder = OneHotEncoder(handle_unknown="ignore")

    clustering = trial.suggest_categorical("clustering", ["KMeans", None])
    if clustering == "KMeans":
        num_of_clusters = trial.suggest_int("num_of_clusters", 2, 15)
        clust_algo = KmeansClustering(n_clusters=num_of_clusters, scaler=scaler)
    else:
        clust_algo = "passthrough"

    class_w = trial.suggest_categorical("class_weights", ["balanced", None])
    estimator = eval_model(class_weight=class_w, random_state=42, n_jobs=-1)

    num_cat_transf = ColumnTransformer(
        [("numeric", scaler, num_col_idx), ("cat_enc", encoder, cat_col_idx),]
    )

    pipeline = make_pipeline(num_cat_transf, clust_algo, estimator)

    score = cross_val_score(pipeline, X, y, cv=cv, scoring=metric)
    return np.mean(score)


def tune_pipeline(
    objective_func: Callable,
    direction: str,
    n_trials: int,
    X: np.ndarray,
    y: np.ndarray,
    eval_model: BaseEstimator,
    num_col_idx: List[int],
    cat_col_idx: List[int],
    cv: StratifiedKFold,
    metric: str,
) -> Tuple[float, dict]:
    """Funtion to tune a numerical pipeline. Returns best value and tuned
    pipeline hyper-parameters"""

    study = optuna.create_study(direction=direction)
    func = lambda trial: objective_func(
        trial, X, y, eval_model, num_col_idx, cat_col_idx, cv, metric
    )
    study.optimize(func, n_trials=n_trials)

    return round(study.best_value, 3), study.best_params


def pipeline_objective_4(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    eval_model: BaseEstimator,
    num_col_idx: List[int],
    cat_col_idx: List[int],
    cv: StratifiedKFold,
    metric: str,
) -> float:
    "Pre-processing pipeline optimization objective function"

    scalers = trial.suggest_categorical("scalers", ["minmax", "standard", "robust"])
    if scalers == "minmax":
        scaler = MinMaxScaler()
    elif scalers == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    cat_encoders = trial.suggest_categorical("cat_encoders", ["ordinal", "one_hot"])
    if cat_encoders == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="ignore")
    else:
        encoder = OneHotEncoder(handle_unknown="ignore")

    clustering = trial.suggest_categorical("clustering", ["KMeans", None])
    if clustering == "KMeans":
        num_of_clusters = trial.suggest_int("num_of_clusters", 2, 15)
        clust_algo = KmeansClustering(n_clusters=num_of_clusters, scaler=scaler)
    else:
        clust_algo = "passthrough"

    class_w = trial.suggest_categorical("class_weights", [False, True])
    
    estimator = eval_model(objective="binary", is_unbalance=class_w, random_state=42, n_jobs=-1)

    num_cat_transf = ColumnTransformer(
        [("numeric", scaler, num_col_idx), ("cat_enc", encoder, cat_col_idx),]
    )

    pipeline = make_pipeline(num_cat_transf, clust_algo, estimator)

    score = cross_val_score(pipeline, X, y, cv=cv, scoring=metric)
    return np.mean(score)


def pipeline_objective_smotenc(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    eval_model: BaseEstimator,
    cat_col_idx: List[int],
    scaler: Callable,
    cv: StratifiedKFold,
    metric: str,
) -> float:
    "Pre-processing pipeline optimization objective function with Smotenc"

    smote_num = trial.suggest_categorical("num_smote_method", ["Smotenc", None])
    if smote_num == "smotenc":
        smote = SMOTENC(categorical_features=cat_col_idx, random_state=42)
    else:
        smote = "passthrough"

    clustering = trial.suggest_categorical("clustering", ["KMeans", None])
    if clustering == "KMeans":
        num_of_clusters = trial.suggest_int("num_of_clusters", 2, 15)
        clust_algo = KmeansClustering(n_clusters=num_of_clusters, scaler=scaler)
    else:
        clust_algo = "passthrough"

    centroids = trial.suggest_categorical("kmeans_centroids", ["KMeansCentroids", None])
    if centroids == "KMeansCentroids":
        num_of_clusters = trial.suggest_int("num_of_centroids", 2, 15)
        centroids_algo = KmeansClusterDistance(
            n_clusters=num_of_clusters, scaler=scaler
        )
    else:
        centroids_algo = "passthrough"

    class_w = trial.suggest_categorical("class_weights", ["balanced", None])
    estimator = eval_model(random_state=42, class_weight=class_w, n_jobs=-1)

    pipeline = imb_make_pipeline(smote, clust_algo, centroids_algo, estimator)

    score = cross_val_score(pipeline, X, y, cv=cv, scoring=metric)
    return np.mean(score)


def pipeline_objective_smotenc_2(
    trial,
    X: np.ndarray,
    y: np.ndarray,
    eval_model: BaseEstimator,
    cat_col_idx: List[int],
    scaler: Callable,
    cv: StratifiedKFold,
    metric: str,
) -> float:
    "Pre-processing pipeline optimization objective function with Smotenc"

    smote_num = trial.suggest_categorical("num_smote_method", ["Smotenc", None])
    if smote_num == "smotenc":
        smote = SMOTENC(categorical_features=cat_col_idx, random_state=42)
    else:
        smote = "passthrough"

    clustering = trial.suggest_categorical("clustering", ["KMeans", None])
    if clustering == "KMeans":
        num_of_clusters = trial.suggest_int("num_of_clusters", 2, 15)
        clust_algo = KmeansClustering(n_clusters=num_of_clusters, scaler=scaler)
    else:
        clust_algo = "passthrough"

    centroids = trial.suggest_categorical("kmeans_centroids", ["KMeansCentroids", None])
    if centroids == "KMeansCentroids":
        num_of_clusters = trial.suggest_int("num_of_centroids", 2, 15)
        centroids_algo = KmeansClusterDistance(
            n_clusters=num_of_clusters, scaler=scaler
        )
    else:
        centroids_algo = "passthrough"

    cat_transf = ColumnTransformer(
        [("cat_enc", OneHotEncoder(handle_unknown="ignore"), cat_col_idx)],
        remainder="passthrough",
    )

    class_w = trial.suggest_categorical("class_weights", ["balanced", None])
    estimator = eval_model(class_weight=class_w, random_state=42, n_jobs=-1)

    pipeline = imb_make_pipeline(
        smote, cat_transf, clust_algo, centroids_algo, estimator
    )

    score = cross_val_score(pipeline, X, y, cv=cv, scoring=metric)
    return np.mean(score)


def tune_pipeline_smotenc(
    objective_func: Callable,
    direction: str,
    n_trials: int,
    X: np.ndarray,
    y: np.ndarray,
    eval_model: BaseEstimator,
    cat_col_idx: List[int],
    scaler: Callable,
    cv: StratifiedKFold,
    metric: str,
) -> Tuple[float, dict]:
    """Funtion to tune a numerical pipeline. Returns best value and tuned
    pipeline hyper-parameters"""

    study = optuna.create_study(direction=direction)
    func = lambda trial: objective_func(
        trial, X, y, eval_model, cat_col_idx, scaler, cv, metric
    )
    study.optimize(func, n_trials=n_trials)

    return round(study.best_value, 3), study.best_params


def xgboost_hptuned_eval_cv(
    clf: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    num_columns_idx: List[int],
) -> pd.DataFrame:
    """Takes a XGBoost model with tuned hyper-parameters, training set, 
    training labels, numerical columns list and returns different classification scores in
    a DataFrame"""

    scores = defaultdict(list)
    best_ntree_limit = np.empty(cv.n_splits)

    start = time.time()
    scores["Classifier"].append(clf.__class__.__name__)

    for i, metric in enumerate(
        [
            balanced_accuracy_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            average_precision_score,
            roc_auc_score,
        ]
    ):

        score_name = (
            metric.__name__.replace("_", " ").replace("score", "").capitalize().strip()
        )
        cv_scores = np.empty(cv.n_splits)
        for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            numeric_scaler = ColumnTransformer(
                [("numeric", StandardScaler(), num_columns_idx)],
                remainder="passthrough",
            )
            X_train_tr = numeric_scaler.fit_transform(X_train)
            X_val_tr = numeric_scaler.transform(X_val)

            clf.fit(
                X_train_tr,
                y_train,
                eval_set=[(X_val_tr, y_val)],
                early_stopping_rounds=100,
                verbose=False,
            )

            if score_name in ["Average precision", "Roc auc"]:
                y_val_pred = clf.predict_proba(X_val_tr)[:, 1]
            else:
                y_val_pred = clf.predict(X_val_tr)
            cv_scores[idx] = metric(y_val, y_val_pred)
            best_ntree_limit[idx] = clf.best_ntree_limit
        if i == 0:
            scores[f"Max numb of trees_{score_name}"] = int(max(best_ntree_limit))

        scores[score_name].append(np.mean(cv_scores))

    end = time.time()
    scores["Total time in sec"].append((end - start))

    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df = score_df.round(3)

    return score_df


def xgboost_reg_hptuned_eval_cv(
    reg: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: KFold,
    num_columns_idx: List[int],
) -> pd.DataFrame:
    """Takes a XGBoost Regression model with tuned hyper-parameters, training set, 
    training labels, numerical columns list and returns different regression scores in
    a DataFrame"""

    scores = defaultdict(list)
    best_ntree_limit = np.empty(cv.n_splits)

    start = time.time()
    scores["Regressor"].append(reg.__class__.__name__)

    for i, metric in enumerate([mean_squared_error, mean_squared_error, r2_score]):

        score_name = (
            metric.__name__.replace("_", " ").replace("score", "").capitalize().strip()
        )
        cv_scores = np.empty(cv.n_splits)
        for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            numeric_scaler = ColumnTransformer(
                [("numeric", StandardScaler(), num_columns_idx)],
                remainder="passthrough",
            )
            X_train_tr = numeric_scaler.fit_transform(X_train)
            X_val_tr = numeric_scaler.transform(X_val)

            reg.fit(
                X_train_tr,
                y_train,
                eval_set=[(X_val_tr, y_val)],
                early_stopping_rounds=100,
                verbose=False,
            )

            y_val_pred = reg.predict(X_val_tr)
            if i == 1:
                cv_scores[idx] = metric(y_val, y_val_pred, squared=False)
            else:
                cv_scores[idx] = metric(y_val, y_val_pred)
            best_ntree_limit[idx] = reg.best_ntree_limit
        if i == 0:
            scores[f"Max numb of trees_Root {score_name.lower()}"] = int(
                max(best_ntree_limit)
            )
        if i == 1:
            score_name = f"Root {score_name.lower()}"
        scores[score_name].append(np.mean(cv_scores))

    end = time.time()
    scores["Total time in sec"].append((end - start))

    score_df = pd.DataFrame(scores).set_index("Regressor")
    score_df = score_df.round(3)

    return score_df


def light_gbm_hptuned_eval_cv(
    clf: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Callable,
    cv: StratifiedKFold,
    num_columns_idx: List[int],
    multi_class: bool = None,
) -> pd.DataFrame:
    """Takes a LightGBM model with tuned hyper-parameters, training set, 
    training labels, numerical columns list and returns different classification scores in
    a DataFrame"""

    scores = defaultdict(list)
    best_ntree_limit = np.empty(cv.n_splits)

    start = time.time()
    scores["Classifier"].append(clf.__class__.__name__)

    for i, metric in enumerate(
        [
            balanced_accuracy_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            average_precision_score,
            roc_auc_score,
        ]
    ):

        score_name = (
            metric.__name__.replace("_", " ").replace("score", "").capitalize().strip()
        )
        if multi_class and score_name in [
            "Precision",
            "Recall",
            "Average precision",
            "Roc auc",
        ]:
            continue
        cv_scores = np.empty(cv.n_splits)
        if multi_class and score_name == "F1":
            score_name = f"{score_name} {multi_class}"
        for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            numeric_scaler = ColumnTransformer(
                [("numeric", scaler, num_columns_idx)], remainder="passthrough",
            )
            X_train_tr = numeric_scaler.fit_transform(X_train)
            X_val_tr = numeric_scaler.transform(X_val)

            clf.fit(
                X_train_tr,
                y_train,
                eval_set=[(X_val_tr, y_val)],
                callbacks=[early_stopping(100)],
            )

            if score_name in ["Average precision", "Roc auc"]:
                y_val_pred = clf.predict_proba(X_val_tr)[:, 1]
            else:
                y_val_pred = clf.predict(X_val_tr)

            if score_name == "F1 macro":
                cv_scores[idx] = metric(y_val, y_val_pred, average=multi_class)
            else:
                cv_scores[idx] = metric(y_val, y_val_pred)
            best_ntree_limit[idx] = clf.best_iteration_

        if i == 0:
            scores["Max numb of trees"] = int(max(best_ntree_limit)) + 1

        scores[score_name].append(np.mean(cv_scores))

    end = time.time()
    scores["Total time in sec"].append((end - start))

    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df = score_df.round(3)

    return score_df


def sgdclf_hptuned_eval_cv(
    clf: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Callable,
    cv: StratifiedKFold,
    num_columns_idx: List[int],
) -> pd.DataFrame:
    """Takes a SGD Classifier model with tuned hyper-parameters, training set, 
    training labels, numerical columns list and returns different classification scores in
    a DataFrame"""

    scores = defaultdict(list)

    start = time.time()
    scores["Classifier"].append(clf.__class__.__name__)

    for i, metric in enumerate(
        [
            balanced_accuracy_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            average_precision_score,
            roc_auc_score,
        ]
    ):

        score_name = (
            metric.__name__.replace("_", " ").replace("score", "").capitalize().strip()
        )
        cv_scores = np.empty(cv.n_splits)
        for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            numeric_scaler = ColumnTransformer(
                [("numeric", scaler, num_columns_idx)], remainder="passthrough",
            )
            X_train_tr = numeric_scaler.fit_transform(X_train)
            X_val_tr = numeric_scaler.transform(X_val)

            clf.fit(X_train_tr, y_train)

            if score_name in ["Average precision", "Roc auc"]:
                y_val_pred = clf.predict_proba(X_val_tr)[:, 1]
            else:
                y_val_pred = clf.predict(X_val_tr)

            cv_scores[idx] = metric(y_val, y_val_pred)

        scores[score_name].append(np.mean(cv_scores))

    end = time.time()
    scores["Total time in sec"].append((end - start))

    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df = score_df.round(3)

    return score_df


def light_gbm_reg_hptuned_eval_cv(
    reg: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    scaler: Callable,
    cv: KFold,
    num_columns_idx: List[int],
) -> pd.DataFrame:
    """Takes a LightGBM Regression model with tuned hyper-parameters, training set, 
    training labels, numerical columns list and returns different regression scores in
    a DataFrame"""

    scores = defaultdict(list)
    best_ntree_limit = np.empty(cv.n_splits)

    start = time.time()
    scores["Regressor"].append(reg.__class__.__name__)

    for i, metric in enumerate([mean_squared_error, mean_squared_error, r2_score]):

        score_name = (
            metric.__name__.replace("_", " ").replace("score", "").capitalize().strip()
        )
        cv_scores = np.empty(cv.n_splits)
        for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            numeric_scaler = ColumnTransformer(
                [("numeric", scaler, num_columns_idx)],
                remainder="passthrough",
            )
            X_train_tr = numeric_scaler.fit_transform(X_train)
            X_val_tr = numeric_scaler.transform(X_val)

            reg.fit(
                X_train_tr,
                y_train,
                eval_set=[(X_val_tr, y_val)],
                callbacks=[early_stopping(100)],
            )

            y_val_pred = reg.predict(X_val_tr)
            if i == 1:
                cv_scores[idx] = metric(y_val, y_val_pred, squared=False)
            else:
                cv_scores[idx] = metric(y_val, y_val_pred)
            best_ntree_limit[idx] = reg.best_iteration_
        if i == 0:
            scores[f"Max numb of trees_Root {score_name.lower()}"] = int(
                max(best_ntree_limit)
            )
        if i == 1:
            score_name = f"Root {score_name.lower()}"
        scores[score_name].append(np.mean(cv_scores))

    end = time.time()
    scores["Total time in sec"].append((end - start))

    score_df = pd.DataFrame(scores).set_index("Regressor")
    score_df = score_df.round(3)

    return score_df


def catboost_hptuned_eval_cv(
    clf: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    num_columns_idx: List[int],
) -> pd.DataFrame:
    """Takes a CatBoost model with tuned hyper-parameters, training set, 
    training labels, numerical columns list and returns different classification scores in
    a DataFrame"""

    scores = defaultdict(list)
    best_ntree_limit = np.empty(cv.n_splits)

    start = time.time()
    scores["Classifier"].append(clf.__class__.__name__)

    for i, metric in enumerate(
        [
            balanced_accuracy_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            average_precision_score,
            roc_auc_score,
        ]
    ):

        score_name = (
            metric.__name__.replace("_", " ").replace("score", "").capitalize().strip()
        )
        cv_scores = np.empty(cv.n_splits)
        for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            numeric_scaler = ColumnTransformer(
                [("numeric", StandardScaler(), num_columns_idx)],
                remainder="passthrough",
            )
            X_train_tr = numeric_scaler.fit_transform(X_train)
            X_val_tr = numeric_scaler.transform(X_val)

            clf.fit(
                X_train_tr,
                y_train,
                eval_set=[(X_val_tr, y_val)],
                early_stopping_rounds=100,
            )

            if score_name in ["Average precision", "Roc auc"]:
                y_val_pred = clf.predict_proba(X_val_tr)[:, 1]
            else:
                y_val_pred = clf.predict(X_val_tr)
            cv_scores[idx] = metric(y_val, y_val_pred)
            best_ntree_limit[idx] = clf.get_best_iteration()

        if i == 0:
            scores[f"Max numb of trees_{score_name}"] = int(max(best_ntree_limit)) + 1

        scores[score_name].append(np.mean(cv_scores))

    end = time.time()
    scores["Total time in sec"].append((end - start))

    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df = score_df.round(3)

    return score_df


def catboost_reg_hptuned_eval_cv(
    reg: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: KFold,
    num_columns_idx: List[int],
) -> pd.DataFrame:
    """Takes a CatBoost Regression model with tuned hyper-parameters, training set, 
    training labels, numerical columns list and returns different regression scores in
    a DataFrame"""

    scores = defaultdict(list)
    best_ntree_limit = np.empty(cv.n_splits)

    start = time.time()
    scores["Regressor"].append(reg.__class__.__name__)

    for i, metric in enumerate([mean_squared_error, mean_squared_error, r2_score]):

        score_name = (
            metric.__name__.replace("_", " ").replace("score", "").capitalize().strip()
        )
        cv_scores = np.empty(cv.n_splits)
        for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            numeric_scaler = ColumnTransformer(
                [("numeric", StandardScaler(), num_columns_idx)],
                remainder="passthrough",
            )
            X_train_tr = numeric_scaler.fit_transform(X_train)
            X_val_tr = numeric_scaler.transform(X_val)

            reg.fit(
                X_train_tr,
                y_train,
                eval_set=[(X_val_tr, y_val)],
                early_stopping_rounds=100,
            )

            y_val_pred = reg.predict(X_val_tr)
            if i == 1:
                cv_scores[idx] = metric(y_val, y_val_pred, squared=False)
            else:
                cv_scores[idx] = metric(y_val, y_val_pred)
            best_ntree_limit[idx] = reg.get_best_iteration()
        if i == 0:
            scores[f"Max numb of trees_Root {score_name.lower()}"] = int(
                max(best_ntree_limit)
            )
        if i == 1:
            score_name = f"Root {score_name.lower()}"
        scores[score_name].append(np.mean(cv_scores))

    end = time.time()
    scores["Total time in sec"].append((end - start))

    score_df = pd.DataFrame(scores).set_index("Regressor")
    score_df = score_df.round(3)

    return score_df


def clfmodels_eval_test(
    clf_list: List[BaseEstimator],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    multi_class: bool = None,
) -> pd.DataFrame:
    """Takes a list of models, transformation pipeline, training set, 
    training labels, test set, test labels and returns
    different classification scores on a test set in a DataFrame"""

    scores = defaultdict(list)

    for clf in clf_list:
        start = time.time()
        scores["Classifier"].append(clf.__class__.__name__)
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)

        for metric in [
            balanced_accuracy_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            average_precision_score,
            roc_auc_score,
        ]:

            score_name = (
                metric.__name__.replace("_", " ")
                .replace("score", "")
                .capitalize()
                .strip()
            )
            if multi_class and score_name in [
                "Precision",
                "Recall",
                "Average precision",
                "Roc auc",
            ]:
                continue
            if score_name in ["Average precision", "Roc auc"]:
                y_test_pred = clf.predict_proba(X_test)[:, 1]
            if multi_class and score_name == "F1":
                score_name = f"{score_name} {multi_class}"
                scores[score_name].append(
                    metric(y_test, y_test_pred, average=multi_class)
                )
            else:
                scores[score_name].append(metric(y_test, y_test_pred))

        end = time.time()
        scores["Total time in sec"].append((end - start))

    score_df = pd.DataFrame(scores).set_index("Classifier")
    score_df = score_df.round(3)

    return score_df


def regmodels_eval_test(
    reg_list: List[BaseEstimator],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Takes a list of models, transformation pipeline, training set, 
    training labels, test set, test labels and returns
    different regression scores on a test set in a DataFrame"""

    scores = defaultdict(list)

    for reg in reg_list:
        start = time.time()
        scores["Regressor"].append(reg.__class__.__name__)
        reg.fit(X_train, y_train)
        y_test_pred = reg.predict(X_test)

        for i, metric in enumerate([mean_squared_error, mean_squared_error, r2_score]):

            score_name = (
                metric.__name__.replace("_", " ")
                .replace("score", "")
                .capitalize()
                .strip()
            )
            if i == 1:
                score_name = f"Root {score_name.lower()}"
                scores[score_name].append(metric(y_test, y_test_pred, squared=False))
            else:
                scores[score_name].append(metric(y_test, y_test_pred))
        end = time.time()
        scores["Total time in sec"].append((end - start))

    score_df = pd.DataFrame(scores).set_index("Regressor")
    score_df = score_df.round(3)

    return score_df


def votingclf_scores(
    votingclf: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Takes a voting model, training set, 
    training labels, test set and returns
    different classification scores on a test set in a DataFrame"""

    scores = defaultdict(list)

    for voting in ("hard", "soft"):

        start = time.time()
        scores["Classifier"].append(f"{voting.capitalize()} Voting Classifier")
        votingclf.set_params(voting=voting)
        votingclf.fit(X_train, y_train)
        y_test_pred = votingclf.predict(X_test)

        for metric in [
            balanced_accuracy_score,
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            average_precision_score,
            roc_auc_score,
        ]:
            score_name = (
                metric.__name__.replace("_", " ")
                .replace("score", "")
                .capitalize()
                .strip()
            )
            if score_name in ["Average precision", "Roc auc"]:
                if voting == "hard":
                    scores[score_name].append(np.nan)
                else:
                    y_test_pred = votingclf.predict_proba(X_test)[:, 1]
                    scores[score_name].append(metric(y_test, y_test_pred))
            else:
                scores[score_name].append(metric(y_test, y_test_pred))

        end = time.time()
        scores["Total time in sec"].append((end - start))

    voting_score_df = pd.DataFrame(scores).set_index("Classifier")
    voting_score_df = voting_score_df.round(3)

    return voting_score_df


def stackingclf_scores(
    stackingclf: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Takes a stacking model, training set, 
    training labels, test set and returns
    different classification scores on a test set in a DataFrame"""

    scores = defaultdict(list)

    start = time.time()
    stackingclf.fit(X_train, y_train)
    y_test_pred = stackingclf.predict(X_test)
    scores["Classifier"].append(
        f"Stacking Classifier with Logistic Reg as final estimator"
    )

    for metric in [
        balanced_accuracy_score,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        average_precision_score,
        roc_auc_score,
    ]:
        score_name = (
            metric.__name__.replace("_", " ").replace("score", "").capitalize().strip()
        )
        if score_name in ["Average precision", "Roc auc"]:
            y_test_pred = stackingclf.predict_proba(X_test)[:, 1]
            scores[score_name].append(metric(y_test, y_test_pred))
        else:
            scores[score_name].append(metric(y_test, y_test_pred))

    end = time.time()
    scores["Total time in sec"].append((end - start))

    stacking_score_df = pd.DataFrame(scores).set_index("Classifier")
    stacking_score_df = stacking_score_df.round(3)

    return stacking_score_df


def quartile_proportions(data: pd.DataFrame, column_name: str) -> pd.Series:
    "Counts proporttions of a column values and returns series"

    return data[column_name].value_counts() / len(data)


def target_stratification_test(data: pd.DataFrame, target: str) -> pd.DataFrame:
    """Takes pandas data frame and target column name and returns a random train_test split,
    a stratified by quartiles train_test split and errors percentage"""

    df = data.copy()
    quartile_list = [0, 0.25, 0.5, 0.75, 1.0]
    labels = [0, 1, 2, 3]
    df[f"{target}_quartiles"] = pd.qcut(df[target], quartile_list, labels)
    train_data_random, test_data_random = train_test_split(df, random_state=42)
    train_data_strat, test_data_strat = train_test_split(
        df, stratify=df[f"{target}_quartiles"], random_state=42
    )
    overall = quartile_proportions(df, f"{target}_quartiles")
    random_train = quartile_proportions(train_data_random, f"{target}_quartiles")
    strat_train = quartile_proportions(train_data_strat, f"{target}_quartiles")
    random_test = quartile_proportions(test_data_random, f"{target}_quartiles")
    strat_test = quartile_proportions(test_data_strat, f"{target}_quartiles")

    compare_props = pd.DataFrame(
        {
            "Overall": overall,
            "Random_train_set": random_train,
            "Stratified_train_set": strat_train,
            "Random_test_set": random_test,
            "Stratified_test_set": strat_test,
        }
    ).sort_index()

    compare_props["Rand_train_set %error"] = (
        100 * compare_props["Random_train_set"] / compare_props["Overall"] - 100
    )
    compare_props["Strat_train_set %error"] = (
        100 * compare_props["Stratified_train_set"] / compare_props["Overall"] - 100
    )
    compare_props["Rand_test_set %error"] = (
        100 * compare_props["Random_test_set"] / compare_props["Overall"] - 100
    )
    compare_props["Strat_test_set %error"] = (
        100 * compare_props["Stratified_train_set"] / compare_props["Overall"] - 100
    )

    return compare_props


def stratify_regression_data(
    data: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, ...]:
    """Takes a pandas DataFrame and returns train and test sets stratified by quartiles
    for data and target"""

    df = data.copy()
    quartile_list = [0, 0.25, 0.5, 0.75, 1.0]
    labels = [0, 1, 2, 3]
    df[f"{target}_quartiles"] = pd.qcut(df[target], quartile_list, labels)
    Y = df[target].copy()
    df = df.drop(target, axis=1)

    (
        train_data_strat,
        test_data_strat,
        train_target_strat,
        test_target_strat,
    ) = train_test_split(df, Y, stratify=df[f"{target}_quartiles"], random_state=42)

    for set_ in (train_data_strat, test_data_strat):
        set_.drop(f"{target}_quartiles", axis=1, inplace=True)

    return train_data_strat, test_data_strat, train_target_strat, test_target_strat


def multiple_models_prediction(model_data_lst: List[Tuple]) -> List[np.ndarray]:
    "Takes model and data in a tuple from the list and outputs predictions"

    results = []
    model_type = [is_classifier(model[0]) for model in model_data_lst]
    for i in range(len(model_type)):
        if model_type[i]:
            results.append(
                model_data_lst[i][0].predict_proba(model_data_lst[i][1])[:, 1]
            )

        else:
            results.append(model_data_lst[i][0].predict(model_data_lst[i][1]))

    return results


def data_preparation(user_input: Dict) -> pd.DataFrame:
    """Takes a dict from Flask API user input form and returns prepared
    pd.DataFrame for model predictions"""

    for key, value in user_input.items():
        try:
            user_input[key] = float(value)
        except:
            user_input[key] = value
    df_data = pd.DataFrame([user_input])
    return df_data


def clf_model_prediction(
    model: BaseEstimator, pipe: Pipeline, data: pd.DataFrame
) -> np.ndarray:
    """Takes model, pre-processing pipeline and a dataframe and returns prediction results
    of classification model"""

    data_tr = pipe.transform(data)
    data_pred = model.predict_proba(data_tr)[:, 1]
    return data_pred


def reg_model_prediction(
    model: BaseEstimator, pipe: Pipeline, data: pd.DataFrame
) -> np.ndarray:
    """Takes model, pre-processing pipeline and a dataframe and returns prediction results
    of regression model"""

    data_tr = pipe.transform(data)
    data_pred = model.predict(data_tr)
    return data_pred


def get_calibration_curve_values(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """Takes model, train and test sets and returns fraction of positive values,
    mean predicted probability values, Brier and Log Loss scores"""

    scores = {}
    model.fit(X_train, y_train)
    y_predicted = model.predict_proba(X_test)[:, 1]
    for metric in [brier_score_loss, log_loss]:

        score_name = (
            metric.__name__.replace("_", " ").replace(" score", "").capitalize()
        )
        scores[score_name] = round(metric(y_test, y_predicted), 3)

    fraction_of_positives, mean_predict_value = calibration_curve(
        y_test, y_predicted, n_bins=10
    )

    return fraction_of_positives, mean_predict_value, scores


def loan_amnt_cap(df: pd.DataFrame) -> pd.DataFrame:
    "Caps a loan_amnt at maximum number allowed by LendingClub"
    df = df.copy()
    return df[df["loan_amnt"] <= 40000]


def percentile_outliers_detect(df: pd.DataFrame) -> pd.DataFrame:
    "Removes outliers from the dataframe based on quantile method"

    df = df.copy()
    out_cap = 100
    df["dti"][df["dti"] > out_cap] = out_cap
    df["dti"][df["dti"] < 0] = out_cap
    return df


def annual_inc_outliers_detect(df: pd.DataFrame) -> pd.DataFrame:
    "Removes outliers from the dataframe based on quantile method"

    df = df.copy()
    out_cap = 300000
    df["annual_inc"][df["annual_inc"] > out_cap] = out_cap
    return df


def get_month(df: pd.DataFrame) -> pd.DataFrame:
    "Gets a month from year/month/day string format"

    df = df.copy()
    df["month"] = pd.DatetimeIndex(df["date"]).month
    df.drop("date", axis=1, inplace=True)
    return df


def cyclic_month(df: pd.DataFrame) -> pd.DataFrame:
    "Transfors month into sin and cos cyclic values"

    df = df.copy()
    df["month_sin"] = np.sin(df["month"] / 12 * 2 * np.pi)
    df["month_cos"] = np.cos(df["month"] / 12 * 2 * np.pi)
    df.drop("month", axis=1, inplace=True)
    return df


def emp_length_missing_imputer(df: pd.DataFrame) -> pd.DataFrame:
    "Imputes missing values in eml_length column"

    df = df.copy()
    df["emp_length"] = df["emp_length"].fillna("Unknow")
    return df


def states_binning(df: pd.DataFrame) -> pd.DataFrame:
    "Bins US states in 4 regions"

    df = df.copy()
    df["state_region"] = df["addr_state"].apply(get_state_region)
    df.drop("addr_state", axis=1, inplace=True)
    return df


def get_earliest_cr_line_month_year(df: pd.DataFrame) -> pd.DataFrame:
    "Gets a year and month from year/month string format"

    df = df.copy()
    df["earliest_cr_line_year"] = pd.DatetimeIndex(df["earliest_cr_line"]).year
    df["earliest_cr_line_month"] = pd.DatetimeIndex(df["earliest_cr_line"]).month
    df.drop("earliest_cr_line", axis=1, inplace=True)
    return df


def cyclic_earliest_cr_line_month(df: pd.DataFrame) -> pd.DataFrame:
    "Transfors month into sin and cos cyclic values"

    df = df.copy()
    df["cr_line_month_sin"] = np.sin(df["earliest_cr_line_month"] / 12 * 2 * np.pi)
    df["cr_line_month_cos"] = np.cos(df["earliest_cr_line_month"] / 12 * 2 * np.pi)
    df.drop("earliest_cr_line_month", axis=1, inplace=True)
    return df


def domain_feature_creation(df: pd.DataFrame) -> pd.DataFrame:
    "Creates some new features from existing data"

    df = df.copy()
    df["dti"] = round(df["amt_annuity"] / df["amt_income_total"] * 100, 2)
    df["loans_term_months"] = round(df["amt_credit"] / df["amt_annuity"])
    df["credit_income_pct"] = round(df["amt_credit"] / df["amt_income_total"] * 100, 2)
    df["days_employed_pct"] = round(df["days_employed"] / df["days_birth"] * 100, 2)
    return df


def cyclic_weekday_appr_process_start(df: pd.DataFrame) -> pd.DataFrame:
    "Transfors weekdays into sin and cos cyclic values"

    map_dict = {
        "MONDAY": 0,
        "TUESDAY": 1,
        "WEDNESDAY": 2,
        "THURSDAY": 3,
        "FRIDAY": 4,
        "SATURDAY": 5,
        "SUNDAY": 6,
    }
    df = df.copy()
    df["weekday_appr_process_start"] = df["weekday_appr_process_start"].replace(
        map_dict
    )
    df["weekday_appr_process_start_sin"] = np.sin(
        df["weekday_appr_process_start"] / 7 * 2 * np.pi
    )
    df["weekday_appr_process_start_cos"] = np.cos(
        df["weekday_appr_process_start"] / 7 * 2 * np.pi
    )
    df.drop("weekday_appr_process_start", axis=1, inplace=True)
    return df


class SGDWithThreshold(SGDClassifier):
    "SGD Classifier with variuos threshold tuning methods"

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        if threshold == None:
            return self.predict(X)
        else:
            y_predicted = self.predict_proba(X)[:, 1]
            y_pred_with_threshold = (y_predicted >= threshold).astype(int)

            return y_pred_with_threshold

    def threshold_optimal_f_score(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_predicted = self.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_predicted)
        fscores = (2 * precisions * recalls) / (precisions + recalls)
        optimal_idx = np.argmax(fscores)

        return thresholds[optimal_idx], fscores[optimal_idx]

    def threshold_desired_precision(
        self, X: np.ndarray, y: np.ndarray, desired_precision: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_predicted = self.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_predicted)

        desired_precision_idx = np.argmax(precisions >= desired_precision)

        return thresholds[desired_precision_idx], recalls[desired_precision_idx]

    def threshold_from_desired_recall(
        self, X: np.ndarray, y: np.ndarray, desired_recall: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_predicted = self.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_predicted)

        desired_recall_idx = np.argmin(recalls >= desired_recall)

        return thresholds[desired_recall_idx], precisions[desired_recall_idx]


class LGBMWithThreshold(LGBMClassifier):
    "LGBM Classifier with variuos threshold tuning methods"

    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        if threshold == None:
            return self.predict(X)
        else:
            y_predicted = self.predict_proba(X)[:, 1]
            y_pred_with_threshold = (y_predicted >= threshold).astype(int)

            return y_pred_with_threshold

    def threshold_from_optimal_f_score(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_predicted = self.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_predicted)
        fscores = (2 * precisions * recalls) / (precisions + recalls)
        optimal_idx = np.argmax(fscores)

        return thresholds[optimal_idx], fscores[optimal_idx]

    def threshold_from_desired_precision(
        self, X: np.ndarray, y: np.ndarray, desired_precision: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_predicted = self.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_predicted)

        desired_precision_idx = np.argmax(precisions >= desired_precision)

        return thresholds[desired_precision_idx], recalls[desired_precision_idx]

    def threshold_from_desired_recall(
        self, X: np.ndarray, y: np.ndarray, desired_recall: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_predicted = self.predict_proba(X)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(y, y_predicted)

        desired_recall_idx = np.argmin(recalls >= desired_recall)

        return thresholds[desired_recall_idx], precisions[desired_recall_idx]
    
    
def make_mi_scores(df: pd.DataFrame, target: str, model_type: str) -> pd.Series:
    "Returns Mutual Information scores by scikit-learn model"

    data = df.copy()
    data.dropna(inplace=True)
    target = data.pop(target)
    for colname in data.select_dtypes(["object", "category"]):
        data[colname], _ = data[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in data.dtypes]
    if model_type == 'classification':
        mi_scores = mutual_info_classif(
            data, target, discrete_features=discrete_features, random_state=42
            )
    else:
        mi_scores = mutual_info_regression(
            data, target, discrete_features=discrete_features, random_state=42
            )
    mi_scores = pd.Series(mi_scores, name="mi_scores", index=data.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    return mi_scores


def plot_mi_scores(scores: pd.Series, log_scale: bool = None) -> plt.figure:
    "PLots mutual information scores from make_mi_scores function"

    fig, ax = plt.subplots(figsize=(8, 14))

    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title(
        "Mutual Information Scores", fontsize=13, y=1.02,
    )
    if log_scale:
        plt.xscale("log")

    plt.show()
    

def shap_values_feature_importance(
    df: pd.DataFrame, model: Callable, target: str, log_scale=False
) -> plt.figure:
    "Plots shap summary plot to investigate feature importance of the dataset"

    data = df.copy()
    data.dropna(inplace=True)
    target = data.pop(target)
    for colname in data.select_dtypes(["object", "category"]):
        data[colname], _ = data[colname].factorize()
    if is_classifier(model):
        X_train, X_test, Y_train, Y_test = train_test_split(
            data, target, stratify=target, random_state=42
        )
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(
            data, target, random_state=42
        )
    model.fit(X_train, Y_train)
    feature_explainer = shap.Explainer(model, X_test)
    shap_values = feature_explainer.shap_values(X_test)

    shap.summary_plot(
        shap_values,
        X_test,
        max_display=X_test.shape[1],
        class_names=np.unique(Y_test),
        plot_size=(8, 16),
        use_log_scale=log_scale,
    )


def get_shap_feature_names(
    df: pd.DataFrame, model: Callable, target: str
) -> pd.DataFrame:
    "Puts shap feature importance values into pandas DataFrame"

    data = df.copy()
    data.dropna(inplace=True)
    target = data.pop(target)
    for colname in data.select_dtypes(["object", "category"]):
        data[colname], _ = data[colname].factorize()
    if is_classifier(model):
        X_train, X_test, Y_train, Y_test = train_test_split(
            data, target, stratify=target, random_state=42
        )
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(
            data, target, random_state=42
        )
    model.fit(X_train, Y_train)
    feature_explainer = shap.Explainer(model, X_test)
    shap_values = feature_explainer.shap_values(X_test)

    values = np.abs(shap_values).mean(0) 
    if is_classifier(model):
        feature_importance = pd.DataFrame(
            list(zip(X_test.columns, values)), #change to sum(values) for multiclass clf
            columns=["col_name", "feature_importance_value"],
        )
    else:
        feature_importance = pd.DataFrame(
            list(zip(X_test.columns, values)),
            columns=["col_name", "feature_importance_value"],
        )
    feature_importance.sort_values(
        by=["feature_importance_value"], ascending=False, inplace=True
    )
    
    return feature_importance


def subgrade_models(
    df: pd.DataFrame, pipeline: Pipeline, grade: str
) -> Tuple[pd.DataFrame, Callable]:
    "Train and return model to predict each grade's subgrades"

    data = df.loc[df["grade"] == grade]
    target_data = data["sub_grade"].copy()
    model_data = data.drop(["sub_grade", "grade"], axis=1)
    train_data, test_data, train_target, test_target = train_test_split(
        model_data, target_data, stratify=target_data, random_state=42
    )
    train_data_tr = pipeline.fit_transform(train_data)
    test_data_tr = pipeline.transform(test_data)
    clf_model = LGBMClassifier(objective="multiclass", num_class=5, random_state=42)
    results = clfmodels_eval_test(
        [clf_model], train_data_tr, train_target, test_data_tr, test_target, "macro"
    )

    return results, clf_model


def subgrade_model_test(
    df: pd.DataFrame, pipeline: Pipeline, models_dict: Dict[str, Callable]
) -> pd.Series:
    "Test a subgrade model on data with grade model inputs"

    pred_lst = []

    for grade in sorted(df.grade.unique()):
        data = df.loc[df["grade"] == grade]
        data_tr = pipeline.transform(data)
        predictions = models_dict[grade].predict(data_tr)
        predictions = pd.Series(data=predictions, index=data.index)
        pred_lst.append(predictions)

    series = pred_lst[0]
    for i in range(1, len(pred_lst)):
        series = pd.concat([series, pred_lst[i]])

    return series


def intrate_models(
    df: pd.DataFrame, pipeline: Pipeline, reg_model: Callable
) -> Dict[pd.DataFrame, Callable]:
    "Train and return model to predict each grade's subgrade's interest rate"

    reg_models_dict = {}
    for grade in sorted(df.grade.unique()):
        for subgrade in sorted(df.sub_grade.unique()):
            if subgrade.startswith(grade):
                data = df.loc[(df["grade"] == grade) & (df["sub_grade"] == subgrade)]
                target_data = data["int_rate"].copy()
                model_data = data.drop(["sub_grade", "grade", "int_rate"], axis=1)
                train_data, test_data, train_target, test_target = train_test_split(
                    model_data, target_data, random_state=42
                )
                train_data_tr = pipeline.fit_transform(train_data)
                test_data_tr = pipeline.transform(test_data)
                results = regmodels_eval_test(
                    [reg_model], train_data_tr, train_target, test_data_tr, test_target
                )
                reg_models_dict[
                    f"Grade_{grade.lower()}_subgrade_{subgrade.lower()}_intrate"
                ] = (results, reg_model)
        else:
            continue

    return reg_models_dict

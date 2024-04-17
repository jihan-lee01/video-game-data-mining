import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  Ridge)
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor


def import_data():
    return pd.read_csv("data/video_games.csv")


def preprocessing(df):
    # Add index column for later purpose to sort out duplicates
    df['Index'] = range(1, 1213)

    # Divide features and targets
    targets = ['Metrics.Review Score', 'Metrics.Sales']
    X = df.drop(targets, axis=1)
    y1 = df[['Metrics.Review Score']]
    y2 = df[['Metrics.Sales']]

    # Drop unnecessary columns
    unnecessary = ['Metrics.Used Price', 'Features.Handheld?', 'Features.Multiplatform?',
                   'Features.Online?', 'Metadata.Licensed?', 'Metadata.Sequel?', 'Release.Re-release?']
    X = X.drop(unnecessary, axis=1)

    # Dealing with missing values (only one column has missing values)
    x_to_impute = X[['Metadata.Publishers']]
    imputer = SimpleImputer(strategy='constant', fill_value='unknown')
    x_imputed = pd.DataFrame(imputer.fit_transform(
        x_to_impute), columns=['Metadata.Publishers'])
    X_nonimputed = X.drop('Metadata.Publishers', axis=1)
    X_imputed = pd.concat([X_nonimputed, x_imputed], axis=1)

    # Change integer columns to categorical before one-hot encoding
    X_imputed['Features.Max Players'] = X_imputed['Features.Max Players'].astype(
        'object')
    X_imputed['Release.Year'] = X_imputed['Release.Year'].astype('object')

    # Split lists of genres and publishers and explode them into different rows
    X_imputed['Metadata.Genres'] = X_imputed['Metadata.Genres'].str.split(',')
    X_imputed['Metadata.Publishers'] = X_imputed['Metadata.Publishers'].str.split(
        ',')
    X_imputed = X_imputed.explode('Metadata.Genres')
    X_imputed = X_imputed.explode('Metadata.Publishers')

    # One-hot encoding
    categorical_cols = ['Features.Max Players', 'Metadata.Genres',
                        'Release.Console', 'Release.Rating', 'Release.Year', 'Metadata.Publishers']
    X_to_encode = X_imputed[categorical_cols]
    X_dummies = pd.get_dummies(X_to_encode)
    X_nonencoded = X_imputed.drop(categorical_cols, axis=1)
    X_title_index = X_nonencoded[['Title', 'Index']]
    X_nonencoded = X_nonencoded.drop(['Title', 'Index'], axis=1)

    # Standardization
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X_nonencoded)
    X_scaled = pd.DataFrame(X_standardized, columns=X_nonencoded.columns)

    # Reset indices
    X_scaled = X_scaled.reset_index(drop=True)
    X_dummies = X_dummies.reset_index(drop=True)
    X_title_index = X_title_index.reset_index(drop=True)

    # Combine everything
    X_scaled = pd.concat([X_title_index, X_dummies, X_scaled], axis=1)

    # Remove duplicates using index
    X_scaled = X_scaled.drop_duplicates(subset=['Index'])
    X_scaled = X_scaled.drop(['Index', 'Title'], axis=1)

    return X_scaled, y1, y2


def split(X, y):
    return train_test_split(X, y, test_size=0.25)


def regression(X_train, y_train):
    lin = LinearRegression()
    lin.fit(X_train, y_train)

    ridge = Ridge()
    ridge.fit(X_train, y_train)

    lasso = Lasso()
    lasso.fit(X_train, y_train)

    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)

    forest = RandomForestRegressor()
    forest.fit(X_train, y_train)

    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)

    return [lin, ridge, lasso, dt, forest, xgb]


def reg_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }
    return metrics


def discretize_y(y):
    col_name = y.columns[0]
    y_median = y[col_name].median()
    y[col_name] = np.where(y[col_name] > y_median, 1, 0)
    return y


def classification(X_train, y_train):
    logit = LogisticRegression(penalty='l1', solver='saga')
    logit.fit(X_train, y_train)

    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)

    forest = RandomForestClassifier()
    forest.fit(X_train, y_train)

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)

    return [logit, dt, forest, xgb]


def clf_eval(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_pred_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": precision_score(y_test, y_pred)
    }
    roc = {
        "fpr": fpr,
        "tpr": tpr
    }
    return metrics, roc, confusion_matrix(y_test, y_pred)


def reg_tuning(model, param_grid, X_train, y_train, X_test, y_test):
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_model.predict(X_test)

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

    return best_params, metrics


def get_param_grid(model_name):
    if model_name == "dt_reg":
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_leaf': [1, 2, 4, 8],
            'ccp_alpha': [0, 0.001, 0.01, 0.1]
        }
    elif model_name == "dt_clf":
        param_grid = {

        }
    elif model_name == "ridge":
        param_grid = {
            'alpha': [0, 0.001, 0.01, 0.1, 1],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
        }
    elif model_name == "lasso":
        param_grid = {
            'alpha': [0, 0.001, 0.01, 0.1, 1],
            'selection': ['cyclic', 'random']
        }
    elif model_name == "forest_reg":
        param_grid = {
            'n_estimators': [20, 40, 60, 80, 100],
            'max_depth': [None, 10, 20, 30],
            'min_samples_leaf': [1, 2, 4, 8],
            'ccp_alpha': [0, 0.001, 0.01, 0.1]
        }
    elif model_name == "forest_clf":
        param_grid = {

        }
    elif model_name == "xgb_reg":
        param_grid = {
        }
    elif model_name == "xgb_clf":
        param_grid = {

        }
    elif model_name == "logit":
        param_grid = {

        }
    else:
        param_grid = {}

    return param_grid


def get_feature_importance(model, X_train, y_train):
    importances = model.feature_importances_
    perm_importances = permutation_importance(
        model, X_train, y_train).importances_mean

    imp_indices = np.argsort(importances)[::-1]
    top10_imp_indices = imp_indices[:10]

    perm_indices = np.argsort(perm_importances)[::-1]
    top10_perm_indices = perm_indices[:10]

    return importances[top10_imp_indices], perm_importances[top10_perm_indices]


def main():
    df = import_data()

    X, y1, y2 = preprocessing(df)

    X_train, X_test, y1_train, y1_test = split(X, y1)
    # X_train, X_test, y2_train, y2_test = split(X, y2)

    X_train = X_train.to_numpy()
    y1_train = y1_train.to_numpy().flatten()
    X_test = X_test.to_numpy()
    y1_test = y1_test.to_numpy().flatten()

    """
    print("----------Getting Parameters----------")
    dt_reg_param = get_param_grid("dt_reg")
    ridge_param = get_param_grid("ridge")
    lasso_param = get_param_grid("lasso")
    forest_reg_param = get_param_grid("forest_reg")

    print("----------Tuning Parameters-----------")
    best_dt_reg_param, dt_reg_metrics = reg_tuning(
        DecisionTreeRegressor(), dt_reg_param, X_train, y1_train, X_test, y1_test)
    best_ridge_param, ridge_metrics = reg_tuning(
        Ridge(), ridge_param, X_train, y1_train, X_test, y1_test)
    best_lasso_param, lasso_metrics = reg_tuning(
        Lasso(), lasso_param, X_train, y1_train, X_test, y1_test)
    best_forest_reg_param, forest_reg_metrics = reg_tuning(
        RandomForestRegressor(), forest_reg_param, X_train, y1_train, X_test, y1_test)\

    print(f"Best DT Reg Params: {
          best_dt_reg_param}/nMetrics: {dt_reg_metrics}")
    print(f"Best Ridge Reg Params: {
          best_ridge_param}/nMetrics: {ridge_metrics}")
    print(f"Best Lasso Reg Params: {
          best_lasso_param}/nMetrics: {lasso_metrics}")
    print(f"Best Random Forest Reg Params: {
          best_forest_reg_param}/nMetrics: {forest_reg_metrics}")
    """

    print("------Running Regression Models-------")
    reg_models = regression(X_train, y1_train)

    print("-----Evaluating Regression Models-----")
    reg_result = {}
    for model in reg_models:
        reg_result[model.__class__.__name__] = reg_eval(model, X_test, y1_test)

    reg_result_df = pd.DataFrame(reg_result)
    print(reg_result_df)

    y1_dis = discretize_y(y1)
    y2_dis = discretize_y(y2)

    X_train, X_test, y1_train, y1_test = split(X, y1_dis)
    # X_train, X_test, y2_train, y2_test = split(X, y2_dis)

    X_train = X_train.to_numpy()
    y1_train = y1_train.to_numpy().flatten()
    X_test = X_test.to_numpy()
    y1_test = y1_test.to_numpy().flatten()

    print("----Running Classification Models-----")
    clf_models = classification(X_train, y1_train)

    print("---Evaluating Classification Models---")
    clf_result = {}
    roc_result = {}
    matrix_result = {}
    for model in clf_models:
        metrics, roc, matrix = clf_eval(model, X_test, y1_test)
        clf_result[model.__class__.__name__] = metrics
        roc_result[model.__class__.__name__] = roc
        matrix_result[model.__class__.__name__] = matrix

    clf_result_df = pd.DataFrame(clf_result)
    print(clf_result_df)

    imp_result = {}
    perm_result = {}
    for model in clf_models[1:]:
        importances, perm_importances = get_feature_importance(
            model, X_train, y1_train)
        imp_result[model.__class__.__name__] = importances
        perm_result[model.__class__.__name__] = perm_importances


if __name__ == "__main__":
    main()

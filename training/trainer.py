from itertools import combinations
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from collections import Counter
import optuna
import copy
import lightgbm as lgb

def get_top_features(model, argList, x_train, y_train, x_test, y_test, type='reg'):
    model_fs = copy.deepcopy(model)
    num_top_features = [round(len(argList)*0.75), 
                        round(len(argList)*0.5),
                        round(len(argList)*0.25)]
    k = 0
    while round(len(argList)*0.25) - k > 0:
        num_top_features.append(round(len(argList)*0.25)-k)
        k += 1   
    top_n_features = len(argList)
    if type == 'reg':
        rmse_train_initial = mean_squared_error(y_train, model_fs.predict(x_train))
        rmse_test_initial = mean_squared_error(y_test, model_fs.predict(x_test))
        print(f'Initial RMSE Train: {rmse_train_initial:.3f}')
        print(f'Initial RMSE Test: {rmse_test_initial:.3f}')
        top_metric = rmse_test_initial
        for top_feature in num_top_features:
            model_fs.fit(x_train[argList[:top_feature]], y_train)
            rmse_train_new = mean_squared_error(y_train, model_fs.predict(x_train[argList[:top_feature]]))
            rmse_test_new = mean_squared_error(y_test, model_fs.predict(x_test[argList[:top_feature]]))
            if rmse_test_new < top_metric:
                top_n_features = top_feature
                min_rmse = rmse_test_new
    else:
        roc_train_initial = roc_auc_score(y_train, model_fs.predict_proba(x_train)[:, 1])
        roc_test_initial = roc_auc_score(y_test, model_fs.predict_proba(x_test)[:, 1])
        print(f'Initial ROC AUC Train: {roc_train_initial:.3f}')
        print(f'Initial ROC AUC Test: {roc_test_initial:.3f}')
        top_metric = roc_test_initial
        for top_feature in num_top_features:
            model_fs.fit(x_train[argList[:top_feature]], y_train)
            roc_train_new = roc_auc_score(y_train, model_fs.predict_proba(x_train[argList[:top_feature]])[:, 1])
            roc_test_new = roc_auc_score(y_test, model_fs.predict_proba(x_test[argList[:top_feature]])[:, 1])
            if roc_test_new > top_metric:
                top_n_features = top_feature
                top_metric = roc_test_new
    print(f'Number of top features: {top_n_features}, Best metric: {top_metric:.3f}')
    return top_n_features, argList[:top_n_features]

    
def objective(trial, model_type, x_train, y_train, type = 'reg'):
    if type == 'reg':
        if model_type == 'LGBM':
            model = lgb.LGBMRegressor()
            param = {
                "objective": "regression",
                # "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "max_depth": trial.suggest_int("max_depth", 1, 10),
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 70),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 10.0),
            }
        model.set_params(**param)
        model.fit(x_train, y_train)
        return -mean_squared_error(y_test, model.predict(x_test), squared = False)
    else:
        if model_type == 'LGBM':
            model = lgb.LGBMClassifier()
            param = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }
        model.set_params(**param)
        model.fit(x_train, y_train)
        return roc_auc_score(y_train, model.predict_proba(x_train)[:, 1])

def calc_hps(model, x_train, y_train, type = 'reg', trials_num = 100):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model, x_train, y_train, type = type), n_trials=trials_num)
    return study.best_params
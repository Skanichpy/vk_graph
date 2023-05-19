from itertools import combinations
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from collections import Counter
import optuna
import copy
import lightgbm as lgb

def forward_selection_regression(argList, model, x_train, y_train, x_test, y_test, max_add_features_per_iter,
                                 quality_improve, max_features_to_select):
    
    if max_features_to_select > len(argList):
        max_features_to_select = len(argList)
        
    if len(quality_improve) != max_add_features_per_iter:
        quality_improve = [0.4 * i for i in range(1, max_add_features_per_iter + 1)]
        
    forward_argList = []
    
    best_metric = 1e10
    
    for k in range(1, max_add_features_per_iter + 1):
        print(f'Add {k} features per iter')
        
        while True:
            if len(forward_argList) + k > max_features_to_select:
                return forward_argList
            
            best_diff = 1e10
            for features in list(combinations(set(argList) - set(forward_argList), k)):
                model.fit(x_train[forward_argList + list(features)], y_train)
                preds_test = model.predict(x_test[forward_argList + list(features)])
                metric = mean_squared_error(y_test, preds_test)
                
                if metric / best_metric <= 1 - quality_improve[k-1]/100:
                    if metric/best_metric < best_dif:
                        best_dif = metric/best_metric
                        feature_to_add = list(features)
                        
            if (best_dif == 1e10):
                break
            else:
                best_metric *= best_dif
                forward_argList += feature_to_add
                print(f'Фичей в наборе: {len(forward_argList)}, MAE = {best_metric}')
    return forward_argList
    


def forward_selection(argList, model, x_train, y_train, x_test, y_test, max_add_features_per_iter, quality_improve,
                      max_features_to_select, n_features_to_select = -1):
    
    if max_features_to_select > len(argList):
        max_features_to_select = len(argList)
        
    if len(quality_improve) != max_add_features_per_iter:
        quality_improve = [0.4 * i for i in range(1, max_add_features_per_iter + 1)]
        
    forward_argList = []
    
    best_roc_auc = 1.e-2
    if n_features_to_select == -1:
        for k in range(1, max_add_features_per_iter + 1):
            print(f'Add {k} features per iter')
            
            while True:
              if len(forward_argList) + k > max_features_to_select:
                  return forward_argList
              
              best_diff = 0
              for features in list(combinations(set(argList) - set(forward_argList), k)):
                  model.fit(x_train[forward_argList + list(features)], y_train)
                  preds_test = model.predict_proba(x_test[forward_argList + list(features)])[:, 1]
                  roc_auc = roc_auc_score(y_test, preds_test)
                  
                  if roc_auc / best_roc_auc >= 1 + quality_improve[k-1]/100:
                      if roc_auc / best_roc_auc > best_diff:
                          best_diff = roc_auc / best_roc_auc
                          feature_to_add = list(features)
                          
              if (best_diff == 0):
                  break
              else:
                  best_roc_auc *= best_diff
                  forward_argList += feature_to_add
                  print(f'Фичей в наборе: {len(forward_argList)}, ROC AUC = {best_roc_auc}')
        return forward_argList
    else:
        print(f'Adding 1 feature per iter while n_features != {n_features_to_select}')
        
        while len(forward_argList) < n_features_to_select:
            best_roc = 0
            for feature in list(set(argList) - set(forward_argList)):
                model.fit(x_train[forward_argList + [feature]], y_train)
                probs_test = model.predict_proba(x_test[forward_argList + [feature]])[:, 1]
                roc_auc = roc_auc_score(y_test, probs_test)
                
                if roc_auc > best_roc:
                    best_roc = roc_auc
                    feature_to_add = feature
                    
            forward_argList.append(feature_to_add)
            print(f'Фичей в наборе: {len(forward_argList)}, ROC AUC = {best_roc}')
        return forward_argList

def backward_selection_regression(argList, model, x_train, y_train, x_test, y_test, 
                                  quality_loss = 0.5, n_features_to_select = -1):
    backward_argList = argList.copy()
    
    model.fit(x_train[backward_argList], y_train)
    probs_train = model.predict(x_test[backward_argList])
    metric = mean_squared_error(y_test, probs_test)
    print('RMSE Test initial = ', metric)
    
    if n_features_to_select == -1:
        while True:
            best_metric_dif = 1e10
            
            for i in backward_argList:
                model.fit(x_train[backward_argList].drop(i, axis = 1), y_train)
                probs_test = model.predict(x_test[backward_argList].drop(i, axis = 1))
                metric_new = mean_squared_error(y_test, probs_test)
                
                if (metric_new/metric <= 1 + quality_loss/100):
                    if metric_new/metric < best_metric_dif:
                        best_metric_dif = metric_new/metric
                        perem_del = i
                        
            if best_metric_dif == 1e10:
                break
            else:
                backward_argList.remove(perem_del)
                print(f'Осталось фичей: {len(backward_argList)}, MAE = {best_metric_dif*metric}, удалена фича {perem_del}')
        return backward_argList
    else:
        while len(backward_argList) > n_features_to_select:
            best_metric = 1e10
            
            for i in backward_argList:
                model.fit(x_train[backward_argList].drop(i, axis = 1), y_train)
                probs_test = model.predict(x_test[backward_argList].drop(i, axis = 1))[:, 1]
                metric_new = mean_squared_error(y_test, probs_test)
                
                if metric_new < best_metric:
                    best_metric = metric_new
                    perem_del = i
            backward_argList.remove(perem_del)
            
            model.fit(x_train[backward_argList], y_train)
            probs_test = model.predict(x_test[backward_argList])
            metric_new = mean_squared_error(y_test, probs_test)
            print(f'Осталось фичей: {len(backward_argList)}, MAE = {metric_new}, удалена фича {perem_del}')
        return backward_argList
    
def backward_selection(argList, model, x_train, y_train, x_test, y_test, quality_loss = 0.5, n_features_to_select = -1):
    backward_argList = argList.copy()
    model.fit(x_train[backward_argList], y_train)
    probs_test = model.predict_proba(x_test[backward_argList])[:, 1]
    roc = roc_auc_score(y_test, probs_test)
    print('ROC AUC Test initial =', roc)
    
    if n_features_to_select == -1:
        while True:
            best_roc_dif = 0
            if len(backward_argList) > 1:
                for i in backward_argList:
                    model.fit(x_train[backward_argList].drop(i, axis = 1), y_train)
                    probs_test = model.predict_proba(x_test[backward_argList].drop(i, axis = 1))[:, 1]
                    roc_new = roc_auc_score(y_test, probs_test)
                    
                    if (roc_new/roc >= 1-quality_loss/100):
                        if (roc_new/roc > best_roc_dif):
                            best_roc_dif = roc_new/roc
                            perem_del = i
                            
                if (best_roc_dif == 0):
                    break
                else:
                    backward_argList.remove(perem_del)
                    print(f'Осталось фичей: {len(backward_argList)}, ROC AUC = {best_roc_dif*roc}, удалена фича {perem_del}')
            else:
                break
                
        return backward_argList
    else:
        while len(backward_argList) > n_features_to_select:
            best_roc = 0
            
            for i in backward_argList:
                model.fit(x_train[backward_argList].drop(i, axis = 1), y_train)
                probs_test = model.predict_proba(x_test[backward_argList].drop(i, axis = 1))[:, 1]
                roc_new = roc_auc_score(y_test, probs_test)
                
                if roc_new > best_roc:
                    best_roc = roc_new
                    perem_del = i
            backward_argList.remove(perem_del)
            model.fit(x_train[backward_argList], y_train)
            probs_test = model.predict_proba(x_test[backward_argList])[:, 1]
            roc_new = roc_auc_score(y_test, probs_test)
            print(f'Осталось фичей: {len(backward_argList)}, ROC AUC = {best_roc}, удалена фича {perem_del}')
        return backward_argList


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
        return -mean_squared_error(y_train, model.predict(x_train))
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

def calc_hps(model, x_train, y_train, type = 'reg'):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model, x_train, y_train, type = type), n_trials=100)
    return study.best_params

        







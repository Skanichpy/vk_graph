import pandas as pd 
import pickle 
import json 
import datetime 

from typing import List
import os

class Evaluator: 
    """evaluation and submition of model for binary classification"""

    def __init__(self, model, model_name: str,
                 model_parameters=None): 
        """model: (sklearn or not) must have method predict_proba(X)"""
        self.model = model 
        self.model_name = model_name
        if model_parameters is not None:
            self.model_parameters = model_parameters
        else:
            try:
                self.model_parameters = self.model.get_params() 
            except: 
                self.model_parameters = None
        

    def make_submition(self, X_train, X_test): 
        """X_train.index must be used for merging"""
        train_proba, test_proba = self.get_train_test_proba(X_train, X_test)

        train_proba_df = pd.DataFrame({'proba': train_proba})
        train_proba_df.index = X_train.index

        test_proba_df = pd.DataFrame({'proba': test_proba})
        test_proba_df.index = X_test.index 

        return train_proba_df, test_proba_df

    
    def get_train_test_proba(self, X_train, X_test): 
        """get positive proba"""
        positive_train_proba = self.model.predict_proba(X_train)[:, 1]
        positive_test_proba = self.model.predict_proba(X_test)[:, 1]
        return positive_train_proba, positive_test_proba
    

    def eval_metrics(self, X_train, y_train, 
                     metric_func_set:List[callable], 
                     X_test, y_test): 
        metrics_json = [] 
        for metric_func in metric_func_set: 
            metric_parameters = metric_func.__code__.co_varnames

            if ('y_score' in metric_parameters) and hasattr(self.model, 'predict_proba'):
                train_proba_df, test_proba_df = self.make_submition(X_train, X_test)

                train_metric_item, test_metric_item = self.eval_proba(train_proba_df.proba, test_proba_df.proba,
                                                                      y_train, y_test, metric_func)

            elif ('y_pred' in metric_parameters) and hasattr(self.model, 'predict'): 
                test_pred = self.model.predict(X_test)
                train_pred = self.model.predict(X_train)

                train_metric_item, test_metric_item = self.eval_prediction(train_pred, test_pred,
                                                                           y_train, y_test, metric_func)

            else: 
                raise AttributeError(f'{self.model} must have method .predict_proba or .predict')

            metrics_json.append({'metric': metric_func.__name__, 
                                 'train_score': train_metric_item,
                                 'test_score': test_metric_item})
        return pd.DataFrame.from_records(metrics_json)
    

    def write_model(self, dir_load: str) -> None: 
        """save model in .parquet file"""
        if f'{self.model_name}' not in os.listdir(dir_load):
            os.mkdir(f'{dir_load}/{self.model_name}')

        with open(f'{dir_load}/{self.model_name}/{self.model_name}.pkl', 'wb') as fp: 
            pickle.dump(self.model, fp)
    
    def write_submitions(self, X_train, X_test,
                        dir_load: str, **kwargs) -> None:
        """save submition and model info"""
        if f'{self.model_name}' not in os.listdir(dir_load):
            os.mkdir(f'{dir_load}/{self.model_name}')

        info_dict = {'model_name': self.model_name,
                     'datetime': str(datetime.datetime.now()),
                     'params': self.model_parameters,
                     **kwargs}
        
        with open(f'{dir_load}/{self.model_name}/model_parameters.json', 'w') as fp:
            json.dump(info_dict, fp, indent='\t')

        train_proba_df, test_proba_df = self.make_submition(X_train, X_test)
        train_proba_df.to_csv(f'{dir_load}/{self.model_name}/train_submition.csv')
        test_proba_df.to_csv(f'{dir_load}/{self.model_name}/test_submition.csv')

 
    def write_metrics(self, X_train, y_train,  
                      X_test, y_test,
                      metric_func_set:List[callable],
                      dir_load:str) -> None:
         
        if f'{self.model_name}' not in os.listdir(dir_load):
            os.mkdir(f'{dir_load}/{self.model_name}')  

        metrics = self.eval_metrics(X_train, y_train, metric_func_set, X_test, y_test)
        metrics.to_csv(f'{dir_load}/{self.model_name}/metrics.csv', index=False)

    def write_all(self, X_train, X_test, 
                  y_train, y_test,
                  dir_load: str, 
                  metric_func_set:List[callable],
                  log:bool=True,
                  **kwargs) -> None: 
        """run all write methods"""
        self.write_model(dir_load)
        print('Model saved...') if log else None
        self.write_submitions(X_train, X_test, dir_load, **kwargs)
        print('Submitions saved...') if log else None
        self.write_metrics(X_train, y_train, X_test, y_test, 
                           metric_func_set, dir_load)
        print('Metrics saved...') if log else None

    @staticmethod
    def eval_proba(y_train_proba, y_test_proba, 
                   y_train, y_test, metric_func:callable): 
        return metric_func(y_train, y_train_proba), metric_func(y_test, y_test_proba)
    
    @staticmethod 
    def eval_prediction(y_train_pred, y_test_pred,
                        y_train, y_test, metric_func:callable): 
        return metric_func(y_train, y_train_pred), metric_func(y_test, y_test_pred)
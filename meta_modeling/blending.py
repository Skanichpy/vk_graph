from sklearn.linear_model import LogisticRegression 
import pandas as pd

from pathlib import Path 
import os 
from tqdm import tqdm

class Blender: 
    def __init__(self, 
                 dir_load: str,
                 model=LogisticRegression(),
                 log=True) -> None:
    
        self.model = model 
        self.train_subm, self.test_subm = self.get_submitions(dir_load, log) 

    def fit(self, X_train: pd.DataFrame, 
            y_train:pd.DataFrame) -> None: 
        assert hasattr(self.model, "fit")
        self.model.fit(X_train, y_train)
    
    @staticmethod
    def get_submitions(dir_load:str, log=True): 
        model_names = os.listdir(dir_load)
        iterat = tqdm(enumerate(model_names)) if log else enumerate(model_names)
        for idx, model_name in iterat: 
            test_path_subm, train_path_subm = Path(f'{dir_load}/{model_name}').glob("*_submition.csv")
            if idx == 0: 
                train_data = pd.read_csv(train_path_subm, index_col=0).rename({'proba': f'score_{model_name}'}, 
                                                                              axis=1)
                test_data = pd.read_csv(test_path_subm, index_col=0).rename({'proba': f'score_{model_name}'}, 
                                                                              axis=1)

            else: 
                train_data = train_data.merge(pd.read_csv(train_path_subm, 
                                                          index_col=0).rename({'proba': f'score_{model_name}'}, 
                                                                               axis=1)[f'score_{model_name}'],
                                              left_index=True,
                                              right_index=True,
                                              how='left')
                
                test_data = test_data.merge(pd.read_csv(test_path_subm,
                                                        index_col=0).rename({'proba': f'score_{model_name}'},
                                                                             axis=1)[f'score_{model_name}'],
                                              left_index=True,
                                              right_index=True,
                                              how='left')
        return train_data, test_data
    
from sklearn.model_selection import StratifiedKFold
import pandas as pd

class KFoldsIterator: 
    def __init__(self, n_splits:int, 
                 X_train:pd.DataFrame,
                 y_train:pd.DataFrame,
                 shuffle:bool=False,
                 random_state:int=None,
                 ) -> None: 
        
        self.splitter = StratifiedKFold(n_splits=n_splits, 
                                        shuffle=shuffle,
                                        random_state=random_state)
        self.X_train = X_train 
        self.y_train = y_train 

        self.iterator = self.splitter.split(X_train, y_train)
        
    def __next__(self):
        current_train_idx, current_test_idx = next(self.iterator) 
        return  [(self.X_train.iloc[current_train_idx], self.X_train.iloc[current_test_idx]), 
                 (self.y_train.iloc[current_train_idx], self.y_train.iloc[current_test_idx])]

    def __iter__(self): 
        return self
    
    def __len__(self): 
        return self.splitter.get_n_splits()
    
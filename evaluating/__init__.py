class Evaluator: 
    def __init__(self, model): 
        """model: (sklearn or not) must have method predict_proba(X)"""
        self.model = model 

    def make_submition(self, X_train, X_test, y_train, y_test): 
        
        positive_train_proba = self.model.predict_proba(X_train)[:, 1]
        positive_test_proba = self.model.predict_proba(X_test)[:, 1]




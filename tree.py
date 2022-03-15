import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin 

class Tree(BaseEstimator,RegressorMixin):
    def __init__(self, Node_class, max_depth=None, min_samples_split=2, min_impurity_decrease=0.0) -> None:
        self.Node_class = Node_class
        self.node = None
        self.X = None
        self.y = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        
        
    def fit(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.build_tree()
    
    def build_tree(self):
        self.node = self.Node_class(self.X, self.y, X_val = self.X_val, y_val = self.y_val, depth = 0,
                                    max_depth = self.max_depth,min_samples_split = self.min_samples_split,
                                    min_impurity_decrease = self.min_impurity_decrease)
        self.node.grow_tree()
    
    def predict(self, X, include_val = False, max_depth = None):
        return self.node.predict(X, include_val, max_depth)
        
    
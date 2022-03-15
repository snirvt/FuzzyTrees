
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("error")

class RegressionNode():
    def __init__(self, X, y, X_val=None, y_val = None, depth=0, max_depth=0, min_samples_split=2, min_impurity_decrease=0.0 ) -> None:
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.contains_val = self.X_val is not None and self.y_val is not None
        self.depth = depth
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.pred = None
        self.pred_all = None
        self.best_feature = None
        self.best_value = None
        self.mse = mean_squared_error(y, [self.predict_mean()]*len(y))
        if self.contains_val:
            self.mse_val = mean_squared_error(y_val, [self.predict_mean()]*len(y_val))

        self.left = None
        self.right = None

    @staticmethod
    def ma(x: np.array, window: int) -> np.array:
        """
        Calculates the moving average of the given list. 
        """
        return np.convolve(x, np.ones(window), 'valid') / window
    
    def get_df(self, X, y):
        df = pd.DataFrame(X)
        df['Y'] = y
        return df
        
    def get_split_error(self, X, feature, value, left_val = None, right_val = None):
        left_y = X[X[feature]<value]['Y'].values # Getting the left and right ys
        right_y = X[X[feature]>=value]['Y'].values
        if len(left_y) == 0 or len(right_y) == 0:
            return float('inf'), None, None
        
        if left_val is None and right_val is None: # non validation
            mean_left_y = np.mean(left_y)
            mean_right_y = np.mean(right_y)
        else:
            mean_left_y, mean_right_y = left_val, right_val # validation prediction
            
        y_true = np.concatenate((left_y, right_y), axis=None)
        y_pred = np.concatenate((np.repeat(mean_left_y, len(left_y)),
                                np.repeat(mean_right_y, len(right_y))),
                                axis=None)
        mse_split = mean_squared_error(y_true, y_pred)
        return mse_split, mean_left_y, mean_right_y
        
    def best_split(self) -> tuple:
        df = self.get_df(self.X.copy(), self.y)
        mse_base = self.mse       
        best_feature, best_value = None, None # Default best feature and split
        best_left_val, best_right_val = None, None
        for feature in list(df)[:-1]:
            Xdf = df.dropna().sort_values(feature)
            unique_vals = Xdf[feature].unique()    # Sorting the values 
            if unique_vals.shape[0] <= 1: # same valued features are ignored 
                continue
            xmeans = self.ma(unique_vals, 2)  # getting the rolling average         
            for value in xmeans:
                mse_split, left_val, right_val  = self.get_split_error(Xdf, feature, value)
                base_impurity_decrease = self.mse - mse_split
                if mse_split < mse_base and base_impurity_decrease >= self.min_impurity_decrease: # Checking if this is the best split so far 
                    best_feature = feature
                    best_value = value 
                    mse_base = mse_split          # Setting the best gain to the current one 
                    best_left_val, best_right_val = left_val, right_val
        if self.contains_val and best_feature is not None:
            df_val = self.get_df(self.X_val.copy(), self.y_val)
            mse_split_val, _, _ = self.get_split_error(df_val, best_feature, best_value, best_left_val, best_right_val)
            base_impurity_decrease_val = self.mse_val - mse_split_val
            if base_impurity_decrease_val < 0:
                best_feature, best_value = None, None
        return (best_feature, best_value)

    def grow_tree(self):
        """
        Recursive method to create the decision tree
        """
        df = self.get_df(self.X.copy(), self.y)
        if (self.depth < self.max_depth) and (len(self.y) >= self.min_samples_split):   # Splitting further 
            
            best_feature, best_value = self.best_split() # Getting the best split
            if best_feature is not None:
                self.best_feature = best_feature # Saving the best split to the current node 
                self.best_value = best_value
                left_df, right_df = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy() # Getting the left and right nodes
                left_X_val, left_y_val, right_X_val, right_y_val = None, None, None, None
                if self.contains_val:
                    df_val = self.get_df(self.X_val.copy(), self.y_val) 
                    left_df_val, right_df_val = df_val[df_val[best_feature]<=best_value].copy(), df_val[df_val[best_feature]>best_value].copy()
                    left_X_val = left_df_val.drop('Y', axis=1)
                    left_y_val = left_df_val['Y'].values.tolist()
                    right_X_val = right_df_val.drop('Y', axis=1)
                    right_y_val = right_df_val['Y'].values.tolist()
                # Creating the left and right nodes
                left = RegressionNode(left_df.drop('Y', axis=1), left_df['Y'].values.tolist(), X_val =  left_X_val, 
                            y_val = left_y_val, depth=self.depth + 1, max_depth=self.max_depth,  
                            min_samples_split=self.min_samples_split
                            )
                self.left = left 
                self.left.grow_tree()

                right = RegressionNode(right_df.drop('Y', axis=1), right_df['Y'].values.tolist(), X_val = right_X_val, 
                             y_val = right_y_val, depth=self.depth + 1, max_depth=self.max_depth,
                             min_samples_split=self.min_samples_split
                            )
                self.right = right
                self.right.grow_tree()
        
    def predict_mean(self, include_val = False):
        if include_val:
            if self.pred_all is None:
                self.pred_all = np.mean(self.y+self.y_val)
            return self.pred_all
        if self.pred is None:
            self.pred = np.mean(self.y)
        return self.pred
    
    def predict(self, X, include_val = False, max_depth = None):
        if (self.left is not None and max_depth is None) or (self.left is not None and self.depth <= max_depth):
            left_idx = X[self.best_feature] < self.best_value
            pred_left = self.left.predict(X[left_idx], include_val, max_depth)
            right_idx = X[self.best_feature] >= self.best_value
            pred_right = self.right.predict(X[right_idx], include_val, max_depth)
        else:
            return np.repeat(self.predict_mean(include_val), X.shape[0])
        res = np.zeros(X.shape[0])
        res[left_idx] = pred_left
        res[right_idx] = pred_right
        return res

    
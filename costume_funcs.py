import numpy as np
from copy import deepcopy
from gplearn.functions import make_function
from gplearn.fitness import make_fitness

from sklearn.metrics import r2_score, mean_squared_error


def _exp(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 30, np.exp(x1), 0.)

def exp():
    func = make_function(function=_exp,
                            name='exp',
                            arity=1)
    return func


def _mse(y, y_pred, w):
    return mean_squared_error(y, y_pred, sample_weight=w)

def mse():
    mse = make_fitness(_mse, greater_is_better=False)
    return mse

def _mse_split_mean(X_y, splitting_cond, w): # gp.fit(X, X_y)
    left_y = X_y[splitting_cond<0]['Y'].values 
    right_y = X_y[splitting_cond>=0]['Y'].values
    if len(left_y) == 0 or len(right_y) == 0:
        return float('inf')
    mean_left_y = np.mean(left_y)
    mean_right_y = np.mean(right_y)
    y_true = np.concatenate((left_y, right_y), axis=None)
    y_pred = np.concatenate((np.repeat(mean_left_y, len(left_y)),
                            np.repeat(mean_right_y, len(right_y))),
                            axis=None)
    mse_split = mean_squared_error(y_true, y_pred)
    return mse_split

def mse_split_mean():
    mse_split_mean = make_fitness(_mse_split_mean, greater_is_better=False)
    return mse_split_mean






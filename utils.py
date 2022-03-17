import numpy as np

from sklearn.model_selection import train_test_split


def split_data(X,y, train_size=0.6, val_size = 0.2, test_size = 0.2, random_state = None):
    if train_size + val_size + test_size != 1:
        raise Exception('Not summing to zero') 
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=test_size/(test_size+train_size), random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test



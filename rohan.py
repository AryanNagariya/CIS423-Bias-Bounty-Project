import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import *
import dill as pkl

def get_g(X):
    """
    Returns a function that takes in a dataframe and returns a series defining the group.
    """

    def g(X):
        return X['RAC1P'] == 1

    return g

def get_h(X, y):
    """
    Returns a function that takes in a dataframe and returns a series of classification predictions.
    """

    # get indices of chosen group
    g = get_g(X)
    indices = g(X)

    # train classifier on group
    clf = sk.tree.DecisionTreeRegressor(max_depth = 5, random_state = 42)
    clf.fit(X[indices], y[indices])

    return clf.predict
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from joblib import dump, load

from sklearn.model_selection import GridSearchCV
import warnings
import os

import platform
if platform.system() == "Windows":
    path = "pre/"
    saveLoad = "src/"
else:
    path = "../pre/"
    saveLoad = "../src/"

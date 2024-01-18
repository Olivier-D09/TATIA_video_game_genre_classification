import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump, load

from sklearn.model_selection import GridSearchCV
import warnings
import os

import platform
if platform.system() == "Windows":
    path = "preprocessing/"
    saveLoad = "src/"
else:
    path = "../pre/"
    saveLoad = "../src/"


data = pd.read_csv(path + "games_clean.csv")

print(data.head(5))

X = data['Summary']

tr = TfidfVectorizer()
pre = ColumnTransformer(
    transformers=[
        ('summary', tr, 'Summary')
    ]
)

print(pre)

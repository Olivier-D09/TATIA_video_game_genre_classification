import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score

import platform

if platform.system() == "Windows":
    path = "preprocessing/"
    saveLoad = "src/"
else:
    path = "../pre/"
    saveLoad = "../src/"


# Load the dataset
data = pd.read_csv(path +'genres_summary_tokenized.csv')
data = data.dropna()

X = TfidfVectorizer().fit_transform(data['Tokenized'])
y = data['Genre']

# Assuming you have your feature matrix X and target variable y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Optimize HyperParameters
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [1,2],
    'min_samples_leaf': [1, 2, 4]
}


# grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, verbose=2, n_jobs=-1)
# grid.fit(X_train, y_train)
# best_params = grid.best_params_
# print("Best Parameter: ", best_params,"\n")

# Create a RandomForestClassifier
# rf_classifier = RandomForestClassifier(random_state=42,**best_params)

rf_classifier = RandomForestClassifier(random_state=42,min_samples_leaf=4,min_samples_split=2,n_estimators=200,max_depth=10)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#plot the accuracy, macro precision, macro recall and macro f1 score
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


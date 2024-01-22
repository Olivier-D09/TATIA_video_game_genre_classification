import platform
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

if platform.system() == "Windows":
    path = "preprocessing/"
    saveLoad = "src/"
else:
    path = "../pre/"
    saveLoad = "../src/"

Data = pd.read_csv(path + "genres_summary_tokenized.csv")

#create SVC classifier with Data 
X = Data['Tokenized']
y = Data['Genre']

y.dropna(inplace=True)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)
clf = SVC(kernel='linear', gamma='auto') 
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#display graphic of the SVC classifier
plt.matshow(confusion_matrix(y_test, y_pred))

plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('Summary')
plt.savefig("images/" + 'SVC_Classifier.png')
plt.show()


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred, average='macro'))
print("Recall:",metrics.recall_score(y_test, y_pred, average='macro'))


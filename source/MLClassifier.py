import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

import platform

if platform.system() == "Windows":
    path = "preprocessing/"
    saveLoad = "src/"
else:
    path = "../pre/"
    saveLoad = "../src/"

# data = pd.read_csv(path + "games_clean.csv")
# result_df = pd.DataFrame(columns=['Title','Genre','Summary'])

## Tokenization
    
newData = pd.read_csv(path + "genres_summary.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newData['Tokenized'])
y = newData['Genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
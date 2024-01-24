import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

import platform

import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

if platform.system() == "Windows":
    path = "preprocessing/"
    saveLoad = "src/"
else:
    path = "../pre/"
    saveLoad = "../src/"

# data = pd.read_csv(path + "games_clean.csv")
# result_df = pd.DataFrame(columns=['Title','Genre','Summary'])

## Tokenization
    
newData = pd.read_csv(path + "genres_summary_tokenized.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newData['Tokenized'])
y = newData['Genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

prediction_train = classifier.predict(X_train)
print("Train prediction")
print(classification_report(y_train, prediction_train))

predictions = classifier.predict(X_test)
print("Test prediction")
print(classification_report(y_test, predictions))

print("with lemming")

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
def tokenstem(text):
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(w) for w in tokens if not w in stop_words]
    return tokens

vectorizer2 = TfidfVectorizer(tokenizer=tokenstem,token_pattern=None)

X = newData.drop(columns=['Genre','Tokenized'])

X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=0.2)

X = vectorizer2.fit_transform(X2_train['Summary'])

X_train_vec = vectorizer2.fit_transform(X2_train['Summary'])
X_test_vec = vectorizer2.transform(X2_test['Summary'])

classifier2 = MultinomialNB()
classifier2.fit(X_train_vec, y2_train)

prediction_train2 = classifier2.predict(X_train_vec)
print("Train prediction with lemming")
print(classification_report(y2_train , prediction_train2))

predictions2 = classifier2.predict(X_test_vec)
print("Test prediction with lemming")
print(classification_report(y2_test, predictions2))


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

plot_confusion_matrix(y_test, predictions, classes=classifier.classes_, title="Confusion Matrix")
plot_confusion_matrix(y2_test, predictions2, classes=classifier.classes_, title="Confusion Matrix with lemming")


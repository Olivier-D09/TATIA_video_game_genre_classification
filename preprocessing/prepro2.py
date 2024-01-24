import nltk
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import re
import platform

if platform.system() == "Windows":
    path = "preprocessing/"
else:
    path = ""

token_pattern = re.compile(r"(?u)\b\w\w+\b")
# tokenizer = token_pattern.findall

##############################################
# #preparing data
# data = pd.read_csv(path + "genres_summary.csv")

# lb = LabelBinarizer()
# labels = data['Genre'] #labels
# binarized_labels = lb.fit_transform(labels)

# df = pd.DataFrame(columns=['OneHot'])
# df['OneHot'] = binarized_labels.tolist()

# data = data.join(df)
# data.to_csv(path + "oneHotencode.csv", index=False)

######################################

data = pd.read_csv(path + "oneHotencode.csv")

X = data.drop(columns=['Genre','OneHot','Title'])
y = data['Genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Dimensions de l'ensemble d'entraînement (X_train, y_train):", X_train.shape, y_train.shape)
print("Dimensions de l'ensemble de test (X_test, y_test):", X_test.shape, y_test.shape)

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
def tokenstem(text):
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(w) for w in tokens if not w in stop_words]
    return tokens


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(tokenizer=tokenstem, min_df=2, max_df=0.98,token_pattern=None)

X_train_vec = vectorizer.fit_transform(X_train['Summary'])
# for voc in X_train_vec.toarray():
#   print(voc)

X_test_vec = vectorizer.transform(X_test['Summary'])
# for voc in X_test_vec.toarray():
#   print(voc)

from sklearn.svm import SVC 

svm = SVC()
svm.fit(X_train_vec, y_train)

y_train_pred = svm.predict(X_train_vec)
y_test_pred = svm.predict(X_test_vec)

print("Train :")
print(classification_report(y_train, y_train_pred))

print("Test :")
print(classification_report(y_test, y_test_pred))


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

plot_confusion_matrix(y_test, y_test_pred, classes=svm.classes_, title="Confusion Matrix with lemming")
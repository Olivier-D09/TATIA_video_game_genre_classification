import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

import platform
import nltk

# nltk.download('stopwords')

if platform.system() == "Windows":
    path = "preprocessing/"
    saveLoad = "src/"
else:
    path = "../pre/"
    saveLoad = "../src/"

# data = pd.read_csv(path + "games_clean.csv")
# result_df = pd.DataFrame(columns=['Title','Genre','Summary'])

# def setGenres(listGenres, Summary,Title,df):
#     list = listGenres
#     list = list.replace('[','')
#     list = list.replace(']','')
#     for i in list.split(', '):
#         newLigne = {"Title":Title,"Genre":i,"Summary":Summary}
#         df = pd.concat([df, pd.DataFrame([newLigne])], ignore_index=True)
#     return df

# for index, row in data.iterrows():
#     result_df = setGenres(row['Genres'],row['Summary'],row['Title'],result_df)

# result_df.to_csv(path + "genres_summary.csv", index=False)
    
## Tokenization
    
newData = pd.read_csv(path + "genres_summary.csv")

stop_words = set(stopwords.words('english'))

def tokenizeStopWord(text):
    word_tokens = word_tokenize(text)
    word_tokens = [word.lower() for word in word_tokens if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(word_tokens)


newData['Tokenized'] = newData['Summary'].apply(tokenizeStopWord)
newData = newData.dropna()
newData.to_csv(path + "genres_summary_tokenized.csv", index=False)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(newData['Tokenized'])
y = newData['Genre']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
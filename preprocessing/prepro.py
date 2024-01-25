########################################################Imports###############################################
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd;
import matplotlib.pyplot as plt;
import os;
import platform;
import seaborn as sns;
import nltk;
from sklearn.preprocessing import LabelBinarizer


nltk.download('stopwords')

########################################################Functions###############################################

#define path for import & export files
def get_path():
    if platform.system() == "Windows":
        path = "preprocessing/"
    else:
        path = ""
    return path

#Show corroleation matrix
def show_correlation_matrix(df):
    plt.figure(figsize=(10,10))
    corr = df.corr()
    sns.heatmap(corr, annot=True)
    plt.show()

#give a list of all gernres and a list of all genres with a number
def get_genres(line):
    acc1 = []
    acc2 = []
    count = 1
    for i in line:
        for j in i.split(', '):
            for k in j:
                #keep only characters
                if not k.isalpha():
                    j = j.replace(k,'')
            #remove empty elements
            if j != '':
                #remove duplicates
                if j not in acc1:
                    acc1.append(j)
                    temp = [j,count,0]
                    acc2.append(temp)
                    count += 1
    return acc1,acc2


#return a list of genres for a row on dataframe
def genre_by_row(line):
    acc = []
    for j in line.split(', '):
            for k in j:
                #keep only characters
                if not k.isalpha():
                    j = j.replace(k,'')
            #remove empty elements
            if j != '':
                #remove duplicates
                if j not in acc:
                    acc.append(j)
    return acc


#convert non ascii character to ascii
def convert_to_ascii(text):
    res = ""
    for x in text:
        if x.isascii():
            res += x
        else:
            if x == 'é' or x == 'è' or x == 'ê':
                res += 'e'
            elif x == 'à' or x == 'â':
                res += 'a'
            elif x == 'ù' or x == 'û':
                res += 'u'
            elif x == 'ô':
                res += 'o'
            elif x == 'î' or x == 'ï':
                res += 'i'
            elif x == 'ç':
                res += 'c'
            else:
                res += ''                
    return res

#remove recursively - from text 
def flat_text(text):
    
    tempo = text.split(' ')
    res = []

    for x in tempo:
        if x.isascii() :
            if x.find('-') != -1:
                tempo2 = x.split('-')
                for y in tempo2:
                    flat_text(y)
            else:
                res.append(x)
        else:
            res.append(convert_to_ascii(x))
    return res
    
#remove recursively genre from summary
def delete_genre(resume,genre):
    if genre in resume:
        resume = resume.replace(genre,'')
        if genre in resume:
            delete_genre(resume,genre)
    return resume

#remove tabulation, new line and empty element from summary
def remove_format(tmp):
    split_correct = flat_text(tmp)
    res = ' '
    for elem in split_correct:
        if elem != '' and elem.find('\n') == -1 and elem.find('\r') == -1:
            res += elem + ' '
    return res

#remove & translate Genres in summary
def remove_genres(numeroted_genres,tmp,i):
    change_genre = []
    gen = genre_by_row(dataset['Genres'][i])
    for j in gen:
        for k in numeroted_genres:
            if j == k[0]:
                change_genre.append(k[1])
        j = str(j).lower()
        tmp = tmp.lower()
        tmp = delete_genre(tmp,j)
    return change_genre,tmp

#remove subchain of title from summary
def subchain_remove(title,tmp):
    for x in title:
            if x.find(":") != -1 or x.isdigit():
                splited_title = title.split(x)
                for k in range(len(splited_title)):
                    if splited_title[k] !="" and not splited_title[k].isdigit():
                        if splited_title[k] in tmp:
                            tmp = tmp.replace(splited_title[k],'video game ')
                    if splited_title[k].isdigit() and k == len(splited_title):
                        return 1


#work on the dataset to remove title, tabulation...  & genre from summary
def normalize(dataset):
    for i in range(len(dataset)):

        tmp = str(dataset['Summary'][i])
        title = str(dataset['Title'][i])

        #remove title & subchain from summary
        if dataset['Title'][i] in dataset['Summary'][i]:
            tmp = tmp.replace(dataset['Title'][i],'The game')
        subchain_remove(title,tmp)
        
        translated_genres,tmp =  remove_genres(numeroted_genres,tmp,i)
        tmp = remove_format(tmp)

        data['Genres'][i] = translated_genres
        dataset['Summary'][i] = tmp

#build new dataframe with one genre by row of the dataframe
def one_genre_by_row(data):
    indice = 0
    for elem in range(len(data)+4):
        if elem not in [649,713,1309,1475]:
            listeofgenres = data['Genres'][elem]
            if len(listeofgenres) >= 1:
                for genre in listeofgenres:
                    genres_summary.loc[indice] = [data['Title'][elem], genre, data['Summary'][elem]]
                    indice += 1

def tokenizeStopWord(text):
    word_tokens = word_tokenize(text)
    word_tokens = [word.lower() for word in word_tokens if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(word_tokens)

def keep_min_occur(tableOfGenre,genres):
    min = 100000
    for genre in genres:
        if tableOfGenre[genre-1][2] < min:
            min = tableOfGenre[genre-1][2]
            keep = genre
    tableOfGenre[keep-1][2] += 1
    return tableOfGenre,keep

def choice_genre(tableOfGenre,dataset):
    for i in range(len(dataset)):
        if i not in [649,713,1309,1475]:
            genres = dataset['Genres'][i] 
            if len(genres) == 1:
                genres = int(genres[0])
                temp = tableOfGenre[genres-1]
                temp[2] += 1
            else:
                tableOfGenre,res =keep_min_occur(tableOfGenre,genres)
                dataset['Genres'][i] = res
    return tableOfGenre

########################################################Code###############################################

#remove pandas warning
pd.options.mode.chained_assignment = None

path = get_path()

#Importing the datasets
dataset = pd.read_csv(path + "games.csv")
stop_words = set(stopwords.words('english'))

genres_summary = pd.DataFrame(columns=['Title','Genre','Summary']) #define genres_summary as pandas dataframe

#drop useless columns
data = dataset.drop(['Release Date','Team','Rating','Times Listed','Number of Reviews','Reviews','Plays','Playing','Backlogs','Wishlist '],axis=1)

genres, numeroted_genres = get_genres(data['Genres']) #list of genres & list of genres with a number

#set type of columns to string
data['Summary'] = dataset['Summary'].astype(str)
data['Title'] = dataset['Title'].astype(str)

normalize(data)
data = data.drop([649,713,1309,1475]) #remove rows with empty summary or genre
neoData = data.copy() #copy of data
numeroted_genres = choice_genre(numeroted_genres,neoData) #select genre for each row
one_genre_by_row(data)
data.to_csv(path + 'games_clean.csv',index=False) #dataframe with clean summary 

genres_summary.to_csv(path + "genres_summary.csv", index=False)#dataframe with one genre by row

genres_summary['Tokenized'] = genres_summary['Summary'].apply(tokenizeStopWord)
newData = genres_summary.dropna()

newData.to_csv(path + "genres_summary_tokenized.csv", index=False) #dataframe with one genre by row & tokenized summary

neoData.to_csv(path + "games_V2.csv", index=False) #dataframe with one genre by row

neoData2 = neoData.copy()

lb = LabelBinarizer()
labels = neoData2['Genres'] #labels
binarized_labels = lb.fit_transform(labels)

df = pd.DataFrame(columns=['OneHot'])
df['OneHot'] = binarized_labels.tolist()

neoData2 = neoData2.join(df)
neoData2.to_csv(path + "oneHotencode.csv", index=False)
import pandas as pd;
import matplotlib.pyplot as plt;
import os;
import platform;
import seaborn as sns;

if platform.system() == "Windows":
    path = "preprocessing/"
else:
    path = ""

#Importing the dataset
dataset = pd.read_csv(path + "games.csv");

def get_genres(line):
    acc = []
    for i in line:
        for j in i.split(', '):
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

def g_line(line):
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

#Show corroleation matrix
def show_correlation_matrix(df):
    plt.figure(figsize=(10,10))
    corr = df.corr()
    sns.heatmap(corr, annot=True)
    plt.show()



def normalize_summary(dataset):
    for i in range(len(dataset)):
        tmp = str(dataset['Summary'][i])
        if dataset['Title'][i] in dataset['Summary'][i]:
            #remove title from summary
            tmp = tmp.replace(dataset['Title'][i],'The game')
        #remove Genres in summary
        gen = g_line(dataset['Genres'][i])
        for j in gen:
            j = str(j).lower()
            tmp = tmp.lower()
            if j in tmp:
                tmp = tmp.replace(j,'')
        dataset['Summary'][i] = tmp


data = dataset.drop(['Release Date','Team','Rating',
                     'Times Listed','Number of Reviews','Reviews','Plays',
                     'Playing','Backlogs','Wishlist'],axis=1)

#print(data.columns)
#print(data)

genres = get_genres(data['Genres'])

data['Summary'] = dataset['Summary'].astype(str)
data['Title'] = dataset['Title'].astype(str)

normalize_summary(data)

data.to_csv(path + 'games_clean.csv',index=False)


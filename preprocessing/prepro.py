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
                    temp = [j,count]
                    acc2.append(temp)
                    count += 1
    return acc1,acc2

def genre_by_line(line):
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


unerferenced_char = []

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

                #save x in a file without encoding
                with open('non_ascii.txt','a') as file:
                    file.write(ascii(x) + '\n')
                
    return res


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
    


def normalize_summary(dataset):
    for i in range(len(dataset)):
        tmp = str(dataset['Summary'][i])

        #remove title & subchain from summary
        title = str(dataset['Title'][i])

        if dataset['Title'][i] in dataset['Summary'][i]:
            tmp = tmp.replace(dataset['Title'][i],'The game')

        for x in title:
            if x.find(":") != -1 or x.isdigit():
                splited_title = title.split(x)
                for k in range(len(splited_title)):
                    if splited_title[k] !="" and not splited_title[k].isdigit():
                        if splited_title[k] in tmp:
                            tmp = tmp.replace(splited_title[k],'video game ')
                    if splited_title[k].isdigit() and k == len(splited_title):
                        return 1

        #remove Genres in summary
        gen = genre_by_line(dataset['Genres'][i])
        for j in gen:
            j = str(j).lower()
            tmp = tmp.lower()
            if j in tmp:
                tmp = tmp.replace(j,'')

        res = '' 

        split_correct = flat_text(tmp)

        for elem in split_correct:
            if elem != '' and elem.find('\n') == -1 and elem.find('\r') == -1:
                res += elem + ' '     

        dataset['Summary'][i] = res


def normalize_genre(data):
    for i in range(len(data)):
        tmp = []
        for j in genre_by_line(data['Genres'][i]):
            for k in numeroted_genres:
                if j == k[0]:
                    tmp.append(k[1])
        data['Genres'][i] = tmp

data = dataset.drop(['Release Date','Team','Rating','Times Listed','Number of Reviews','Reviews','Plays','Playing','Backlogs','Wishlist '],axis=1)

#print(data.columns)
#print(data)

genres, numeroted_genres = get_genres(data['Genres'])


data['Summary'] = dataset['Summary'].astype(str)
data['Title'] = dataset['Title'].astype(str)

normalize_summary(data)
normalize_genre(data)




data.to_csv(path + 'games_clean.csv',index=False)


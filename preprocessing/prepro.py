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

#Show corroleation matrix
def show_correlation_matrix(df):
    plt.figure(figsize=(10,10))
    corr = df.corr()
    sns.heatmap(corr, annot=True)
    plt.show()

data = dataset.drop(['Release Date','Team','Rating',
                     'Times Listed','Number of Reviews','Reviews','Plays',
                     'Playing','Backlogs','Wishlist'],axis=1)

print(data.columns)
print(data)
import pandas as pd;
import matplotlib.pyplot as plt;
import os;
import platform;

if platform.system() == "Windows":
    path = "preprocessing/"
else:
    path = ""

#Importing the dataset
dataset = pd.read_csv(path + "games.csv");
print(dataset.head());
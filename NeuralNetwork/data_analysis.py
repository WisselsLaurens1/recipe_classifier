import torch
import torch.nn as nn
import pandas as pd
import sys
sys.path.append('../')
from scripts.tools import *
from sklearn.metrics import accuracy_score
from pathlib import Path
from scripts.tools import _train_test_split
from scripts.tools import data_loader
import hashlib
from torch.utils.data import Dataset, DataLoader
import json
from collections import defaultdict


PATH = '../meta/cuisines.json'

with open(PATH,'r') as f:
    raw = f.read()
    data = json.loads(raw)

ground_thruts = data['cuisines']

df = pd.read_csv('../data/train.csv')

labels = list(df.columns)[1:-1]
X = df[labels].values
Y = df['cuisine'].values
cuisines = list(set(Y))
cuisines.sort()
# Y = [cuisines.index(y) for y in Y]


X_train = X
Y_train = Y

df = pd.read_csv('../data/cleaned_validation.csv')

labels = list(df.columns)[:-1]
X = df[labels].values
Y = df['cuisine'].values
cuisines = list(set(Y))
cuisines.sort()


X_validation = X
Y_validation = Y

def get_class_dict(Y):
    return dict.fromkeys(list(set(Y)))

def get_all_indexes(x,labels,element):
    indexes = []
    for i in range(len(x)):
        if x[i] == element:
            indexes.append(i)
    return indexes

def ingredients_frequency(X,Y,labels):

    class_dict = get_class_dict(Y)
    for x,y in zip(X,Y):
        indexes = get_all_indexes(x,labels,1)
        ingredients = [labels[i] for i in indexes]

        if class_dict[y] == None:
            class_dict[y] = defaultdict(None)
        for ingredient in ingredients:
            class_dict[y].setdefault(ingredient,0)
            class_dict[y][ingredient] += 1

    return class_dict



def _print(X,Y,labels):

    result = ingredients_frequency(X,Y,labels)

    limit = 40
    for cuisine in result:
        i = 0
        print(f'\n* {cuisine}\n')
        ing = result[cuisine]
        for w in sorted(ing, key=ing.get, reverse=True):
            
            if ing[w] > 5:
                print(w, ing[w])




            i+=1



_print(X_train,Y_train,labels)
print('#'*30)
_print(X_validation,Y_validation,labels)
        



from numpy import result_type
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus
import math
import json
from tools import *
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier


print('started')
df = pd.read_csv('../data/train.csv')


X = df.values
X = [x[:-1] for x in X]  # remove cuisine
Y = list(df['cuisine'])

features = list(df.columns)
features = [feature for feature in features if feature != 'cuisine']


train_ratios = class_ratio(Y)
n = len(train_ratios)

min_samples_leaf = list(range(100, 600, 100))
min_impurity_decrease = [0]+[1/math.pow(10, i) for i in range(1, 6)]
n_estimators = [i for i in range((n*2)+10, 550, 50)]
max_features = ['auto', 'log2', 'sqrt', None]
cv = 5


print(min_samples_leaf)
print(min_impurity_decrease)
print(n_estimators)
print(max_features)


tot_perm = len(min_samples_leaf)*len(min_impurity_decrease) * \
    len(n_estimators)*len(max_features)*cv
print(tot_perm)

parameters = {
    'min_samples_leaf': list(range(1, 600, 100)),
    'min_impurity_decrease': [1/math.pow(10, i) for i in range(1, 6)]+[0],
    'n_estimators': [i for i in range(n*2, 500, 50)],
    'max_features': ['auto', 'log2', 'sqrt', None]
}


clf = RandomForestClassifier(n_jobs=27)
gscv = GridSearchCV(clf, parameters, cv=cv, n_jobs=n-1)
gscv.fit(X, Y)

best = gscv.best_params_
score = gscv.best_score_
print(best)
print(score)
with open('../dt_results', 'a')as f:
    f.write(parameters)
    f.write(best)
    f.write(score)

print('finished')

import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
import copy
import sys
sys.path.append('../')
from scripts.tools import *
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt  
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def predict(X,class_pred_rf,invalid_rf):

    #predict which class 
    class_predictions = class_pred_rf.predict(X)
    
    #predict if class is a invalid class based on prob distributions
    prob_predicitions = class_pred_rf.predict_proba(X)

    #prepare inputs for invalid_rf 
    #inputs [class probabilities, number of zeros, max prob]
    _input = []
    for prob in prob_predicitions:
        tmp = []
        prob = list(prob)
        tmp.extend(prob)
        tmp.append(prob.count(0))
        tmp.append(max(prob))
        _input.append(tmp)


    invalid_predictions = invalid_rf.predict(_input)

    pred_classes = []
    for inv,_class in zip(invalid_predictions,class_predictions):
        if inv == 'invalid':
            pred_classes.append('invalid')
        else:
            pred_classes.append(_class)
    
    return pred_classes

def main():
    clf = pickle.load(open('../models/class_pred_rf.sav', 'rb'))
    class_pred_rf = copy.deepcopy(clf)
    invalid_rf = pickle.load(open('../models/invalid_pred_rf.sav', 'rb'))




    clf.predict = predict

    INPUT_PATH = '../data/train.csv'
    df = pd.read_csv(INPUT_PATH)

    X = df.values
    X = [x[:-1] for x in X] # remove cuisine
    Y = list(df['cuisine'])
    Y = [y.lower() for y in Y]


    _, X_test, _, Y_test = train_test_split(X, Y, test_size=0.05, random_state=0)


    scores = cross_val_score(clf, X, Y, cv=5,n_jobs=-2)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    Y_pred = clf.predict(X,class_pred_rf=class_pred_rf,invalid_rf=invalid_rf)
    accuracy = accuracy_score(Y, Y_pred, normalize=True)
    print("Accuracy:",accuracy)

    INPUT_PATH = '../data/cleaned_validation.csv'

    df = pd.read_csv(INPUT_PATH)

    print(len(df.columns))

    X = df.values
    print(f'n samples: {len(X)}')
    X = [x[:-1] for x in X] # remove cuisine
    Y = list(df['cuisine'])
    Y = [y.lower() for y in Y]


if __name__ == '__main__':
    print('starting')
    main()
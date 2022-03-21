from os import write
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import csv


def main():
    clf = pickle.load(open(('rf.sav'), 'rb'))

    INPUT_PATH = '../data/test.csv'

    df = pd.read_csv(INPUT_PATH)
    X = df.values
    X = [x[:-1] for x in X]  # remove cuisine
    Y = list(df['cuisine'])
    Y = [y.lower() for y in Y]

    data = []
    headers = []

    # headers: prob of all classes, n_zeros, max_score
    headers.extend(clf.classes_)
    headers.append('n_zeros')
    headers.append('max_score')
    headers.append('class')

    prob_pred = pickle.load(open(('prob_pred.sav'), 'rb'))

    for x, y, pred in zip(X, Y, prob_pred):
        sample = []

        pred = list(pred)

        sample.extend(pred)
        sample.append(pred.count(0))
        sample.append(max(pred))

        if y.lower() != 'invalid':
            sample.append('valid')
        else:
            sample.append('invalid')

        data.append(sample)

    with open('../data/prob_train.csv', 'w') as f:
        writer = csv.writer(f)

        writer.writerow(headers)
        writer.writerows(data)


if __name__ == '__main__':
    print('started')
    main()
    print('finished')

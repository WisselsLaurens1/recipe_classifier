from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import pickle
import time


def main():

    basedir = Path(__file__).resolve().parents[2]

    INPUT_PATH = f'{basedir}/data/train.csv'
    # INPUT_PATH = '../data/train.csv'
    # INPUT_PATH = './data/cleaned_test.csv'

    df = pd.read_csv(INPUT_PATH)

    X = df.values
    X = [x[:-1] for x in X]  # remove cuisine
    Y = list(df['cuisine'])
    Y = [y.lower() for y in Y]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.05, random_state=0)

    clf = RandomForestClassifier(
        random_state=0, n_jobs=-1, class_weight='balanced_subsample')
    clf.fit(X_train, Y_train)

    filename = f'{basedir}/models/class_pred_rf.sav'
    pickle.dump(clf, open(filename, 'wb'))


if __name__ == '__main__':
    print('started')
    start = time.time()
    main()
    end = time.time()
    print('finished')
    print(f'Time: {end-start}')

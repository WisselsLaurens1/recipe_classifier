from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parents[2]

    df = pd.read_csv(f'{base_dir}/data/prob_train.csv')
    X = df.values
    X = [x[:-1] for x in X]  # remove cuisine
    Y = list(df['class'])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.05, random_state=0)
    clf = RandomForestClassifier(
        random_state=0, n_jobs=-2)
    clf.fit(X_train, Y_train)

    pickle.dump(clf, open(f'{base_dir}/models/invalid_pred_rf.sav', 'wb'))


if __name__ == '__main__':
    print('started')
    main()
    print('finished')

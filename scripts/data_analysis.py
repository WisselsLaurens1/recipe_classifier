
from pathlib import Path
import pandas as pd

basedir = Path(__file__).resolve().parents[1]

def main():
    
    df_train = pd.read_csv(f'{basedir}/data/train.csv')


    
    Y_train = df_train['cuisine'].values
    df = df_train.drop(['cuisine','index'],axis=1)
    labels = list(df.columns)
    X_train = df[labels].values
    cuisines = list(set(Y_train))
    cuisines.sort()
    Y_train = [cuisines.index(y) for y in Y_train]

    print(len(X_train[0]))


    df_test = pd.read_csv(f'{basedir}/data/cleaned_validation.csv')


    Y_test = df_test['cuisine'].values
    df = df_test.drop(['cuisine'],axis=1)
    labels = list(df.columns)
    X_test = df[labels].values
    cuisines = list(set(Y_test))
    cuisines.sort()
    Y_test = [cuisines.index(y) for y in Y_test]
    
    print(len(X_test[0]))



if __name__ == '__main__':
    main()

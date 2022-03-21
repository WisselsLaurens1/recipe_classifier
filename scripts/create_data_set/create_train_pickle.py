import sys
sys.path.append('../')
import pandas as pd
import pickle
import copy
from pathlib import Path

def main():

    base_dir = Path(__file__).resolve().parents[2]

    INPUT_PATH = f'{base_dir}/data/train.csv'
    df = pd.read_csv(INPUT_PATH)

    pickle.dump(df, open(f'{base_dir}/data/train_pickle.sav', 'wb'))

if __name__ == '__main__':
    main()

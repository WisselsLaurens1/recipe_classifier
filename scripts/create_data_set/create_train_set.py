import sys
sys.path.append('../')
from pathlib import Path
import json
from os import name
from preprocessor import Preprocessor
from tools import *
from preprocess import *

basedir = Path(__file__).resolve().parents[2]

INPUT_PATH = f'{basedir}/data/train.json'
OUTPUT_PATH = f'{basedir}/data/train.csv'


def main():

    # read data from dataset from kaggle
    with open(INPUT_PATH, 'r') as f:
        raw = f.read()
        data = json.loads(raw)

    preprocessor = Preprocessor(f'{basedir}/meta/units.json', f'{basedir}/meta/cuisines.json')

    x = [sample['ingredients'] for sample in data]
    x = preprocess_ingredients(x)
    # x = preprocessor.preprocess_ingredients(x)

    y = [sample['cuisine'] for sample in data]
    y = preprocessor.preprocess_cuisines(y)
    # show_class_ratio(y)

    voc = create_voc(x=x)

    train_vectors = vectorize(x=x, voc=voc)

    #add indexes
    voc = ['index'] + voc 
    train_vectors = [[i]+sample for i,sample in enumerate(train_vectors)]

    write(X=train_vectors, Y=y, headers=voc, path=OUTPUT_PATH)


if __name__ == '__main__':
    print('creating data')
    main()
èèè-
from preprocessor import Preprocessor
from tools import *
from preprocess import *
import os
import pickle
import json
from pathlib import Path
import sys
sys.path.append('../')


basedir = Path(__file__).resolve().parents[2]

def main():

    INPUT_PATH = f'{basedir}/data/train.json'

    with open(INPUT_PATH, 'r') as f:
        raw = f.read()
        data = json.loads(raw)

    preprocessor = Preprocessor(
        f'{basedir}/meta/units.json', f'{basedir}/meta/cuisines.json')

    print("preprocessing train")
    X = [sample['ingredients'] for sample in data]
    X = preprocess_ingredients(X)

    Y = [sample['cuisine'] for sample in data]
    Y = preprocessor.preprocess_cuisines(Y)

    voc = create_voc(x=X)

    vectors = vectorize(x=X, voc=voc)

    data = {"ingredients": X, "vectors": vectors, "cuisines": Y}

    file = open(f'{basedir}/data/train.sav', "wb")
    pickle.dump(data, file)
    file.close()
    print("finished preprocessing train")

    INPUT_PATH = f'{basedir}/data/test/classified'

    X = []
    Y = []

    for file in os.listdir(INPUT_PATH):
        with open(f'{INPUT_PATH}/{file}', 'r') as f:
            raw = f.read()
            data = json.loads(raw)
            key = list(data.keys())[0]
            data = data[key]
            X.append(data['ingredients'])
            Y.append(data['cuisine'])

    print("preprocessing test")
    X = preprocess_ingredients(X)
    Y = preprocessor.preprocess_cuisines(Y)

    vectors = vectorize(x=X, voc=voc)

    data = {"ingredients": X, "vectors": vectors, "cuisines": Y}

    file = open(f'{basedir}/data/validation.sav', "wb")
    pickle.dump(data, file)
    file.close()
    print("finished preprocessing test")


if __name__ == '__main__':
    print('-creating data')
    main()

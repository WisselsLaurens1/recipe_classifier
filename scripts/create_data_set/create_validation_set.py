import sys

from pandas.core.indexes import base
sys.path.append('../')
import json
from pathlib import Path
from preprocessor import Preprocessor
from tools import *
import os
import re
from preprocess import *


basedir = Path(__file__).resolve().parents[2]


INPUT_PATH = f'{basedir}/data/test/classified'
OUTPUT_PATH = f'{basedir}/data/validation.csv'
CLEAN = True

if CLEAN:
    OUTPUT_PATH = f'{basedir}/data/cleaned_validation.csv'


def clean(test_ingredients, test_cuisines, IGNORE, reg):
    tmp_X = []
    tmp_Y = []

    # simplify, translate and remove unwanted cuisines
    for ingredients, cuisine in zip(test_ingredients, test_cuisines):
        if not (cuisine in IGNORE):
            match = re.search(reg, cuisine, re.IGNORECASE)
            if match:

                tmp_Y.append(match.group())
            else:
                tmp_Y.append(cuisine)

            tmp_X.append(ingredients)

    return tmp_X, tmp_Y


def main():
    '''
    use web scraped data and transform it using data from kaggle dataset
    '''

    with open(f'{basedir}/data/train.json') as f:
        raw = f.read()
        data = json.loads(raw)

    preprocessor = Preprocessor(f'{basedir}/meta/units.json', f'{basedir}/meta/cuisines.json')

    x = [sample['ingredients'] for sample in data]
    x = preprocess_ingredients(x)
    # x = preprocessor.preprocess_ingredients(x)

    y = [sample['cuisine'] for sample in data]

    voc = create_voc(x=x)

    train_ingredients = x

    data = []
    for sample in train_ingredients:
        data.extend(sample)
    data = list(set(data))

    # load ingredients and cuisines from web scraped data
    test_ingredients = []
    test_cuisines = []

    for file in os.listdir(INPUT_PATH):
        with open(f'{INPUT_PATH}/{file}', 'r') as f:
            raw = f.read()
            data = json.loads(raw)
            key = list(data.keys())[0]
            data = data[key]
            test_ingredients.append(data['ingredients'])
            test_cuisines.append(data['cuisine'])

    ######### clean up test set to match train set ##############

    with open(f'{basedir}/meta/cuisines.json', 'r') as f:
        raw = f.read()
        data = json.loads(raw)
        cuisines = data['cuisines']

    cuisines.append('american')

    reg = r'\b|\b'.join(cuisines)
    reg = r'\b'+reg+r'\b'

    IGNORE = ['Continental', 'Asian']

    if CLEAN:
        test_ingredients, test_cuisines = clean(
            test_ingredients, test_cuisines, IGNORE, reg)

    test_ingredients = preprocess_ingredients(test_ingredients)
    # test_ingredients = preprocessor.preprocess_ingredients(test_ingredients)
    test_cuisines = preprocessor.preprocess_cuisines(test_cuisines)
    
    # cap data set with a limited number of datapoints with an invalid class
    # this is optional
    tmp_X = []
    tmp_Y = []
    i = 0
    limit = 1500
    limit = 100000
    for ingredients, cuisine in zip(test_ingredients, test_cuisines):

        if cuisine == 'invalid':
            if i < limit:
                tmp_X.append(ingredients)
                tmp_Y.append(cuisine)
                i += 1
        else:
            tmp_X.append(ingredients)
            tmp_Y.append(cuisine)

    test_ingredients = tmp_X
    test_cuisines = tmp_Y

    # remove recipes with empty ingredients
    tmp_X = []
    tmp_Y = []
    for sample, y in zip(test_ingredients, test_cuisines):
        if len(sample) != 0:
            tmp_X.append(sample)
            tmp_Y.append(y)

    vectors = vectorize(x=tmp_X, voc=voc)

    print(len(vectors[0]))


    # write(X=vectors, Y=tmp_Y, headers=voc, path=OUTPUT_PATH)


if __name__ == '__main__':
    print("creating data")
    main()

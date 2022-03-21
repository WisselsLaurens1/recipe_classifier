from pandas.core.indexes import base
# sys.path.append('../')
import json
from pathlib import Path
import os
from preprocess import *
import multiprocessing
import pickle

print("run")

basedir = Path(__file__).resolve().parents[1]


file = open(f'{basedir}/data/test_ingredients', 'rb')
test_ingredients = pickle.load(file)
file.close()

test_ingredients = list(set(test_ingredients))

file = open(f'{basedir}/data/train_ingredients', 'rb')
train_ingredients = pickle.load(file)
file.close()

train_ingredients = list(set(train_ingredients))

print(f'len test: {len(test_ingredients)}')
print(f'len train: {len(train_ingredients)}')
size = len(test_ingredients)
i = 0

invalid = []
valid = []

for ing in test_ingredients:
    if ing in train_ingredients:
        i += 1
        valid.append(ing)

    else:
        invalid.append(ing)

print(i/size)
print(f'invalid: {len(invalid)}')
print(f'valid: {len(valid)}')


for ing in valid:
    print(ing)





'''''
INPUT_PATH = f'{basedir}/data/test/classified'
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

print(len(test_ingredients))
_result = []
for ing_list in test_ingredients:

    result = [get_base_ing(ing) for ing in ing_list]
    _result.extend(result)

test_ingredients = _result

file = open(f'{basedir}/data/test_ingredients', 'wb')
pickle.dump(test_ingredients, file)
file.close()

INPUT_PATH = f'{basedir}/data/train.json'

# read data from dataset from kaggle
with open(INPUT_PATH, 'r') as f:
    raw = f.read()
    data = json.loads(raw)

train_ingredients = []

for ing_list in data:
    train_ingredients.extend(ing_list["ingredients"])

train_ingredients = [get_base_ing(ing) for ing in train_ingredients]


size = len(train_ingredients)
train_ingredients = set(train_ingredients)


file = open(f'{basedir}/data/train_ingredients', 'wb')
pickle.dump(train_ingredients, file)
file.close()

i = 0

for ing in train_ingredients:
    if ing in test_ingredients:
        i += 1

print(i/size)
'''''
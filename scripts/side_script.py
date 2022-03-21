from pandas.core.indexes import base
import json
from pathlib import Path
import os
basedir = Path(__file__).resolve().parents[1]


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


cuisines = list(set(test_cuisines))

for c in cuisines:
    print(c)
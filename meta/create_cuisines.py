from pandas.core.indexes import base
import json
from pathlib import Path

basedir = Path(__file__).resolve().parents[1]


INPUT_PATH = f'{basedir}/data/train.json'
with open(INPUT_PATH, 'r') as f:
    raw = f.read()
    data = json.loads(raw)

y = [sample['cuisine'] for sample in data]

cuisines = list(set(y))
cuisines.sort()

with open(f'{basedir}/meta/cuisines.json', 'w') as outfile:
    json.dump({
        "cuisines":cuisines
    }, outfile)

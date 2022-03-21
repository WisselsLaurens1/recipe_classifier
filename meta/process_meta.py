import re
import json
from nltk.util import pr
from numpy import mat

with open('/home/ubuntu/random_forest/recipe_classifier/meta/foo','r') as f:
    data = f.read()

data = data.split('\n')

# tmp = []

# for d in data:
#     # print(d)
#     match = re.search('[A-Za-z]*(?=:)',d)
#     if match != None:
#         tmp.append(match.group().lower())
#         # print(match.group().lower())
#         # print(f'"{match.group().lower()}",')

# data = tmp
# tmp = []

# for d in data:
#     # print(d)
#     match = re.sub('\d','',d)
#     tmp.append(match.strip())
#     # if match != None:
#     #     tmp.append(match.group().lower())
#         # print(match.group().lower())
#         # print(f'"{match.group().lower()}",')

for d in data:
    print(f'"{d}",')


with open('./adjectives.json', 'w') as outfile:
    json.dump({"adjectives":tmp}, outfile)
import re
import json
import spacy
from nltk.util import pr
from spacy import displacy


basedir = "/home/ubuntu/random_forest/recipe_classifier"

def get_unit_list(path_to_units: str):
        '''
        returns list of units of measurment,stored in the units json file,in the correct order.
        '''
        with open(path_to_units, 'r') as f:
            raw = f.read()
            data = json.loads(raw)
            unit_list = data['units']
            result = []
            for unit_data in unit_list:
                # plural has to come first because of the way regex matches words.
                plural = unit_data['plural']
                unit = unit_data['unit']
                abbreviation = unit_data['abbreviation']

                if plural != '':
                    result.append(plural)
                if unit != '':
                    result.append(unit)
                if abbreviation != '':
                    result.append(abbreviation)
            return result

units = get_unit_list(path_to_units=f'{basedir}/meta/units.json')
unit_pattern = [fr' {unit} ' for unit in units]
unit_pattern = f'({"|".join(unit_pattern)})'

nlp = spacy.load('en_core_web_sm')


unwanted_adjectives = []
with open(f'{basedir}/meta/coocking_adjectives/coocking_adjectives.json','r') as f:
    raw = f.read()
    data = json.loads(raw)
        
for key in data:
    unwanted_adjectives.extend(data[key])

with open(f'{basedir}/meta/adjectives.json','r') as f:
    raw = f.read()
    data = json.loads(raw)

unwanted_adjectives.extend(data['adjectives'])

unwanted_adjectives = set(unwanted_adjectives)

with open(f'{basedir}/meta/colors.json','r') as f:
        raw = f.read()
        data = json.loads(raw)

colors = data['colors']

wanted_adjectives = []

wanted_adjectives.extend(colors)
wanted_adjectives = set(wanted_adjectives)


with open(f'{basedir}/meta/manual_wanted.json','r') as f:
    raw = f.read()
    data = json.loads(raw)

manual_wanted = set(data['manual_wanted'])

with open(f'{basedir}/meta/manual_unwanted.json','r') as f:
    raw = f.read()
    data = json.loads(raw)

manual_unwanted = set(data['manual_unwanted'])


###############################################################################


def remove_brackets_and_content(ing:str)->str:
        # remove brackets and content
    _ing = []
    bracket = False
    for word in ing:
        if word == "(":
            bracket = True

        if not bracket:
            _ing.append(word)

        if word == ")":
            bracket = False

    ing = ''.join(_ing)

    return ing

def remove_measurments(ing:str)->str:
    #remove numbers
    #remove (e.g:  1/2,  1/4)
    
    ing = re.sub(r'(\d\/\d)','',ing)
    ing = re.sub(r'\d', '', ing)

    return ing 

def remove_units(ing:str)->str:

    #add extra space at the beginning to match units at start of sentence 
    ing = f' {ing}'
    ing = re.sub(unit_pattern, ' ', ing)

    
    #remove extra space 
    ing = ing.strip()

    return ing

def remove_slash_and_word_after(ing:str)->str: 
    #alway take the first ingredient in row of slashes

    ing = ing.split('/')[0]

    return ing

def remove_unwanted_chars(ing:str)->str:
    ing = re.findall("[A-Za-z0-9-']*", ing)
    ing = [w for w in ing if w != '']
    ing = ' '.join(ing)
    return ing

def remove_extra_spaces(ing:str)->str:

    ing = ing.split()
    ing = [w for w in ing if w != ' ']
    ing = ' '.join(ing)
    return ing 

def remove_unwanted_words(ing:str)->str:
    ing = ing.split(' ')

    _ing = []
    for word in ing:
        if (not(word in unwanted_adjectives) and not(word in manual_unwanted)) or word in wanted_adjectives:
            #remove adjectives of form e.g: extra-virgin, extra-small...
            if re.match("([A-Za-z]*-[A-Za-z]*)",word) == None:
                _ing.append(word)
    ing = ' '.join(_ing)

    return ing

def lematize_ing(ing:str)->str:
    # ing = ing.split(' ')
    # ing = nlp(ing)
    # ing = [word.lemma_ for word in ing]
    # print(ing)
    # ing = ' '.join(ing)
    # return ing

    #made to handle "-"

    ing = ing.split()
    ing = [nlp(word) for word in ing]

    _ing = []
    for word in ing:
        _ing.append(''.join([token.lemma_ for token in word ]))

    ing = ' '.join(_ing)

    return ing

def get_base_ing(ing:str,display=False)->str:

    ing = ing.lower()
    ing = remove_brackets_and_content(ing)
    ing = remove_measurments(ing)
    ing = remove_units(ing)
    ing = remove_slash_and_word_after(ing)
    ing = remove_unwanted_chars(ing)
    ing = remove_extra_spaces(ing)
    ing = remove_unwanted_words(ing)
    ing = lematize_ing(ing)

    ing = f'I consumed a {ing}'
    
        
    text = nlp(ing)

    base_ing = ''

    wanted_dep = ["dobj","compound","amod","dep"]
    wanted_tags = ["JJ","JJS"]

    if display:
            displacy.serve(text, style="dep")

    for word in text:
        
        #dobj for direct object
        if word.text in manual_wanted or word.dep_ in wanted_dep or word.tag_ in wanted_tags :
            base_ing = f'{base_ing} {word.text}'

    base_ing = base_ing.strip()

    return base_ing


# result = get_base_ing("extra-virgin olive oil",False)
# print(f'base_ing: {result}')

def preprocess_ingredients(ingredients):
    return [[get_base_ing(ing) for ing in ing_list] for ing_list in ingredients]



# def preprocess_ingredients(ingredients):
#     result = []
#     conjunctions = ["and",'or']

#     for ing_list in ingredients:
#         length = len(ing_list)
#         # result = []
#         start = True
#         _ing_list = ing_list
#         while start or length != len(ing_list):
#             for conj in conjunctions:
#                 for ing in _ing_list:
#                     ing = _ing_list.pop(0)
#                     _ing_list.extend(ing.split(conj))

#             length = len(_ing_list)
                
#             start = False
#         result.append(_ing_list)    

    
#     # result = [[preprocess_single_ingredient(ing) for ing in ing_list] for ing_list in result]

#     _result = []
#     for ing_list in result:
#         _result.append([preprocess_single_ingredient(ing) for ing in ing_list])


#     result = result.strip()

#     return result

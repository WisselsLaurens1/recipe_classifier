import re
import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor:
    '''
    class that processes the instructions and ingredients.
    '''

    def __init__(self, path_to_units: str, path_to_cuisines: str):
        self._units = self.get_unit_list(path_to_units)
        self.path_to_cuisines = path_to_cuisines
        self._units_reg = [fr' {unit} ' for unit in self._units]
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.stopwords = stopwords.words('english')


    def remove_units(self,X):
        tmp = []
        for x in X:
            x  = self.remove_char_set(string=x, char_set=self._units_reg)
            tmp.append(x)
        return tmp

    def get_unit_list(self, path_to_units: str):
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

    def remove_char_set(self, string: str, char_set: list) -> str:
        '''
        removes all char of char_set out of string
        '''
        reg = f'({"|".join(char_set)})'
        return re.sub(reg, '', string)

    def remove_brackets_and_contents(self, string: str) -> str:
        '''
        removes all brackets and all char inside of them, from string
        '''
        reg = r"((?=\().*(?<=\)))"
        return re.sub(reg, '', string)

    def remove_all_none_letters(self, string: str) -> str:
        return re.sub('[^a-z]', ' ', string)

    def preprocess_ingredients(self, all_ingredients: list) -> list:

        all_processed_ingredients = []
        for ingredient_list in all_ingredients:

            # remove stopwords and lemmatize ingredients
            processed_ingredients = []
            for ingredient in ingredient_list:
                # remove digits, punctuation marks,...
                # ingredient = self.remove_all_none_letters(string=ingredient)
                ingredient = self.remove_char_set(string=ingredient,char_set=["\d"])
                # remove brackets and its content
                ingredient = self.remove_brackets_and_contents(ingredient)
                # remove al units of measurments
                ingredient = self.remove_char_set(
                    string=ingredient, char_set=self._units_reg)
                # remove extra whitspaces
                ingredient = ingredient.strip()

                tmp = []
                for word in ingredient.split():
                    if word not in self.stopwords:
                        # transform words to lowercase
                        word = word.lower()
                        tmp.append(self.lemmatizer.lemmatize(word))
                if len(tmp) > 0:
                    processed_ingredients.append(" ".join(tmp))

            all_processed_ingredients.append(processed_ingredients)

        return all_processed_ingredients

    def preprocess_instructions(self, all_instructions: list) -> list:
        all_processed_instructions = []

        char_set = ["[0-9]", "\n", "\.", "\!", "\?", "-", ",", "\(", "\)", "/"]
        for instructions in all_instructions:
            processed_instrudtions = self.remove_char_set(
                string=instructions, char_set=char_set)
            processed_instrudtions = processed_instrudtions.lower()

            #remove units of measurements
            processed_instrudtions = self.remove_char_set(
                    string=processed_instrudtions, char_set=self._units_reg)

            # remove stop words
            # lemmatize instructions
            # transform words to lowercase
            tmp = []
            for word in processed_instrudtions.split():
                if word not in self.stopwords:
                    word = word.lower()
                    tmp.append(self.lemmatizer.lemmatize(word))
            if len(tmp) > 0:
                all_processed_instructions.append(" ".join(tmp))

        return all_processed_instructions

    def add_space_to_elements(self, list):
        return [f' {item} ' for item in list]

    def preprocess_cuisines(self, all_cuisines: list) -> list:
        with open(self.path_to_cuisines, 'r') as f:
            raw = f.read()
            data = json.loads(raw)
            cuisines = data["cuisines"]

            cuisines_reg = '|'.join(self.add_space_to_elements(cuisines))

        processed_cuisines = []

        for cuisine in all_cuisines:

            # add spaces to make reg work for all words (nationalities existing out of a single word and nationalities exitisng of multiple words)
            cuisine = f' {cuisine} '
            result = re.search(cuisines_reg, cuisine, flags=re.IGNORECASE)
            
            if result != None:
                result = result.group()
                result = result.strip()
                result = result[0].upper() + result[1:]
                processed_cuisines.append(result.lower())
            else:
                processed_cuisines.append("invalid")

        return processed_cuisines

    def tf_idf_ingredient_analysis(self, all_instructions: list, all_ingredients: list) -> list:
        '''
        extract simplified ingredients based on tf_idf analysis
        '''

        # tokenize and build vocab
        vectors = self.vectorizer.fit_transform(all_instructions)
        features = self.vectorizer.get_feature_names()
        scores = vectors.todense()
        scores = scores.tolist()

        # TF_DF_scores = [{word: score for word, score in zip(features, scores_vector)} for scores_vector in scores]

        all_processed_ingredients = []

        for instructions, ingredients, ingredients_scores in zip(all_instructions, all_ingredients, scores):
            tot_TF_IDF = sum(ingredients_scores)
            avg_TF_IDF = tot_TF_IDF / len(ingredients_scores)

            simplified_ingredients = []
            for ingredient in ingredients:
                simplified_ingredient = []

                # word is a single word in a ingredient e.g: "chopped tomato = [chopped, tomato]"
                for word in ingredient.split():
                    if word in features:
                        i = features.index(word)
                        # todo: is word in instructions necessary
                        if ingredients_scores[i] > avg_TF_IDF and word in instructions:
                            simplified_ingredient.append(word)

                if(len(simplified_ingredient) > 0):
                    simplified_ingredients.append(
                        " ".join(simplified_ingredient))

            all_processed_ingredients.append(simplified_ingredients)

            # print(f'ingredients: {all_ingredients[i]}')
            # print(f'base ingredients: {base_ingredients}')
            # print(all_instructions[i])
            # print(f'{avg_TF_IDF}')
            # print(i)
            # for i,bi in zip(all_ingredients[i],base_ingredients):
            #     print(f'{i} = {bi}')

        return all_processed_ingredients

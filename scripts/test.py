from os import name
import re
import unittest

from numpy import result_type
from preprocess import *


class test_preprocess(unittest.TestCase):


    # def test_remove_brackets_and_contents(self):
    #     TC = "1 cup capsicum ( green pepper)"
    #     result = remove_brackets_and_content(TC)
    #     self.assertEquals(result,"1 cup capsicum ")

    #     TC = "1/2 cup yoghurt (curd)"
    #     result = remove_brackets_and_content(TC)
    #     self.assertEquals(result,"1/2 cup yoghurt ")
    
    # def test_remove_measurments(self):
    #     TC = "1/2 cup yoghurt (curd)"
    #     result = remove_measurments(TC)
    #     self.assertEquals(result," cup yoghurt (curd)")

    #     TC = "3 teaspoon lemon juice"
    #     result = remove_measurments(TC)
    #     self.assertEquals(result," teaspoon lemon juice")
 
    #     TC = "1/2 cup yoghurt (curd) and 3 teaspoon lemon juice"
    #     result = remove_measurments(TC)
    #     self.assertEquals(result," cup yoghurt (curd) and  teaspoon lemon juice")

    # def test_remove_units(self):

    #     self.assertEquals(remove_units("2 teaspoons sugar"),"2 sugar")

    #     self.assertEquals(remove_units("⅓ cup fish sauce"),"⅓ fish sauce")

    #     TC = "16 ounces sliced mushrooms (I used baby bella and white button mushrooms)"
    #     result = remove_units(TC)
    #     self.assertEquals(result,"16 sliced mushrooms (I used baby bella and white button mushrooms)")

    #     self.assertEquals(remove_units("650 gm all purpose flour"),"650 all purpose flour")

    #     self.assertEquals(remove_units("1 handful basil"),"1 basil")

    #     self.assertEquals(remove_units("handful Thai basil (or regular basil), chopped "),"Thai basil (or regular basil), chopped")

    # def test_remove_slash_and_word_after(self):
        
    #     TC = remove_measurments("pinch of hing / asafoetida")
    #     result = remove_slash_and_word_after(TC)
    #     self.assertEquals(result,"pinch of hing ")

    #     TC = remove_measurments("1/2 cup tomato ketchup")
    #     result = remove_slash_and_word_after(TC)
    #     self.assertEquals(result," cup tomato ketchup")

    #     TC = remove_measurments("¼ tsp turmeric/haldi")
    #     result = remove_slash_and_word_after(TC)
    #     self.assertEquals(result,"¼ tsp turmeric")

    #     TC = remove_measurments("½ cup maida / plain flour / all-purpose flour")
    #     result = remove_slash_and_word_after(TC)
    #     self.assertEquals(result,"½ cup maida ")

    #     TC = remove_measurments("pinch of hing/asafoetida")
    #     result = remove_slash_and_word_after(TC)
    #     self.assertEquals(result,"pinch of hing")

    #     TC = remove_measurments("½ cup maida/plain flour/all-purpose flour")
    #     result = remove_slash_and_word_after(TC)
    #     self.assertEquals(result,"½ cup maida")

    #     TC = remove_measurments("1/2 cup tomato ketchup / 1/2 cup curry ketchup")
    #     result = remove_slash_and_word_after(TC)
    #     self.assertEquals(result," cup tomato ketchup ")

    #     TC = remove_measurments("1/2 cup tomato ketchup/1/2 cup curry ketchup")
    #     result = remove_slash_and_word_after(TC)
    #     self.assertEquals(result," cup tomato ketchup")

    # def test_remove_unwanted_chars(self):

    #     TC = "½ tsp coriander powder / daniya powder"
    #     result = remove_unwanted_chars(TC)
    #     self.assertEquals(result,"tsp coriander powder daniya powder")

    #     self.assertEquals(remove_unwanted_chars("½ ಟೀಸ್ಪೂನ್ ಶುಂಠಿ ಬೆಳ್ಳುಳ್ಳಿ ಪೇಸ್ಟ್"),"")
    #     self.assertEquals(remove_unwanted_chars("1 ಇಂಚಿನ ಶುಂಠಿ"),"1")

    # def test_remove_extra_spaces(self):
    #     TC = "½   teaspoon    salt,   or   to   taste"
    #     result = remove_extra_spaces(TC)
    #     self.assertEquals(result,"½ teaspoon salt, or to taste")

    #     self.assertEquals(remove_extra_spaces(" 5 clove "),"5 clove")
    #     self.assertEquals(remove_extra_spaces("        2 tablespoon        corn     "),"2 tablespoon corn")

    def test_unwanted_adjectives(self):
        TC = "¼ cup crumbled feta cheese (optional)"
        result = remove_unwanted_words(TC)
        self.assertEqual(result,"¼ cup feta cheese (optional)")

        TC = "1 cup peanut oil"
        result = remove_unwanted_words(TC)
        self.assertEqual(result,"1 cup peanut oil")

        TC = "1/2 cup powdered jaggery"
        result = remove_unwanted_words(TC)
        self.assertEqual(result,"1/2 cup jaggery")

        TC = "1 pint cherry tomatoes or ¾ pound small tomatoes"
        result = remove_unwanted_words(TC)
        self.assertEqual(result,"1 pint cherry tomatoes or ¾ pound tomatoes")

        TC = "1 small green tomatoe"
        result = remove_unwanted_words(TC)
        self.assertEqual(result,"1 green tomatoe")
    
        TC = "½ cup chopped fresh basil, plus a handful more small basil leaves or torn leaves for garnish"
        result = remove_unwanted_words(TC)
        self.assertEqual(result,"½ cup basil, plus a handful more basil leaves or leaves for garnish")

        TC = "part chili powder"
        result = remove_unwanted_words(TC)
        self.assertEqual(result,"part chili powder")

        TC = "50 ml Thai green curry paste"
        result = remove_unwanted_words(TC)
        self.assertEqual(result,"50 ml Thai green curry paste")

        TC = "50 ml extra-virgin olive oil"
        result = remove_unwanted_words(TC)
        self.assertEqual(result,"50 ml olive oil")

        TC = "extra-small chicken"
        result = remove_unwanted_words(TC)
        self.assertEqual(result,"chicken")

        TC = "1 teaspoon crushed to paste Thai bird chilies"
        result = remove_unwanted_words(TC)
        self.assertEqual(result,"1 teaspoon paste Thai bird chilies")

    def test_get_base_ing(self):
        self.assertEquals(get_base_ing("2 tablespoon corn"),"corn")
        self.assertEquals(get_base_ing("½ teaspoon smoked hot paprika (or chipotle powder)"),"paprika")
        self.assertEquals(get_base_ing("¼ cup crumbled feta cheese (optional)"),"feta cheese")


if __name__ == '__main__':
    unittest.main()
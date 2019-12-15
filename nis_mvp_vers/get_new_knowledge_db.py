from gensim.models.keyedvectors import FastTextKeyedVectors
#fasttext = FastTextKeyedVectors.load("D:/fasttext_word2vec/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model")
#fasttext = FastTextKeyedVectors.load("/Users/nigula/input/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model")

from text_processing import get_text_map
from os import listdir
from os.path import join
import random
import json
from collections import OrderedDict
import numpy as np
import pandas as pd
from collections import OrderedDict

texts_lenta = pd.read_csv("./articles/music_lenta.csv")


with open("text_map_improved_example.json", "r") as f:
    text_map = json.load(f)


shan = {0: OrderedDict([(0, True), (2, False), (5, False)]),
    17: OrderedDict([(0, False),
                 (2, False),
                 (4, False),
                 (8, True),
                 (10, False),
                 (12, False)]),
    26: OrderedDict([(0, False), (5, True), (7, True), (13, False)]),
    32: OrderedDict([(3, True), (5, True), (14, False), (18, False)]),
    98: OrderedDict([(0, True), (4, True), (5, True), (8, False)]),
    121: OrderedDict([#(0, True),вопрос проебан просто не вставлен в тест!!!!
                 (1, True),
                 (3, False),
                 (4, True),
                 (6, False),
                 (11, True)]),
    130: OrderedDict([(0, True),
                 (1, False),
                # (3, True),вопрос проебан просто не вставлен в тест!!!!
                 (4, False),
                 (5, False),
                 (8, False),
                 (9, True),
                 (16, True)]),
    133: OrderedDict([(2, False),
                 (6, False),
                 (8, False),
                 (9, False),
                 (18, False),
                 (20, False),
                 (40, True),
                 (49, False)]),
    200: OrderedDict([(0, False),(2, False), (4, False), (10, True), (12, True)]),
    231: OrderedDict([(2, False), (4, True), (7, False), (12, False)]),
    240: OrderedDict([(1, True), (11, False), (15, False)]),
    316: OrderedDict([(0, True), (5, False), (7, False), (11, True)]),
    331: OrderedDict([(2, False), (4, False), (10, False)]),
    334: OrderedDict([(1, False), (2, False), (5, True), (8, False)]),
    336: OrderedDict([(1, True), (7, True), (10, True), (13, True)]),
    366: OrderedDict([(0, True),
                 (1, False),
                 (5, False),
                 (8, False),
                 (10, True),
                 (11, True),
                 (15, False)]),
    371: OrderedDict([(0, False),
                 (2, False),
                 (3, False),
                 (6, False),
                 (12, False),
                 (21, False)])}#!

def get_user_db(ans_dict, ans_dict_name):
    DEBUG = False
    user = user_vector(debug = DEBUG)
    for text in text_maps_json_dict.keys():
        print("text_index", text)
        text_map = text_maps_json_dict[text]
        user.start_new_text()
        for question_sentence_index in question_text_sent_collocind_dict[str(text)].keys():

            user.update_vector_with_answer_sentence(text_map['sentences_map'][int(question_sentence_index)], 
                                            effected_collocations_start_indexes_list = question_text_sent_collocind_dict[str(text)][str(question_sentence_index)],
                                            correct_answer = ans_dict[int(text)][int(question_sentence_index)])

            user.end_text(text_map)
    user.export_user_db(ans_dict_name)


get_user_db(shan, "shan_big_musician")
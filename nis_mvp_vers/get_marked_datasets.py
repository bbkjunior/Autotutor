from test_and_recommendation_w2v import user_vector
from tqdm.auto import tqdm
from collections import OrderedDict
#from text_processing_udpipe_w2v import get_text_map
import os
from random import randint
import random
import numpy as np
import pandas as pd
import json

"""
import argparse
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('answers_json', help='path to the file with raw text')
args = parser.parse_args()
"""
#with open(args.answers_json, encoding = "utf-8") as f:
 #   answer_dict = json.load(f)


"""
json_text_map = get_text_map("./text_8.txt")

#print(json_text_map['sentences_map'][0])

user.update_vector_with_answer_sentence(json_text_map['sentences_map'][0],1)
user.update_vector_with_answer_sentence(json_text_map['sentences_map'][4],1)
user.update_vector_with_answer_sentence(json_text_map['sentences_map'][6],0)

user.end_text(json_text_map)
"""

"""
answer_dict = {"222" :['0-' ,'1-' ,'2+' ,'4+'],
"862":['1+','2-', '4+', '6-', '12-'],
"321":['5+' ,'0-' ,'4-' ,'1+'],
"364":['2+', '6-', '3-', '7-' ],
"502":['13-', '3+', '0-', '4+'],
"878":['3-', '0+', '4-', '1-'],
"666":['3-', '1-', '2-' ],
"92":['1+', '2-'],
"615":['1+', '2-'],
"450":['6-', '7-', '11+', '9+' ,'10+' ],
"722":['1+', '6-', '2+', '3+', '6-', '8-'],
"732":['8-', '0-', '3-', '4-', '2+', '1-'],
"611":['1+'],
"251":['5-', '4+', '2+', '1+', '3+']}
"""
answer_dict = {
    "user_id":"pupkin_zalupkin",
    "answers": {0: OrderedDict([(0, True), (2, False), (5, False)]),
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
                 (21, False)])}
                 }


def generate_user_knowledge_database(answers_dict_json, raw_json = False):
    if raw_json:
        with open(answers_dict_json, "r") as f:
            answers_dict = json.load(f)
    else:
        answers_dict = answers_dict_json
    # texts = pd.read_csv("3000.csv")
    user_id = answers_dict['user_id']
    user = user_vector(debug = False)
    for txt_ind in tqdm(answers_dict['answers'].keys()):
        #path = os.path.join("./for_test/", txt)
        #print(txt_ind)
        #text_map = get_text_map(texts.iloc[txt_ind]["texts_3000"], raw_text_input = True)
        text_path = "maps_music/" + str(txt_ind) + ".json"
        with open(text_path, "r", encoding = "utf-8") as f:
            text_map = json.load(f)
        
        user.start_new_text()
        answers_to_current_text_questions = answers_dict['answers'][txt_ind]
        #print(answers_to_current_text_questions)
        for sentence_index in answers_to_current_text_questions:
            correctness = answers_to_current_text_questions[sentence_index]
            #print(sentence_index, correctness)
            # for text_sentence_q_ans in current_text_answers_dict:
            #     print(text_sentence_q_ans, current_text_answers_dict[text_sentence_q_ans])
            #     correctness = current_text_answers_dict[text_sentence_q_ans]

           
            #print("sentence_ind", sentence_ind, "correctness", correctness)
            user.update_vector_with_answer_sentence(text_map['sentences_map'][int(sentence_index)],correctness)
        user.end_text(text_map)
    user.export_user_db(user_id)

    # user.start_new_text()
    # for answer_ind, correctness in zip(answer_indexes,correct_ans):
    #     user.update_vector_with_answer_sentence(text_map['sentences_map'][answer_ind],correctness)
    # user.end_text(text_map)
    # user.export_user_vector()

generate_user_knowledge_database(answer_dict, raw_json = False)
from test_and_recommendation_w2v import user_vector
from text_processing_udpipe_w2v import get_text_map
import os
from random import randint
import random
import numpy as np
import pandas as pd


"""
from collections import OrderedDict
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Perceptron, SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from statistics import mean
"""

user = user_vector(debug = True)
"""
json_text_map = get_text_map("./text_8.txt")

#print(json_text_map['sentences_map'][0])

user.update_vector_with_answer_sentence(json_text_map['sentences_map'][0],1)
user.update_vector_with_answer_sentence(json_text_map['sentences_map'][4],1)
user.update_vector_with_answer_sentence(json_text_map['sentences_map'][6],0)

user.end_text(json_text_map)
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

user_id = "vasya_pupkin"

texts_ind_list = [222,862,321, 364, 502, 878, 666,92, 615, 450, 722, 732, 611, 251]
texts = pd.read_csv("3000.csv")

for txt_ind in texts_ind_list:
    #path = os.path.join("./for_test/", txt)
    print(txt_ind)
    text_map = get_text_map(texts.iloc[txt_ind]["texts_3000"], raw_text_input = True)
    user.start_new_text()
    answer_indexes = answer_dict[str(txt_ind)]
    for answer in answer_indexes:
        sentence_ind = answer[:-1]
        correctness = answer[-1]
        if correctness == "-":correctness = False
        elif correctness == "+":correctness = True
        print("sentence_ind", sentence_ind, "correctness", correctness)
        user.update_vector_with_answer_sentence(text_map['sentences_map'][int(sentence_ind)],correctness)
    user.end_text(text_map)
user.export_user_db(user_id)

"""
    user.start_new_text()
    for answer_ind, correctness in zip(answer_indexes,correct_ans):
        user.update_vector_with_answer_sentence(text_map['sentences_map'][answer_ind],correctness)
    user.end_text(text_map)
user.export_user_vector()"""
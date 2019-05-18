from test_and_recommendation_w2v import user_vector
from text_processing_udpipe_w2v import get_text_map
import os
from random import randint
import random
import numpy as np



from collections import OrderedDict
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Perceptron, SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from statistics import mean

user = user_vector(debug = True)
"""
json_text_map = get_text_map("./text_8.txt")

#print(json_text_map['sentences_map'][0])

user.update_vector_with_answer_sentence(json_text_map['sentences_map'][0],1)
user.update_vector_with_answer_sentence(json_text_map['sentences_map'][4],1)
user.update_vector_with_answer_sentence(json_text_map['sentences_map'][6],0)

user.end_text(json_text_map)
"""

texts = [file_name for file_name in os.listdir("./for_test/") if file_name.endswith(".txt")]
print(texts)

for txt in texts:
    path = os.path.join("./for_test/", txt)
    print(path)
    text_map = get_text_map(path)
    text_len = len(text_map['sentences_map'])
    correct_ans = [0,1,1,1,1]
    answer_indexes = random.sample(range(text_len), min(5,text_len) )
    print("answer_indexes", answer_indexes)
    user.start_new_text()
    for answer_ind, correctness in zip(answer_indexes,correct_ans):
        user.update_vector_with_answer_sentence(text_map['sentences_map'][answer_ind],correctness)
    user.end_text(text_map)
user.export_user_vector()
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier, RadiusNeighborsRegressor
import json
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import math
import operator
from tqdm import tqdm
"""
import argparse
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('answers_json', help='path to the file with raw text')
args = parser.parse_args()

with open(args.answers_json, encoding = "utf-8") as f:
    ans_dict = json.load(f)
#print(ans_dict)
"""

def extract_text_map_features(text_map_path, raw_json = False):
    with open(text_map_path, "r", encoding = "utf-8") as f:
        rec_text_map = json.load(f)
        #load text features

    rec_text_features_vector = []
    rec_text_features_vector.append(rec_text_map['lix'])
    rec_text_features_vector.append(rec_text_map['ttr'])
    rec_text_features_vector.extend(rec_text_map['sent_properties'])

    #load sentence features
    sentence_map = rec_text_map['sentences_map']
    recommended_sentences = []
    for sentence_ind in range(len(sentence_map)):
        rec_sent_feat = []
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['negation'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['coreference'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['vozvr_verb'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['prich'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['deepr'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['case_complexity'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['mean_depend_length'])
        rec_sent_vec = np.array(rec_sent_feat).reshape(1, -1)
        recommended_sentences.append(rec_sent_vec)

    #load words features
    recommended_words = []
    for sentence in sentence_map:
        for word_element in sentence['sentence_words']:
            word_vector = word_element['lex_vector']
            word_vector = np.array(word_vector).reshape(1, -1)
            recommended_words.append(word_vector)
    return recommended_words, recommended_sentences, rec_text_features_vector

def predict_text_understanding(rec_text_words_vectors,rec_text_sentences_vectors, rec_text_txt_vector, word_model, sent_model, text_model):
    word_predictions = []
    for word in rec_text_words_vectors:
        word_arr = np.array(word).reshape(1, -1)
        prediction = word_model.predict(word_arr)
        word_predictions.append(prediction)
    positive_understanding = 0 
    negative_understanding = 0 
    for w_pred in word_predictions:
        if w_pred > 0:
            positive_understanding += 1
        elif w_pred < 0:
            negative_understanding += 1    
    words_understanding_probability = positive_understanding / (positive_understanding + negative_understanding)
    #print(words_understanding_probability)
    
    sent_predictions = []
    for sent in rec_text_sentences_vectors:
        sent_arr = np.array(sent).reshape(1, -1)
        prediction = sent_model.predict(sent_arr)
        sent_predictions.append(prediction)
    positive_understanding = 0 
    negative_understanding = 0 
    for snt_pred in sent_predictions:
        if snt_pred[0] == 1:
            positive_understanding += 1
        elif snt_pred[0] == 0:
            negative_understanding += 1    
    sent_understanding_probability = positive_understanding / (positive_understanding + negative_understanding)
    #print(sent_understanding_probability)
    
    text_arr = np.array(rec_text_txt_vector).reshape(1, -1)
    text_understanding_pobability = text_model.predict(text_arr)[0]
    #print(text_understanding_pobability)
    
    return words_understanding_probability, sent_understanding_probability, text_understanding_pobability

def calc_dev_from_eighty_percent(understanding_vector):
    diff_squared = 0
    for value in understanding_vector:
        diff_squared += (value - 0.8) ** 2
        #print(value - 0.8, (value - 0.8) ** 2,diff_squared )
    diff_squared /= 3
    #print(diff_squared)
    st_dev = math.sqrt(diff_squared)
    return st_dev

def get_recommended_text_json(answers_dict_json, raw_json = False, save_json_to_directory = False):
    #print("GET RECOMMENDED STARTED")
    if raw_json:
        with open(answers_dict_json, encoding = "utf-8") as f:
                ans_dict = json.load(f)
    else:
        ans_dict = answers_dict_json
    user_id = ans_dict['user_id']
    #подгружаем базу текстов
    db_3000 = pd.read_csv("3000.csv")

    #подгружаем базу знаний ученика
    text_db_path = user_id + "_text_db.csv"
    sentence_db_path = user_id + "_sentence_db.csv"
    word_db_path = user_id + "_word_db.csv"
    text_db = pd.read_csv(text_db_path, header = None)
    sentence_db = pd.read_csv(sentence_db_path, header = None)
    word_db = pd.read_csv(word_db_path, header = None)

    #обучаем ближайших соседей
    word_db_arr = np.array([np.array(list(word_db.iloc[ind])) for ind in range(len(word_db))])
    word_db_arr_X = word_db_arr[:,:-1]
    word_db_arr_y = word_db_arr[:,-1]
    #X_train, X_test, y_train, y_test = train_test_split(word_db_arr_X, word_db_arr_y, test_size=0.3)
    neigh_words = KNeighborsRegressor(n_neighbors=6)
    neigh_words.fit(word_db_arr_X, word_db_arr_y)

    sentence_db_arr = np.array([np.array(list(sentence_db.iloc[ind])) for ind in range(len(sentence_db))])
    sentence_db_arr_X = sentence_db_arr[:,:-1]
    sentence_db_arr_y = sentence_db_arr[:,-1]
    #X_train, X_test, y_train, y_test = train_test_split(sentence_db_arr_X, sentence_db_arr_y, test_size=0.15)
    neigh_sent = KNeighborsClassifier(n_neighbors=2)
    neigh_sent.fit(sentence_db_arr_X, sentence_db_arr_y)

    text_db_arr = np.array([np.array(list(text_db.iloc[ind])) for ind in range(len(text_db))])
    text_db_arr_X = text_db_arr[:,:-1]
    text_db_arr_y = text_db_arr[:,-1]
    #X_train, X_test, y_train, y_test = train_test_split(text_db_arr_X, text_db_arr_y, test_size=0.15)
    neigh_text = KNeighborsRegressor(n_neighbors=5)
    neigh_text.fit(text_db_arr_X, text_db_arr_y)

    text_recommendation_vector_dict = {}
    text_recommendation_dict = {}

    
    for text_ind in tqdm(range(int(len(db_3000)/4))):
        if str(text_ind) not in list(ans_dict.keys()):
            text_map_path = "./text_maps/text_" + str(text_ind) + ".json"
            wrd_feat, snts_feat, text_feat = extract_text_map_features (text_map_path)
            predict_text_understanding(wrd_feat, snts_feat, text_feat, neigh_words, neigh_sent, neigh_text)    
            trigr_recommendation, sentence_recommendation, text_recommendation = predict_text_understanding(wrd_feat, 
                                                                                                            snts_feat, text_feat, 
                                                                                                            neigh_words, neigh_sent, neigh_text)
            text_standard_deviation = calc_dev_from_eighty_percent([trigr_recommendation, sentence_recommendation, text_recommendation])
            text_recommendation_vector_dict[text_ind] = [trigr_recommendation, sentence_recommendation, text_recommendation]
            text_recommendation_dict[text_ind] = text_standard_deviation
    sorted_text_feat_dict = sorted(text_recommendation_dict.items(), key=operator.itemgetter(1), reverse = False)

    output_texts = {}
    rex_text_index = 1
    for rec_text_el in sorted_text_feat_dict[:5]:
        text_ind = int(rec_text_el[0])
        raw_text = db_3000.iloc[text_ind]['texts_3000']
        if len (raw_text) > 50:
            text_name = "recommended_text_" + str(rex_text_index)
            output_texts[text_name] ={"text_ind":text_ind,"raw_text": raw_text}
            rex_text_index += 1
    if save_json_to_directory:
        recommendation_json_path = user_id + "_text_recommendation.json"
        with open(recommendation_json_path, "w", encoding = "utf-8") as f:
                json.dump(output_texts, f, ensure_ascii=False, indent = 4)
    else:
        return output_texts
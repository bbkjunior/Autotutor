
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import operator

def extract_text_features(text_map_path, raw_json = False):
    with open(text_map_path, "r", encoding = "utf-8") as f:
        rec_text_map = json.load(f)
        #load text features
    rec_text_features_vector = []
    rec_text_features_vector.append(rec_text_map['lix'])
    rec_text_features_vector.append(rec_text_map['ttr'])
    rec_text_features_vector.extend(rec_text_map['average_sent_properties'])
    return rec_text_features_vector, len(rec_text_map['sentences_map'])

def extract_sent_features(text_map_path, raw_json = False):
    with open(text_map_path, "r", encoding = "utf-8") as f:
        rec_text_map = json.load(f)
        #load text features
    recommended_sentences = []
    sentence_map = rec_text_map['sentences_map']
    for sentence_ind in range(len(sentence_map)):
        rec_sent_feat = []
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['negation'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['coreference'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['vozvr_verb'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['prich'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['deepr'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['case_complexity'])
        rec_sent_feat.append(sentence_map[sentence_ind]['spec_sentence_features']['mean_depend_length'])
        recommended_sentences.append(rec_sent_feat)
    return recommended_sentences, rec_text_map

def get_sentence_text(text_map, sentence_ind):
    sent_map = text_map['sentences_map'][sentence_ind]
    text = ''
    for word_el in sent_map['sentence_words']:
        text += word_el['word'] + ' '
    text = text.strip()
    return text

def get_complicated_texts(already_handled_texts_ids, raw_json = False, save_json_to_directory = False):
    text_sentence_complexity_dict = {}
    for text_ind in tqdm(range(int(100))):
        if text_ind not in already_handled_texts_ids:
            text_map_path = "./maps_music/" + str(text_ind) + ".json"
            text_feat, sentences_count  = extract_text_features (text_map_path)
            if sentences_count > 10 and sentences_count <= 25:
                sent_complexity = sum(text_feat)/len(text_feat)
                #print(text_feat, sum(text_feat)/len(text_feat))
                text_sentence_complexity_dict[text_ind] = sent_complexity
    sorted_text_feat_dict = sorted(text_sentence_complexity_dict.items(), key=operator.itemgetter(1), reverse = False)
    complicated_texts_ids = []
    for rec_text_el in sorted_text_feat_dict[:5]:
        text_ind = int(rec_text_el[0])
        print("rec", text_ind )
        complicated_texts_ids.append(text_ind)
    questions_json = {}
    for compl_text_id in complicated_texts_ids:
        questions_json[compl_text_id] = []
        sent_complicated_dicts = {}
        text_map_path = "./maps_music/" + str(compl_text_id) + ".json"
        sent_features_list, current_txt_map  = extract_sent_features (text_map_path)
        sent_index = 0
        for sent_feat in sent_features_list:
            # print(sent_feat)
            av_sent_feat = sum(list(sent_feat))/len(sent_feat)
            sent_complicated_dicts[sent_index] = av_sent_feat
            sent_index += 1
        sorted_sent_feat_dict = sorted(sent_complicated_dicts.items(), key=operator.itemgetter(1), reverse = False)
        # print(sorted_sent_feat_dict)
        for rec_sent_el in sorted_sent_feat_dict[:min(5,len(current_txt_map['sentences_map']))]:
            sent_ind = int(rec_sent_el[0])
            #print(sent_ind)
            sentence_text = get_sentence_text(current_txt_map,sent_ind)
            questions_json[compl_text_id].append({"sent_ind":sent_ind,"sent_text":sentence_text})
    print(questions_json)
    return questions_json

already_handled_texts_ids = [1,2]
get_complicated_texts(already_handled_texts_ids)


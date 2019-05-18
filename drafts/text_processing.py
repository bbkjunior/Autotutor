import copy 

import pandas as pd
import copy

import re

import os

import operator

import pymorphy2
from pymystem3 import Mystem

import nltk

import itertools 

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from collections import OrderedDict

import progressbar
import time

from collections import OrderedDict
from string import punctuation
full_punctuation = punctuation + "–" + "," + "»" + "«"
full_punctuation


import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")
russian_stopwords.remove("бы")
russian_stopwords.remove("не")
russian_stopwords.remove("ни")

nltk.download('punkt')
tokenizer = nltk.data.load('russian.pickle')
text = "Ай да А.С. Пушкин! Ай да сукин сын!"

df_sence = pd.read_csv("./for calculation/mult_sence.csv")
sence_dict = pd.Series(df_sence.count_rank.values,index=df_sence.word).to_dict()

def low(word):
    return word.lower()
df_freq = pd.read_csv("./for calculation/freq_lables.csv")
df_freq['Lemma'] = df_freq['Lemma'].apply(low)
freq_pos_dict = pd.Series(df_freq.PoS.values,index=df_freq.Lemma).to_dict()
freq_dict = pd.Series(df_freq.freq_lable.values,index=df_freq.Lemma).to_dict()

df_eng = pd.read_csv("./for calculation/eng_words.csv", sep = ' ')
eng_words = set(df_eng['orth'])

def delete_empty_newlines(file):
    end_line_punctuation = "!.?»"
    new_lines = []
    collected_line = ''
    empty_endline = False
    with open (file, "r", encoding = "utf-8") as f:
        for line in f.readlines():
            cl_line = line.strip()
            if(cl_line[-1] not in end_line_punctuation):
                collected_line += ' ' + cl_line
                empty_endline = True
            else:
                if(collected_line):
                    collected_line += ' ' + cl_line
                empty_endline = False
            
            if not (empty_endline):
                if(collected_line):
                    new_lines.append(collected_line)
                    collected_line = ''
                    
                else:
                    new_lines.append(cl_line)
                
    return new_lines

	
def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    clean_sentence = []
    #print(sentences)
    for sentence in sentences:
        words = sentence.split()
        clean_text= ''
        for word in words:
            clean_word = ''
            for char in word:
                if char != " " and char not in full_punctuation:
                    clean_word += char.lower()
            clean_text += clean_word + ' '
        clean_text = re.sub(' +', ' ', clean_text)
        clean_text = clean_text.strip()
        clean_sentence.append(clean_text)
    return clean_sentence   

def clean_file(file_lines_list):
    preprocessed_text = []
    for line in file_lines_list:
        t = preprocess_text(line)
        #print(t)
        preprocessed_text.extend(t)

    return preprocessed_text
	
def get_sent_gramm_features_map(clean_text):
    text_grammar_map = []
    m = Mystem()
    
    bar = progressbar.ProgressBar(maxval=len(clean_text),
                                      widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    line_number = 0
    bar.start()
        
    for sentence in clean_text:
        parsed_sentence_clean = []
        parsed_sentence =  m.analyze(sentence)
        for word in parsed_sentence:
            if word['text'].isalpha() or word['text'].isdigit():
                parsed_sentence_clean.append(word)
        text_grammar_map.append(parsed_sentence_clean)
        
        line_number += 1
        bar.update(line_number)
        time.sleep(0.1)
        
    return text_grammar_map
	
def lemmatize_text_from_grammar_map(text_gr_map):
    lemm_text = []
    for sentence_gr_map in text_gr_map:
        sentence = ''
        for word_grammar_features in sentence_gr_map:
            keys = list(word_grammar_features.keys())
            values = list(word_grammar_features.values())
            word  = word_grammar_features['text']
            #print(word)
            if(word.isdigit()):
                #print("digir")
                lemma = word
                sentence += lemma + ' '
            elif ('analysis' not in keys):
                pass
            elif(word_grammar_features['analysis'] ==[]):
                pass
            else:
                lemma = word_grammar_features['analysis'][0]['lex']
                sentence += lemma + ' '
        sentence = sentence.strip()
        lemm_text.append(sentence)
    return lemm_text
	
def get_weights_empty_list(clean_lemm_sentences_list, clean_orig_sentences_list):
    weights_list = []
    assert (len(clean_lemm_sentences_list) == len(clean_orig_sentences_list))
    
    for lemm_sentence, sentence in zip(clean_lemm_sentences_list,clean_orig_sentences_list):
        #print(lemm_sentence)
        #print(sentence)
        sentence_weights = []
        assert (len(lemm_sentence.split()) == len(sentence.split()))
        for lemm, word in zip (lemm_sentence.split(),sentence.split()) :
            
            if(lemm in russian_stopwords):
                weight = {"lemma" : lemm, "orig_word": word, "weight": 0, "stop_word":True}
            elif(lemm.isdigit()):
                weight = {"lemma" : lemm, "orig_word": word,"weight": 0, "digit":True}
            else:
                weight = {"lemma" : lemm, "orig_word": word,"weight": 0}
            sentence_weights.append(weight)
        weights_list.append(sentence_weights)
    return weights_list
	
def apply_udpipe_properties(ud_parsed_file, weights_input):
    weights_sentences = copy.deepcopy(weights_input)
    with open(ud_parsed_file, "r", encoding = "utf-8") as f:
        ud_properties = []
        for line in f.readlines():
            if(line[0].isdigit()):
                word_properties = line.split('\t')
                if(word_properties[3] != 'PUNCT'):
                    ud_properties.append(word_properties)
    nltk_word_count = 0 
    for sent_w in weights_input:
        nltk_word_count += len(sent_w)
    assert (nltk_word_count == len(ud_properties))
    ud_word_index = 0
    for sent_w in weights_sentences:
        for word_w in sent_w:
            #print(word_w['orig_word'])
            #print(ud_properties[ud_word_index][6])
            
            assert (word_w['orig_word'] == ud_properties[ud_word_index][1].lower())
            #print(type(word_w))
            word_w['synt_tree'] =OrderedDict()
            word_w['synt_tree']["word_index"] = ud_properties[ud_word_index][0]
            word_w['synt_tree']["head_word_index"] = ud_properties[ud_word_index][6]
            word_w['synt_tree']["univ_relat"] = ud_properties[ud_word_index][7]
            ud_word_index += 1
    return weights_sentences
	
def get_tf_idf_dict(lemm_text_list, save_to_csv = False):
    vect = TfidfVectorizer(stop_words = russian_stopwords)
    tfidf_matrix = vect.fit_transform(lemm_text_list)
    df = pd.DataFrame(tfidf_matrix.toarray(), columns = vect.get_feature_names())
    #print(df.head())
    if (save_to_csv): df.to_csv("./text_0_tfidf.xlsx", sep = '\t')
    tf_idf_dict = df.to_dict()
    return tf_idf_dict
	
def assign_tf_idf(weights_list_input, tf_idf_dict):
    weights_list = copy.deepcopy(weights_list_input)
    for sentence_ind in range(len(weights_list)):
        for el_ind in range(len(weights_list[sentence_ind])):
            lemma = weights_list[sentence_ind][el_ind]["lemma"]
            
            if (lemma in tf_idf_dict):
                weights_list[sentence_ind][el_ind]["weight"] = tf_idf_dict[lemma][sentence_ind]
                #print(lemma, tf_idf_dict[lemma][sentence_ind])
            #else:
                #weights_list[sentence_ind][el_ind]["weight"] = 0.05
                #print(lemma, "not found")
                
    return weights_list
	

	
def update_pos(weights_list, gr_map):
    
    sentences_weights_list_output = copy.deepcopy(weights_list)
    assert len(sentences_weights_list_output) == len(gr_map)#кол-во предложений
    for weights_sentence, gr_map_sentence in zip(sentences_weights_list_output, gr_map):
        assert len(weights_sentence) == len(gr_map_sentence )#кол-во слов
        for weight_word, gramm_map_word in zip(weights_sentence, gr_map_sentence):
            keys = list(gramm_map_word.keys())
            values = list(gramm_map_word.values())
            word  = gramm_map_word['text']
            #print(weight_word['word'])
            if(word.isdigit()):
                weight_word['POS'] = "number"
            elif ('analysis' not in keys):
                pass
            elif(gramm_map_word['analysis'] ==[]):
                pass
            else:
                grammar = gramm_map_word['analysis'][0]['gr']
                assert (weight_word['lemma'] == gramm_map_word['analysis'][0]['lex'])
                grammar_sep_by_comma = grammar.split(',')
                pos = re.match('[A-Z\s]+', grammar_sep_by_comma[0])[0]
                if(pos):
                    weight_word['POS'] = pos
                else:
                    weight_word['POS'] = "undefined"
                    
    return sentences_weights_list_output
	
def create_lex_vector(lexema, pos):
    #если слова нет в часотном словаре, значит оно скорее всего довольно редкое и признаем его относящ к категории 2
    #если слово не заимствовано то признаем сложным 1 
    sence_value, freq_value, eng_value, abstract= 0,2,1,0
    diff_analysis = {}
    
    if(lexema in sence_dict):
        #print(lexema, "found")
        sence_value = sence_dict[lexema]
        
        
    if(lexema in freq_dict):
        freq_value = freq_dict[lexema]
    else:
        diff_analysis["freq_value_note"] = "not in freq dict"
    
    if(lexema in eng_words):
        eng_value = 0
    abs_suff={'аж', 'есть', 'ие', 'изм', 'изна', 'ина', 'ость', 'ота', 'ствo', 'ция',
 'чина','щина','ёж','еж'}   
    if(pos == "S"):
        for suff in abs_suff:
            if (lexema.endswith(suff)):
                abstract = 1

    diff_analysis['raw_diff_values'] = [sence_value, freq_value, eng_value, abstract]
    diff_analysis_named = OrderedDict()
    diff_analysis_named['sence_value'] = sence_value
    diff_analysis_named['freq_value'] = freq_value
    diff_analysis_named['eng_value'] = eng_value
    diff_analysis_named['abstract'] = abstract
    #diff_analysis_named = { "freq_value":freq_value, "eng_value":eng_value, "abstract":abstract, "sence_value":sence_value}

    return diff_analysis, diff_analysis_named
	
def update_with_lex_vector(weights_list_input):
    weights_list = copy.deepcopy(weights_list_input)
    for sentence_ind in range(len(weights_list)):
        for el_ind in range(len(weights_list[sentence_ind])):
            lemma = weights_list[sentence_ind][el_ind]["lemma"]
            if('digit' in weights_list[sentence_ind][el_ind]):
                diff_analysis['raw_diff_values'] = [0,0,0,0]
                diff_analysis_named = {"sence_value":0, "freq_value":0, "eng_value":0, "abstract":0}
            else:
                diff_analysis,diff_analysis_named  = create_lex_vector(lemma,weights_list[sentence_ind][el_ind]["POS"])
            weights_list[sentence_ind][el_ind]["lex_vector"] = {"raw": diff_analysis, "named": diff_analysis_named}
            
    return weights_list
	
def update_coreference_negation(weight_list):
    weight_list_output = copy.deepcopy(weight_list)
    coreference_list = ["который","которая","это","этот","эти","которое"]
    negation_list = ["не","ни","бы"]
    all_sentences_map = []
    for sentence in weight_list_output:
        sentence_map = {'words':[],"negation_items":[],"coreference_items":[]}
        for word_weight in sentence:
            #print(word_weight)
            if word_weight['lemma'] in negation_list:
                sentence_map['negation_items'].append(word_weight['lemma'])
                word_weight['negation'] = True
            elif word_weight['lemma'] in coreference_list:
                sentence_map['coreference_items'].append(word_weight['lemma'])
                word_weight['coreference'] = True
            sentence_map['words'].append(word_weight)
        all_sentences_map.append(sentence_map)
    return all_sentences_map
	
def extract_dependencies (sentences_weights_input, debug = False):
    weights = copy.deepcopy(sentences_weights_input)
    """посчиать расстояние до главного слова
    посчитать кол-во зависимых слов"""
    
    for sentence_w in weights:
        #print(sentence_w)
        dep_dict = {}
        #ищем количество зависмых слов jn rf;ljuj ckjdf
        real_index = 1
        for word_w in sentence_w['words']:
            word_w['synt_tree']['real_index'] = real_index
            head_index = word_w['synt_tree']['head_word_index']
            if(head_index in dep_dict):
                dep_dict[head_index] += 1
            elif(head_index not in dep_dict and head_index!=0):
                dep_dict[head_index] = 1
            real_index += 1
        """присваиваем посчитаное количество зависимых слов прибалвяя 1"""
        for word_w in sentence_w['words']:
            if(word_w['synt_tree']['word_index'] in dep_dict):
                word_w['synt_tree']['dep_words_count'] = dep_dict[word_w['synt_tree']['word_index']] + 1 
            else:
                word_w['synt_tree']['dep_words_count'] = 1    
        dependencies_dist = []
        for word_w in sentence_w['words']:
            if debug: print("DITANCE BETWEEN",word_w)
            current_word_real_index = word_w['synt_tree']['real_index']
            nominal_head_index = int(word_w['synt_tree']['head_word_index'])
            if(nominal_head_index != 0):
                #look for head word
                for check_index_word_w in sentence_w['words']:
                    if (check_index_word_w != word_w):
                        nominal_word_index = int(check_index_word_w['synt_tree']['word_index'])
                        if (nominal_word_index == nominal_head_index):
                            if debug:print("AND",check_index_word_w)
                            real_head_index = check_index_word_w['synt_tree']['real_index']
                            distance = abs(real_head_index - current_word_real_index)
                            if debug:print("==", distance)
                            dependencies_dist.append(distance)
                            break
        if debug:
            print(sentence_w['words'])
            #print(dependencies_dist)
        sentence_w['dependencies_length'] = dependencies_dist
    return weights
	
def update_weight(s_weight, w_weight, pos):
    if ("special_pos" in w_weight):
        w_weight['special_pos'].append(pos)
    else:
        w_weight['special_pos'] = []
        w_weight['special_pos'].append(pos)
    s_weight['sent_special_pos'].append(w_weight['lemma'] + '_' + pos)

def update_special_pos(sentences_weights_list, gr_map):
    special_pos_list = ["инф", "прич", "деепр"]
    
    sentences_weights_list_output = copy.deepcopy(sentences_weights_list)
    assert len(sentences_weights_list_output) == len(gr_map)#кол-во предложений
    for weights_sentence, gr_map_sentence in zip(sentences_weights_list_output, gr_map):
        assert len(weights_sentence['words']) == len(gr_map_sentence )#кол-во слов
        weights_sentence['sent_special_pos'] = []
        for weight_word, gramm_map_word in zip(weights_sentence['words'], gr_map_sentence):
            keys = list(gramm_map_word.keys())
            values = list(gramm_map_word.values())
            word  = gramm_map_word['text']
            #print(weight_word['word'])
            if ('analysis' not in keys):
                pass
            elif(gramm_map_word['analysis'] ==[]):
                pass
            else:
                grammar = gramm_map_word['analysis'][0]['gr']
                assert (weight_word['lemma'] == gramm_map_word['analysis'][0]['lex'])
                for spec_pos in special_pos_list:
                    if (spec_pos in grammar):
                        update_weight(weights_sentence, weight_word,spec_pos)
                grammar_sep_by_comma = grammar.split(',')
                if (len(grammar_sep_by_comma) == 1):
                    pass
                else:
                    pos = re.match('[A-Z\s]+', grammar_sep_by_comma[0])[0]
                    if(pos == "V"):
                        if(gramm_map_word["text"].endswith("сь") or gramm_map_word["text"].endswith("ся")):
                            update_weight(weights_sentence, weight_word,"возвратный")
    return sentences_weights_list_output
	
def calculate_lix_from_list_of_sentences(processed_text_sentences):
		sentences_count = len(processed_text_sentences)
		words_count = sum([len(line.split(' ')) for line in processed_text_sentences])
		long_words_count = 0 #more than 6
		for line in processed_text_sentences:
			for word in line.split():
				if len(word) > 6:
					long_words_count += 1
		lix = words_count/ sentences_count + (long_words_count * 100) / words_count
		
		return round(lix,2)
		
def calculate_type_token_ratio(lemm_text_sentences):
      all_words = []
      for sentence in lemm_text_sentences:
          words = sentence.split()
          for word in words:
              all_words.append(word)

      unqie_words = set(all_words)
      types = len(unqie_words)
      tokens = len (all_words)

      return round(types/tokens,2)
	  
def caclulate_overall_text_features(sentences_map, processed_text_sentences, lemm_text_sentences, raw_aligned_text):
    text_map = {}
    non_stop_word_count = 0
    verbs_count = 0
    inf, prich, deeprich, vozvr = 0,0,0,0
    all_spec_pos = 0
    negation_count, coreference_count = 0,0
    
    
    for sentence in sentences_map:
        for spec_pos in sentence['sent_special_pos']:
            if("_инф" in spec_pos):
                
                inf += 1
                all_spec_pos += 1
            elif("_прич" in spec_pos):
                
                prich += 1
                all_spec_pos += 1
            elif("_деепр" in spec_pos):
                
                deeprich += 1
                all_spec_pos += 1
            elif("_возвратный" in spec_pos):
                
                vozvr += 1
        negation_count += len(sentence['negation_items'])      
        coreference_count += len(sentence['coreference_items']) 
        
        for word in sentence['words']:
            #print(word)
            if (word["POS"] == "V"):
                verbs_count += 1
            if("stop_word" not in word):
                non_stop_word_count += 1
                
        
    text_map = OrderedDict()
    text_map['lix'] = calculate_lix_from_list_of_sentences(processed_text_sentences)
    text_map['ttr'] = calculate_type_token_ratio(lemm_text_sentences)
    #text_map['inf_verb'] = inf/verbs_count
    #text_map['prich_verb'] = prich/verbs_count
    #text_map['deeprich_verb'] = deeprich/verbs_count
    #text_map['vozvr_verb'] = vozvr/verbs_count
    text_map['spec_pos_verb'] = round(all_spec_pos/ verbs_count,3)
    
    text_map['negation_per_n_stop_words'] =  round(negation_count/non_stop_word_count,3)
    text_map ['coreference_per_n_stop_words'] =  round(coreference_count/non_stop_word_count,3)
    
    text_map['sentences&words_analysis'] = sentences_map
    
    text_map['raw_aligned_text'] = raw_aligned_text
    text_map['cleaned_sentences'] = processed_text_sentences
    text_map['lemm_sentences'] = lemm_text_sentences
    
    return  text_map
	
def calculate_text_map(file,udpipe_file):
    raw_aligned_sentences = delete_empty_newlines(file)
    clean_sentences_list = clean_file(raw_aligned_sentences)
    text_grammar_map =  get_sent_gramm_features_map(clean_sentences_list)
    lemm_sentences_list = lemmatize_text_from_grammar_map(text_grammar_map)
    empty_weights_1 = get_weights_empty_list(lemm_sentences_list, clean_sentences_list)
    
    text_weighs_synt = apply_udpipe_properties(udpipe_file, empty_weights_1) 
    
    tfidf_dict = get_tf_idf_dict(lemm_sentences_list)
    text_weighs_tfidf_2 = assign_tf_idf(text_weighs_synt, tfidf_dict)
    pos_words_weights_4 = update_pos(text_weighs_tfidf_2, text_grammar_map)
    lex_vector_weights_5 = update_with_lex_vector(pos_words_weights_4)
    sentence_map_1 = update_coreference_negation(lex_vector_weights_5)
    sts_map_dep = extract_dependencies(sentence_map_1, debug = False)   
    sentence_spec_2 = update_special_pos(sts_map_dep, text_grammar_map)
    text_map = caclulate_overall_text_features(sentence_spec_2, clean_sentences_list,lemm_sentences_list, raw_aligned_sentences)
    return text_map

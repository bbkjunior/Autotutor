#!/usr/bin/env python
# -*- coding: utf-8 -*-

from string import punctuation
full_punctuation = punctuation + "–" + "," + "»" + "«" + "…" +'’'

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from gensim.models.keyedvectors import FastTextKeyedVectors
from collections import OrderedDict
import copy

import re
from statistics import mean 
import numpy as np

import progressbar
import time

import pymorphy2

from ud_class import Model

import json

import operator

#fasttext = FastTextKeyedVectors.load("/Users/nigula/input/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model")
fasttext = FastTextKeyedVectors.load("D:/fasttext_word2vec/araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model")
#fasttext = FastTextKeyedVectors.load("/Users/lilyakhoang/input/araneum_none_fasttextskipgram_300_5_2018/araneum_none_fasttextskipgram_300_5_2018.model")

#"smart_colloc_freq.json"


with open ("lyashevskaya_freq_dict.json" , "r", encoding="utf-8") as f:
    lyashevskaya_freq_dict = json.load(f)
def read_text(path):
    raw_text = ''
    with open (path, 'r', encoding = "utf-8") as f:
        for line in f.readlines():
            raw_text += line + ' ' 
    return raw_text

def get_conllu_from_unite_line_text(text, model):
    sentences = model.tokenize(text)
    for s in sentences:
        model.tag(s)
        model.parse(s)
    conllu = model.write(sentences, "conllu")
    return conllu
    
def get_conllu_text_map(conllu_parsed_object):
    conllu_text_map = []
    conllu_sentence_map = []
    for line in conllu_parsed_object.split('\n'):
        if line:
            if line[0].isdigit():
                #print(line.split('\t'))
                conllu_sentence_map.append(line.split('\t'))
            else:
                if(len(conllu_sentence_map) > 0):
                    conllu_text_map.append(conllu_sentence_map)
                    conllu_sentence_map = []   
                    #print("appended")
    if(len(conllu_sentence_map) > 0):
        conllu_text_map.append(conllu_sentence_map)
    return conllu_text_map
    
def get_lemm_and_orig_text_from_udmap(conllu_map):
    lemm_sentences_list = []
    sentences_list = []
    for sentence in conllu_map:
        lemm_line = ''
        line = ''
        for word in sentence: 
            if (word[3] != 'PUNCT'):
                #print(word[2])
                clean_lemma = ''
                for char in word[2]:
                    if char not in full_punctuation:
                        clean_lemma += char.lower()
                lemm_line += clean_lemma + ' '
                line += word[1] + ' '
        
        lemm_sentences_list.append(lemm_line.strip())
        sentences_list.append(line.strip())
        #print()
    return lemm_sentences_list, sentences_list
    
def get_tf_idf_dict(lemm_text_list, save_to_csv = False):
    vect = TfidfVectorizer()#stop_words = russian_stopwords
    tfidf_matrix = vect.fit_transform(lemm_text_list)
    df = pd.DataFrame(tfidf_matrix.toarray(), columns = vect.get_feature_names())
    #print(df.head())
    if (save_to_csv): df.to_csv("./text_0_tfidf.xlsx", sep = '\t')
    tf_idf_dict = df.to_dict()
    return tf_idf_dict
    
def get_verb_prop(word, morph):
    analysis = morph.parse(word)[0]
    try: 
        if (analysis.tag.POS == 'GRND' ):
            return 'GRND'
        elif("PRT" in analysis.tag.POS):
            return 'PRT'
        else:
            return None
    except:
        return None
        
def create_map(conllu_map, tf_idf_dict):
    morph = pymorphy2.MorphAnalyzer()
    text_map = []
    sentence_ind = 0
    #bar = progressbar.ProgressBar(maxval=len(conllu_map),widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    #bar.start()
    
    for sentence in conllu_map:
        sentence_map = []
        for word in sentence: 
            if (word[3] != 'PUNCT'):
                clean_lemma = ''
                for char in word[2]:
                    if char not in full_punctuation:
                        clean_lemma += char.lower()
                weight = OrderedDict([("word", word[1]),("lemma",clean_lemma), ("vocabulary_prop",(OrderedDict([("tf_idf", 0),("nominal_index",word[0])]))), 
                                     ("grammar_prop", OrderedDict([('pos',word[3] )]))])#,("lex_vector",None)
                if(word[3] == "VERB"):
                    verb_prop = get_verb_prop (word[1], morph)
                    if verb_prop:
                        weight['grammar_prop']['verb_prop'] = verb_prop
                if (word[3] == "NOUN"):
                    if("Case" in word[5]):
                        grammar = word[5].split("|")
                        for gr in grammar:
                            if("Case" in gr):
                                case = gr.split("=")[1]
                                weight['grammar_prop']['case'] = case
                
                #lemma_lower = word[2].lower()
                if (clean_lemma in tf_idf_dict):
                    weight["vocabulary_prop"]["tf_idf"] = tf_idf_dict[clean_lemma][sentence_ind]
                sentence_map.append(weight)
                
        text_map.append(sentence_map)
        sentence_ind += 1
        
        #bar.update(sentence_ind)
        #time.sleep(0.1)
    return text_map
    
    
def get_dependencies (conllu_map, text_map_input):
    sentence_map = []
    assert len(conllu_map) == len(text_map_input) #sentences count is equal
    text_map = copy.deepcopy(text_map_input)
    for sentence, text_map_sentence in zip(conllu_map,text_map):
        one_sentence_map = OrderedDict([("spec_sentence_features",(OrderedDict([("negation", 0),('coreference',0),("vozvr_verb",0),("total_vozvr",0),
                                                                                ("prich",0),("total_prich",0),("deepr",0),("total_deepr",0),
                                                                       ("case_complexity",0),("total_case",0)]))), ("syntax_prop",OrderedDict()), 
                                         ("sentence_words", [])])#("average_vocabulary", []),
                                        
        nominal2real_index_dict = {}
        real_index = 1
        
        for word in sentence: 
            if (word[3] != 'PUNCT'):
                nominal2real_index_dict[int(word[0])] = int(real_index)
                real_index += 1
                #print(word[1], "head_word_nominal_index =", word[6])

        #print(nominal2real_index_dict)                
        distances_list= []
        for word in sentence:
            if (word[3] != 'PUNCT'):
                #print("DIST CALC PROCESS")
                head_nominal_index = word[6]
                if (int(head_nominal_index) != 0 and int(head_nominal_index) in nominal2real_index_dict):
                    current_element_real_index = nominal2real_index_dict[int(word[0])]
                    head_element_real_index = nominal2real_index_dict[int(head_nominal_index)]
                    distance = abs(current_element_real_index - head_element_real_index)
                    distances_list.append(distance) 
                    #print(word[1],current_element_real_index,  head_element_real_index)
                else:
                    #print(head_nominal_index, "NOT IN DIST LIST")
                    pass
        #print(distances_list)
        one_sentence_map["syntax_prop"]["distances_list"] = distances_list
        sentence_vocab_importance = 0 
        for map_word in text_map_sentence:
            sentence_vocab_importance += map_word["vocabulary_prop"]["tf_idf"]
            one_sentence_map["sentence_words"].append(map_word)
        one_sentence_map["syntax_prop"]["sent_vocab_imp"] = sentence_vocab_importance
        sentence_map.append(one_sentence_map)   
    return sentence_map
    

def get_colloc(ngr, words_list, handled_words_indexes, collocations_dict, sentence_collected_collocation, unigramm_db, debug = False):#присваивать по диту и потом cортировать по ключам
    if ngr <  len(words_list):
        for word_ind in range(ngr, len(words_list) + 1):
            #print("word_ind", word_ind)
            ngramm = ''
            sub_ind = []
            for word_sub_ind in range(word_ind - ngr, word_ind): 
                if word_sub_ind in handled_words_indexes:
                    if debug:print("ALREADY HANDLED")
                    ngramm = None
                    break
                sub_ind.append(word_sub_ind)
                #print("word_sub_ind", word_sub_ind)
                ngramm += words_list[word_sub_ind]['lemma'] + ' '
                if debug:print(ngramm)
            if ngramm:
                if debug:print(sub_ind)
                ngramm = ngramm.strip() 
                #print(collocations_dict)
                if ngramm in collocations_dict.keys():
                    if debug:print("COLLOC FOUND")
                    ngramm_lemmas_list = ngramm.split()
                    freq_list = []
                    local_corpora_freq_list = []
                    for lemma in ngramm_lemmas_list:
                        if lemma in lyashevskaya_freq_dict:
                            freq_list.append(lyashevskaya_freq_dict[lemma])
                        else:
                            freq_list.append(0)
                        if lemma in unigramm_db:
                            local_corpora_freq_list.append(unigramm_db[lemma])
                        else:
                            local_corpora_freq_list.append(0)
                    freq_mean = mean(freq_list)
                    local_corpora_mean = mean(local_corpora_freq_list)
                    handled_words_indexes.extend(sub_ind)
                    sentence_collected_collocation[sub_ind[0]] = (ngramm, local_corpora_mean,freq_mean)
                    if debug:print("sentence_collected_collocation", sentence_collected_collocation)
            if debug:print(ngramm)
 
#vector function here
def update_with_colloc_vectors(text_map_input,colloc_db, unigramm_db):
    text_map = copy.deepcopy(text_map_input)
    #print(text_map)
    
    for sentence in text_map:
        sentence['collocation_index_list'] = []
        sentence['collocation_vectors_list'] = []
        sentence_collocations = {}
        handled_words_ind = []
        #print(colloc_db['4'])
        get_colloc(4, sentence['sentence_words'], handled_words_ind, colloc_db['4'], sentence_collocations,unigramm_db)
        #print("handled_words_ind after qgramm", sorted(handled_words_ind))
        get_colloc(3, sentence['sentence_words'], handled_words_ind, colloc_db['3'], sentence_collocations,unigramm_db)
        #print("handled_words_ind after trigramm", sorted(handled_words_ind))
        get_colloc(2, sentence['sentence_words'], handled_words_ind, colloc_db['2'], sentence_collocations,unigramm_db)
        #print("handled_words_ind after bigramm", sorted(handled_words_ind))
        for ind in range (len(sentence['sentence_words'])):
            if ind not in handled_words_ind:
                #try:
                #print(len(sentence['sentence_words']), ind)
                lemma = sentence['sentence_words'][ind]['lemma']
                try:
                    w2v = fasttext[lemma]
                    try:
                        unigr_freq = unigramm_db[lemma]
                    except:
                        unigr_freq = 0
                    if lemma in lyashevskaya_freq_dict:
                        sentence_collocations[ind] = (lemma, unigr_freq,lyashevskaya_freq_dict[lemma])
                    else:
                        sentence_collocations[ind] = (lemma, unigr_freq,0)
                        #print(lemma, "out of dict")
                    
                except:
                    #print(lemma, "missed during unigr extraction")
                    pass
                  
        #print("FINAL COLLOCATIONS")
        #print(sentence_collocations)
        colloc_list = []
        #print(sorted (sentence_collocations))
        for i in sorted (sentence_collocations) : 
            #print(i, sentence_collocations[i])
            sentence['collocation_index_list'].append((i, sentence_collocations[i]))
            ngramm = sentence_collocations[i][0]
            ngramms_list = ngramm.split()
            w2v_list = []
            for word in ngramms_list:
                #print(word)
                try:
                    w2v = fasttext[word]
                    w2v_list.append(w2v)
                except:
                    pass
            colleted_w2v_count = len(w2v_list)
            if colleted_w2v_count > 0:
                vect_sum = 300 * [0]
                for w2v in w2v_list:
                    vect_sum += w2v
                vect_sum /=  colleted_w2v_count
            else:
                print(ngramms_list, "none in fasttext")
                vect_sum = None
            if np.any(vect_sum):
               sentence['collocation_vectors_list'].append((i, sentence_collocations[i],vect_sum.reshape(1,-1).tolist()))
               #sentence['collocation_vectors_list'].append((ngramm, vect_sum.reshape(1,-1).tolist()))
            else:
               sentence['collocation_vectors_list'].append((i, sentence_collocations[i],None))
               #sentence['collocation_vectors_list'].append((ngramm, None))
            
            #print ((i, sentence_collocations[i]))
        
    return text_map
           
def increment_dict(dict_name, property_name, value):
    if property_name in dict_name:
        dict_name[property_name] += value
    else:
        dict_name[property_name] = value
   
def features_extraction(sentence_map_input):
    sentence_map = copy.deepcopy(sentence_map_input) 
    for sentence in sentence_map:
        previous_word_is_noun = False
        previous_noun_case = None
        previous_noun_vocab_importance = 0 
        
        current_sentence_vocab_vectors = []
        
        if (len(sentence['syntax_prop']['distances_list']) > 0):
            sentence['spec_sentence_features']['mean_depend_length'] = mean(sentence['syntax_prop']['distances_list']) * 0.1
        else:
            sentence['spec_sentence_features']['mean_depend_length'] = 0
        
        for word in sentence['sentence_words'] :
            #current_lex_vector = word['lex_vector']
            #current_sentence_vocab_vectors.append(current_lex_vector)
            
            if word['lemma'] == 'который' or word['lemma'] == 'это' or word['lemma'] == 'этот':
                #spec_word_partial_importance = word['vocabulary_prop']['tf_idf']/sentence['syntax_prop']['sent_vocab_imp']
                increment_dict(sentence['spec_sentence_features'],'coreference', 1/len(sentence['sentence_words']))
            elif word['lemma'] == 'бы' or word['lemma'] == 'не' or word['lemma'] == 'ни':
                increment_dict(sentence['spec_sentence_features'],'negation', 1/len(sentence['sentence_words']))
            elif (word['grammar_prop']['pos'] == 'VERB'):
                if (word['word'].endswith('ся') or word['word'].endswith('ся')):
                    spec_word_partial_importance = word['vocabulary_prop']['tf_idf']/sentence['syntax_prop']['sent_vocab_imp']
                    increment_dict(sentence['spec_sentence_features'], 'total_vozvr', word['vocabulary_prop']['tf_idf'])
                    increment_dict(sentence['spec_sentence_features'], 'vozvr_verb', spec_word_partial_importance)
                    
            if 'verb_prop' in word['grammar_prop']:
                if word['grammar_prop']['verb_prop'] == 'PRT':
                    spec_word_partial_importance = word['vocabulary_prop']['tf_idf']/sentence['syntax_prop']['sent_vocab_imp']
                    increment_dict(sentence['spec_sentence_features'], 'total_prich', word['vocabulary_prop']['tf_idf'])
                    increment_dict(sentence['spec_sentence_features'], 'prich', spec_word_partial_importance)
                elif word['grammar_prop']['verb_prop'] == 'GRND':
                    spec_word_partial_importance = word['vocabulary_prop']['tf_idf']/sentence['syntax_prop']['sent_vocab_imp']
                    increment_dict(sentence['spec_sentence_features'], 'total_deepr', word['vocabulary_prop']['tf_idf'])
                    increment_dict(sentence['spec_sentence_features'], 'deepr', spec_word_partial_importance)
                    
            if (word['grammar_prop']['pos'] == 'NOUN'):       
                if previous_word_is_noun == True and 'case' in word['grammar_prop']:
                    if previous_noun_case != word['grammar_prop']['case']:
                        total_importance = previous_noun_vocab_importance + word['vocabulary_prop']['tf_idf']
                        #print(sentence)
                        if (sentence['syntax_prop']['sent_vocab_imp'] > 0):
                            spec_word_partial_importance = total_importance/sentence['syntax_prop']['sent_vocab_imp'] 
                        increment_dict(sentence['spec_sentence_features'], 'total_case', total_importance)
                        increment_dict(sentence['spec_sentence_features'], 'case_complexity', spec_word_partial_importance)
                elif('case' in word['grammar_prop']):
                    #передаем инфу для следующего потенциального существительного
                    #print(word)
                    previous_word_is_noun = True
                    previous_noun_case = word['grammar_prop']['case']
                    previous_noun_vocab_importance = word['vocabulary_prop']['tf_idf']
            else:
                previous_word_is_noun = False
        
        current_sentence_vocab_vectors = np.matrix(current_sentence_vocab_vectors)
        """
        mean_sentence_vocab_vector = current_sentence_vocab_vectors.mean(0)
        mean_sentence_vocab_vector = mean_sentence_vocab_vector.tolist()
        sentence['average_vocabulary'] = mean_sentence_vocab_vector[0]
        """
    return sentence_map
    
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
      
def text_features_cal(sentence_map, orig_sentences_list, lemm_sentences_list):
    #negation coreference (sentences per all sent) vozvr_verb prich deepr case_complexity (per overall voc_imp) mean_depend_length (mean)qq
    #sence freq eng abstr
    text_map = OrderedDict([("lix", 0), ("ttr", 0), ("overall_colloc_text",[]), ("sent_properties",[]),("sentences_map", sentence_map)])
    
    lix = calculate_lix_from_list_of_sentences(orig_sentences_list)
    ttr = calculate_type_token_ratio(lemm_sentences_list)
    text_map['lix'] = lix *0.01
    text_map['ttr'] = ttr
    text_map['sentences_count'] = 0
    text_map['average_sentence_length'] = 0
    sentence_ind = 0
    words_count = 0
    for sentence in sentence_map:
        sentencce_json = {}
        sentencce_json[sentence_ind] = []
        words_count += len(sentence['sentence_words'])
        for word_element in sentence['collocation_index_list']:
            sentencce_json[sentence_ind].append((word_element[0],word_element[1][0]))
        text_map['overall_colloc_text'].append(sentencce_json)
        sentence_ind += 1
    #
    sentences_count = 0
    
    negation_count = 0
    coreference_count = 0
    #
    overall_vocab_importance = 0
    vozvr_verb_importance = 0
    prich_verb_importance = 0 
    deepr_verb_importance = 0
    case_complexity_importance = 0 
    #
    synt_distance = 0
    
    vocabulary_vectors = []
    for sentence in sentence_map:
        sentences_count += 1
        
        synt_distance += sentence['spec_sentence_features']['mean_depend_length']
        
        if(sentence['spec_sentence_features']['negation'] > 0  ):
            negation_count += 1
        
        if(sentence['spec_sentence_features']['coreference'] > 0  ):
            coreference_count += 1
        
        overall_vocab_importance += sentence['syntax_prop']['sent_vocab_imp']
        
        vozvr_verb_importance += sentence['spec_sentence_features']['total_vozvr']
        prich_verb_importance += sentence['spec_sentence_features']['total_prich']
        deepr_verb_importance += sentence['spec_sentence_features']['total_deepr']
        case_complexity_importance += sentence['spec_sentence_features']['total_case']
        
        """
        for word in sentence['sentence_words']:
            current_lex_vector = word['lex_vector']
            vocabulary_vectors.append(current_lex_vector)
    
    vocabulary_vectors = np.matrix(vocabulary_vectors)
    mean_vocab_vector = vocabulary_vectors.mean(0)
    mean_vocab_vector = mean_vocab_vector.tolist()
    
    text_map['vocab_properties'] = mean_vocab_vector [0]"""
    text_map['sentences_count'] = sentences_count * 0.01
    text_map['average_sentence_length'] = words_count/sentences_count * 0.01

    text_map['sent_properties'].append(negation_count/sentences_count)#negation_count
    text_map['sent_properties'].append(coreference_count/sentences_count)#coreference_count
    text_map['sent_properties'].append(vozvr_verb_importance/overall_vocab_importance)
    text_map['sent_properties'].append(prich_verb_importance/overall_vocab_importance)
    text_map['sent_properties'].append(deepr_verb_importance/overall_vocab_importance)
    text_map['sent_properties'].append(case_complexity_importance/overall_vocab_importance)
    text_map['sent_properties'].append(synt_distance/sentences_count)
    
    
    return text_map
    
    #"D:\input\music_smart_colloc_freq.json"
    #"C:\Autotutor\improved_approach\colloc\music_unigr_freq.json"
def get_text_map(text, unigramm_db_path, colloc_db_path, raw_text_input = False):
    with open (colloc_db_path, "r", encoding='utf-8') as f:
        colloc_db = json.load(f)
    #print("COLLOC", colloc_db['2'])
    #"unigr_freq.json"
    with open (unigramm_db_path, "r", encoding='utf-8') as f:
        unigramm_db = json.load(f)
    model = Model('./colloc/russian-syntagrus-ud-2.0-170801.udpipe')
    if raw_text_input:
        raw_text = text 
    else:
        raw_text = read_text(text)
    conllu = get_conllu_from_unite_line_text(raw_text, model)
    conllu_text_map = get_conllu_text_map(conllu)
    lemm_sentences,sentences_list = get_lemm_and_orig_text_from_udmap(conllu_text_map)
    tf_idf_dict = get_tf_idf_dict (lemm_sentences)
    text_map = create_map(conllu_text_map, tf_idf_dict)
    sentence_map_dep =  get_dependencies(conllu_text_map, text_map)
    sentence_map_colloc = update_with_colloc_vectors (sentence_map_dep,colloc_db,unigramm_db)
    sentence_map_feat = features_extraction(sentence_map_colloc) 
    json_text_map = text_features_cal(sentence_map_feat, sentences_list, lemm_sentences)
    return json_text_map
    
   
text = """Указом президента России Бориса Ельцина внесены изменения в  существующую структуру Федеральной службы безопасности РФ, утвержденную в июле прошлого года. Как говорится в поступившем сегодня сообщении Центра общественных связей ФСБ, в соответствии с Основами (Концепцией) государственной политики Российской Федерации по военномустроительству на период до 2005 года, на базе Департамента по борьбе с терроризмом и Управления конституционной безопасности ФСБ создан Департамент по защите конституционного строя и борьбе с терроризмом. В составе департамента организуются три управления с четко определенной компетенцией. В ФСБ отмечают, что "в современных условиях для российскойгосударственности имеют приоритетное значение вопросы защитыконституционного строя, сохранения целостности страны, борьбыс терроризмом и всеми формами экстремизма, а также разведывательно-подрывной деятельностью спецслужб и организаций иностранных государств". Как подчеркивается в сообщении, "органам безопасности в решении данных проблем отведена особая роль"""

text_short = """Однажды в поликлинику пришел больной.
– Что у вас болит? – спросил врач.
– У меня болит живот, – ответил молодой человек.
– Что вы ели вчера?
– Зеленые яблоки.
– Хорошо. Я дам вам лекарство для глаз, – сказал врач больному.
– Почему для глаз? Ведь у меня болит живот? – удивился молодой человек.
– Я дам вам лекарство для глаз, чтобы вы лучше видели, что вы едите, – сказал врач.
"""
#json_text_map = get_text_map(text, raw_text_input = True)


json_text_map = get_text_map(text, "C:\Autotutor\improved_approach\colloc\music_unigr_freq.json", "D:\input\music_smart_colloc_freq.json",raw_text_input = True)

with open("text_map_improved_example.json", "w") as f:
    json.dump(json_text_map,f, indent = 4, ensure_ascii = False) 
"""
print( json_text_map['sent_properties'])

for sent in json_text_map['sentences_map']:
    print(sent["collocation_index_list"],'\n')
    print(sent["spec_sentence_features"],'\n')
    print("~~~~")
    for word in sent["sentence_words"]:
        print(word['word'],word['vocabulary_prop'], word['grammar_prop'])
        print("\n")
    print ("====================")
"""
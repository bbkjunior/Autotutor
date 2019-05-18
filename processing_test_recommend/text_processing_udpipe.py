from string import punctuation
full_punctuation = punctuation + "–" + "," + "»" + "«" + "…" +'’'

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from collections import OrderedDict

import copy

import re

from statistics import mean 
import numpy as np

import progressbar
import time

import pymorphy2

import ufal.udpipe

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

class Model:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()

        return output
        
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
                lemm_line += word[2] + ' '
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
    if (analysis.tag.POS == 'GRND' ):
        return 'GRND'
    elif("PRT" in analysis.tag.POS):
        return 'PRT'
    else:
        return None
        
def create_map(conllu_map, tf_idf_dict):
    morph = pymorphy2.MorphAnalyzer()
    text_map = []
    sentence_ind = 0
    bar = progressbar.ProgressBar(maxval=len(conllu_map),widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    for sentence in conllu_map:
        sentence_map = []
        for word in sentence: 
            if (word[3] != 'PUNCT'):
                weight = OrderedDict([("word", word[1]),("lemma",word[2]), ("vocabulary_prop",(OrderedDict([("tf_idf", 0),("nominal_index",word[0])]))), 
                                     ("grammar_prop", OrderedDict([('pos',word[3] )]))])
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
                
                lemma_lower = word[2].lower()
                if (lemma_lower in tf_idf_dict):
                    weight["vocabulary_prop"]["tf_idf"] = tf_idf_dict[lemma_lower][sentence_ind]
                sentence_map.append(weight)
                
        text_map.append(sentence_map)
        sentence_ind += 1
        
        bar.update(sentence_ind)
        time.sleep(0.1)
    return text_map
    
def get_dependencies (conllu_map, text_map_input):
    sentence_map = []
    assert len(conllu_map) == len(text_map_input) #sentences count is equal
    text_map = copy.deepcopy(text_map_input)
    for sentence, text_map_sentence in zip(conllu_map,text_map):
        one_sentence_map = OrderedDict([("spec_sentence_features",(OrderedDict([("negation", 0),('coreference',0),("vozvr_verb",0),("total_vozvr",0),
                                                                                ("prich",0),("total_prich",0),("deepr",0),("total_deepr",0),
                                                                       ("case_complexity",0),("total_case",0)]))), ("syntax_prop",OrderedDict()), 
                                        ("average_vocabulary", []), ("sentence_words", [])])
             
        #DISTANCE CALCULATION
        
        
        #dep_words_count CALCULATION
        dep_dict = {}
        nominal2real_index_dict = {}
        real_index = 1
        
        for word in sentence: 
            if (word[3] != 'PUNCT'):
                nominal2real_index_dict[word[0]] = real_index
                real_index += 1
                #print(word[1], "head_word_nominal_index =", word[6])
                if(word[6] in dep_dict):
                    dep_dict[word[6]] += 1
                elif(word[6] != 0):
                    dep_dict[word[6]] = 1
        #print(nominal2real_index_dict)            
        distances_list= []
        for word in sentence:
            if (word[3] != 'PUNCT'):
                head_nominal_index = word[6]
                if (int(head_nominal_index) != 0):
                    current_element_real_index = nominal2real_index_dict[word[0]]
                    head_element_real_index = nominal2real_index_dict[head_nominal_index]
                    distance = abs(current_element_real_index - head_element_real_index)
                    distances_list.append(distance) 
                    #print(word[1],current_element_real_index,  head_element_real_index)
                    
        
        #print(distances_list)
        one_sentence_map["syntax_prop"]["distances_list"] = distances_list
        sentence_vocab_importance = 0 
        for map_word in text_map_sentence:
            if(map_word["vocabulary_prop"]["nominal_index"] in dep_dict):
                map_word["vocabulary_prop"]["dep_words_count"] = dep_dict[map_word["vocabulary_prop"]["nominal_index"]] + 1
                word_voc_importance = map_word["vocabulary_prop"]["dep_words_count"] * map_word["vocabulary_prop"]["tf_idf"]
                sentence_vocab_importance += word_voc_importance
                map_word["vocabulary_prop"]["vocab_importane"] = word_voc_importance
            else:
                map_word["vocabulary_prop"]["dep_words_count"] = 1
                word_voc_importance = map_word["vocabulary_prop"]["dep_words_count"] * map_word["vocabulary_prop"]["tf_idf"]
                sentence_vocab_importance += word_voc_importance
                map_word["vocabulary_prop"]["vocab_importane"] = word_voc_importance
            one_sentence_map["sentence_words"].append(map_word)
        one_sentence_map["syntax_prop"]["sent_vocab_imp"] = sentence_vocab_importance
            
        sentence_map.append(one_sentence_map)
    return sentence_map

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
    if(pos == "NOUN"):
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
    diff_analysis['named'] = diff_analysis_named
    

    prefix_positions_ohe = {'sence_0':0,'sence_1':1,'sence_2':2,'freq_0':3,'freq_1':4,'freq_2':5,'eng_0':6, 'eng_1':7, 
                                'abs_0':8, 'abs_1':9}
    prefix_list = ['sence_', 'freq_', 'eng_', 'abs_']
    current_lex_vector = [0,0,0,0,0,0,0,0,0,0]
    for prefix, lex_value in zip(prefix_list, diff_analysis['raw_diff_values'] ):
        complexity_group = prefix+str(lex_value)
        vector_index = prefix_positions_ohe[complexity_group]
        current_lex_vector[vector_index] = 1
    diff_analysis['ohe_vector'] = current_lex_vector
    return diff_analysis
    
def update_with_lex_vector(text_map_input):
    text_map = copy.deepcopy(text_map_input)
    for sentence in text_map:
        for word in sentence['sentence_words']:
            lex_vector = create_lex_vector(word['lemma'],word['grammar_prop']['pos'])
            word['lex_vector'] = lex_vector
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
            sentence['spec_sentence_features']['mean_depend_length'] = mean(sentence['syntax_prop']['distances_list'])
        else:
            sentence['spec_sentence_features']['mean_depend_length'] = 0
        
        for word in sentence['sentence_words'] :
            current_lex_vector = word['lex_vector']['ohe_vector']
            current_sentence_vocab_vectors.append(current_lex_vector)
            
            if word['lemma'] == 'который' or word['lemma'] == 'это' or word['lemma'] == 'этот':
                #spec_word_partial_importance = word['vocabulary_prop']['vocab_importane']/sentence['syntax_prop']['sent_vocab_imp']
                increment_dict(sentence['spec_sentence_features'],'coreference', 1/len(sentence['sentence_words']))
            elif word['lemma'] == 'бы' or word['lemma'] == 'не' or word['lemma'] == 'ни':
                increment_dict(sentence['spec_sentence_features'],'negation', 1/len(sentence['sentence_words']))
            elif (word['grammar_prop']['pos'] == 'VERB'):
                if (word['word'].endswith('ся') or word['word'].endswith('ся')):
                    spec_word_partial_importance = word['vocabulary_prop']['vocab_importane']/sentence['syntax_prop']['sent_vocab_imp']
                    increment_dict(sentence['spec_sentence_features'], 'total_vozvr', word['vocabulary_prop']['vocab_importane'])
                    increment_dict(sentence['spec_sentence_features'], 'vozvr_verb', spec_word_partial_importance)
                    
            if 'verb_prop' in word['grammar_prop']:
                if word['grammar_prop']['verb_prop'] == 'PRT':
                    spec_word_partial_importance = word['vocabulary_prop']['vocab_importane']/sentence['syntax_prop']['sent_vocab_imp']
                    increment_dict(sentence['spec_sentence_features'], 'total_prich', word['vocabulary_prop']['vocab_importane'])
                    increment_dict(sentence['spec_sentence_features'], 'prich', spec_word_partial_importance)
                elif word['grammar_prop']['verb_prop'] == 'GRND':
                    spec_word_partial_importance = word['vocabulary_prop']['vocab_importane']/sentence['syntax_prop']['sent_vocab_imp']
                    increment_dict(sentence['spec_sentence_features'], 'total_deepr', word['vocabulary_prop']['vocab_importane'])
                    increment_dict(sentence['spec_sentence_features'], 'deepr', spec_word_partial_importance)
                    
            if (word['grammar_prop']['pos'] == 'NOUN'):       
                if previous_word_is_noun == True and 'case' in word['grammar_prop']:
                    if previous_noun_case != word['grammar_prop']['case']:
                        total_importance = previous_noun_vocab_importance + word['vocabulary_prop']['vocab_importane']
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
                    previous_noun_vocab_importance = word['vocabulary_prop']['vocab_importane']
            else:
                previous_word_is_noun = False
        
        current_sentence_vocab_vectors = np.matrix(current_sentence_vocab_vectors)
        mean_sentence_vocab_vector = current_sentence_vocab_vectors.mean(0)
        mean_sentence_vocab_vector = mean_sentence_vocab_vector.tolist()
        sentence['average_vocabulary'] = mean_sentence_vocab_vector[0]
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
    text_map = OrderedDict([("lix", 0), ("ttr", 0),("vocab_properties",[]), ("sent_properties",[]),("sentences_map", sentence_map)])
    
    lix = calculate_lix_from_list_of_sentences(orig_sentences_list)
    ttr = calculate_type_token_ratio(lemm_sentences_list)
    text_map['lix'] = lix
    text_map['ttr'] = ttr
    
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
        

        for word in sentence['sentence_words']:
            current_lex_vector = word['lex_vector']['ohe_vector']
            vocabulary_vectors.append(current_lex_vector)
    
    vocabulary_vectors = np.matrix(vocabulary_vectors)
    mean_vocab_vector = vocabulary_vectors.mean(0)
    mean_vocab_vector = mean_vocab_vector.tolist()
    
    text_map['vocab_properties'] = mean_vocab_vector [0]
        
    
    text_map['sent_properties'].append(negation_count/sentences_count)#negation_count
    text_map['sent_properties'].append(coreference_count/sentences_count)#coreference_count
    
    text_map['sent_properties'].append(vozvr_verb_importance/overall_vocab_importance)
    text_map['sent_properties'].append(prich_verb_importance/overall_vocab_importance)
    text_map['sent_properties'].append(deepr_verb_importance/overall_vocab_importance)
    text_map['sent_properties'].append(case_complexity_importance/overall_vocab_importance)

    text_map['sent_properties'].append(synt_distance/sentences_count)
    
    
    return text_map
    
def get_text_map(text):
    model = Model('./udpipe parsers/russian-syntagrus-ud-2.0-170801.udpipe')
    conllu = get_conllu_from_unite_line_text(text, model)
    conllu_text_map = get_conllu_text_map(conllu)
    lemm_sentences,sentences_list = get_lemm_and_orig_text_from_udmap(conllu_text_map)
    tf_idf_dict = get_tf_idf_dict (lemm_sentences)
    text_map = create_map(conllu_text_map, tf_idf_dict)
    sentence_map_dep =  get_dependencies(conllu_text_map, text_map)
    sentence_map_lex = update_with_lex_vector(sentence_map_dep)
    sentence_map_feat = features_extraction(sentence_map_lex)   
    json_text_map = text_features_cal(sentence_map_feat, sentences_list, lemm_sentences)
    
    return json_text_map
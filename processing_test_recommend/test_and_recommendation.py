from text_processing_udpipe import get_text_map
import numpy as np
from sklearn.model_selection import train_test_split

import random
import numpy as np

from collections import OrderedDict
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Perceptron, SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from statistics import mean

class user_vector:
    def __init__(self,debug = False):
        self.debug = debug
        self.vocab_features =[]
        self.sentence_features =[]
        #coreference_items, negation_items, sent_special_pos, dependencies_length, Y (answer)                   
        self.text_fearues = [] #OrderedDict([("lix",[]),("ttr",[])])
        self.answers_count = OrderedDict([("correct_answers",0),("incorrect_answers",0)])
    
    def start_new_text(self):
        
        self.answers_count['correct_answers'] = 0
        self.answers_count['incorrect_answers'] = 0
        if self.debug:print("answers count has been reset", self.answers_count['correct_answers'], self.answers_count['incorrect_answers'])
        
    def end_text(self, text_map):
        if self.debug:
            print("\n========")
            print("SUM UP TEXT VALUES")
            print("========\n")
        correct_answers_rate = round(self.answers_count['correct_answers'] / (self.answers_count['correct_answers'] 
                                                                              + self.answers_count['incorrect_answers']),2)
        current_text_features = []
        if self.debug: 
            print("answers_count", self.answers_count)
        current_text_features.append(text_map['lix'])
        current_text_features.append(text_map['ttr'])
        current_text_features.extend(text_map['vocab_properties'])
        current_text_features.extend(text_map['sent_properties'])
        current_text_features.append(correct_answers_rate)
        self.text_fearues.append(current_text_features)
        if self.debug: print("TEXT FEATURES",self.text_fearues)
            
    def update_vector_with_answer_sentence(self, sentence_map, correct_answer):
        #update setnence and text features
        if self.debug:
            print("\n===NEW REPLY CALCULATION====")
            print("\n========")
            print("ADDING SENTENCE RESULTS")
            print("========\n")
            print("check answers count", self.answers_count['correct_answers'], self.answers_count['incorrect_answers'])
        #update setnence and text features
        if correct_answer == True:
            answer_value = 1
            self.answers_count['correct_answers'] += 1
            if self.debug: print("Answer for this question is correct")
        else:
            answer_value = 0
            self.answers_count['incorrect_answers'] += 1
            if self.debug: print("Answer for this question is incorrect")

        current_sentence_features = []
        current_sentence_features.append(sentence_map['spec_sentence_features']['negation'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['coreference'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['vozvr_verb'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['prich'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['deepr'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['case_complexity'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['mean_depend_length'])
        current_sentence_features.extend(sentence_map['average_vocabulary'])
        current_sentence_features.append(answer_value)#target variable
        self.sentence_features.append(current_sentence_features)
        
        if self.debug: print("SENTENCE FEATURES", current_sentence_features)
        
        if self.debug:
            print("\n========")
            print("ADDING VOCABULARY RESULTS")
            print("========\n")
        understanding_importance_list = []
        understanding_importance_sum = 0
        for word_w in sentence_map['sentence_words']:
            understanding_importance = word_w['vocabulary_prop']['vocab_importane']
            understanding_importance_sum += understanding_importance
            understanding_importance_list.append([word_w['lemma'], understanding_importance,word_w['lex_vector']['ohe_vector']])
        #"normalizing importance"
        for un_unit in understanding_importance_list:
            if(understanding_importance_sum > 0):
                un_unit[1] /= understanding_importance_sum
        if self.debug:print("understanding_importance_list", 
                       understanding_importance_list)
 
        if self.debug: print(understanding_importance_list)
        #УЧИТЫВАЕМ КОНТЕКСТ СЛОВА! СЛОВО СПРАВА И СЛЕВА
        
        for unit_index in range(len(understanding_importance_list) ):
            current_element = understanding_importance_list[unit_index][2]
            null_left_element = False
            left_unit_index = unit_index - 1
            if left_unit_index <0:
                null_left_element = True
                
            null_right_element = False    
            right_unit_index = unit_index + 1
            if right_unit_index >=  len(understanding_importance_list):
                null_right_element = True   
                
            if  null_left_element:
                left_element = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            else:
                left_element = understanding_importance_list[left_unit_index][2]
                
            if  null_right_element:
                right_element = [3,3,3,3,3,3,3,3,3,3]
            else:
                right_element = understanding_importance_list[right_unit_index][2]
                
            current_lex_vector = []
            current_lex_vector.extend(left_element)
            current_lex_vector.extend(current_element)
            current_lex_vector.extend(right_element)
            
        
        if (correct_answer): 
            current_lex_vector.append(un_unit[1])
            self.vocab_features.append(current_lex_vector)
        else:
            current_lex_vector.append(-1 * un_unit[1])
            self.vocab_features.append(current_lex_vector)
        if self.debug: 
            print("ELEMENT IN QUESTION",current_lex_vector,"len = ", len(current_lex_vector))
  
    def export_user_vector(self):
        #vocabulary vector
        
        #y_words = self.vocab_features[:,-1]
        words_db = np.array([np.array(word) for word in self.vocab_features])
        #words_db = np.matrix(words_db)
        X_words = words_db[:,:-1]
        y_words = words_db[:,-1]
        
        if self.debug:
            print("\n========")
            print("VOCAB VECTOR CALCULCATION")
            print("========\n")
            print("word db example\n",words_db[1,:])
        X_train, X_test, y_train, y_test = train_test_split(X_words, y_words, test_size=0.15)
        words_feat_reg = SGDRegressor(max_iter=100, tol=1e-3)
        words_feat_reg.fit(X_train, y_train)
        accuracy = words_feat_reg.score(X_test, y_test)
        
        rf_vocab = RandomForestRegressor(max_depth=2, random_state=0,
                              n_estimators=100)
        rf_vocab.fit(X_train, y_train)
        rf_accuracy = rf_vocab.score(X_test, y_test)
        
        vocab_lin_reg = LinearRegression()
        vocab_lin_reg.fit(X_train, y_train)
        linreg_accuracy = rf_vocab.score(X_test, y_test)
        
        vocab_model = rf_vocab
        
        if self.debug:
            print("SGDRegressor model trained with accuracy = ", accuracy)
            print("RandomForestRegressor model trained with accuracy = ", rf_accuracy)
            print("LinearRegression model trained with accuracy = ", linreg_accuracy)
        #sentence vector
        sentence_db = np.array([np.array(sent) for sent in self.sentence_features])
        #sentence_db = np.matrix(self.sentence_features)
        X_sent = sentence_db[:,:-1]
        y_sent = sentence_db[:,-1]
        sent_feat_reg = LinearRegression().fit(X_sent, y_sent)
        if self.debug:
            print("\n========")
            print("SENTENCE VECTOR CALCULCATION")
            print("========\n")
            print("sentence db example\n",sentence_db[1,:])
        sentence_model = sent_feat_reg
        
        #text vector
        if self.debug:
            print("\n========")
            print("TEXT VECTOR CALCULCATION")
            print("========\n")
        
        print()
        text_db = np.array([np.array(text) for text in self.text_fearues])
        #text_db = np.matrix(self.text_fearues)
        X_text = text_db[:,:-1]
        y_text = text_db[:,-1]
        text_feat_reg = LinearRegression().fit(X_text, y_text)
        if self.debug: 
            print("text db example\n", text_db[1,:])
        text_model = text_feat_reg
            
        return vocab_model, sentence_model, text_model

def recommendation(text_map, vocab_model, sent_model, text_model, debug = False):
    if debug:
            print("\n========")
            print("TEXT RECOMENDATIONS CALCULCATION")
            print("========\n")
    
    #text
    text_features = [text_map['lix'], text_map['ttr']]
    text_features.extend(text_map['vocab_properties'])
    text_features.extend(text_map['sent_properties'])
    text_fetures = np.array(text_features).reshape(1, -1)
    
    predict_answer_rate = text_model.predict(text_fetures)
    if debug:
        print("TEXT")
        print(text_fetures)
        print("predict_answer_rate", predict_answer_rate)
    
    #sentence
    if debug:
            print("\n========")
            print("SENT RECOMENDATIONS CALCULCATION")
            print("========\n")
    understanding_prob_list = []
    for sentence_map in text_map['sentences_map']:
        current_sentence_features = []
        current_sentence_features.append(sentence_map['spec_sentence_features']['negation'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['coreference'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['vozvr_verb'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['prich'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['deepr'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['case_complexity'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['mean_depend_length'])
        current_sentence_features.extend(sentence_map['average_vocabulary'])
        
        current_sentence_features = np.array(current_sentence_features).reshape(1, -1)
        predict_setence_understading = sent_model.predict(current_sentence_features)
        if debug:
            print("sentence in questuion", current_sentence_features, "prediction", predict_setence_understading)
        understanding_prob_list.append(predict_setence_understading)
    understanding_prob_list = [float (pred) for pred in understanding_prob_list]
    overall_sentences_understanding_prob = mean(understanding_prob_list)
    if debug:
        print("overall_sentences_understanding_prob", overall_sentences_understanding_prob)
    
    
    #WORDS
    if debug:
        print("\n========")
        print("WORDS RECOMENDATIONS CALCULCATION")
        print("========\n")
    positive_undertanding = 0
    negative_uderstanding = 0
    overall_vocab_imp = 0
    positive_understanding_list = []
    negative_understanding_list = []
    
    for sentence in text_map['sentences_map']: 
        overall_vocab_imp += sentence['syntax_prop']['sent_vocab_imp']
        understanding_importance_list = []
        understanding_importance_sum = 0
        for word_w in sentence['sentence_words']:
            understanding_importance = word_w['vocabulary_prop']['vocab_importane']
            understanding_importance_sum += understanding_importance
            understanding_importance_list.append([word_w['lemma'], understanding_importance,word_w['lex_vector']['ohe_vector']])
        if debug: print("understanding_importance_list", 
                   understanding_importance_list)
        for unit_index in range(len(understanding_importance_list) ):
            current_element = understanding_importance_list[unit_index][2]
            null_left_element = False
            left_unit_index = unit_index - 1
            if left_unit_index <0:
                null_left_element = True
                
            null_right_element = False    
            right_unit_index = unit_index + 1
            if right_unit_index >=  len(understanding_importance_list):
                null_right_element = True   
                
            if  null_left_element:
                left_element = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
            else:
                left_element = understanding_importance_list[left_unit_index][2]
                
            if  null_right_element:
                right_element = [3,3,3,3,3,3,3,3,3,3]
            else:
                right_element = understanding_importance_list[right_unit_index][2]
                
            current_lex_vector = []
            current_lex_vector.extend(left_element)
            current_lex_vector.extend(current_element)
            current_lex_vector.extend(right_element)
            current_lex_vector = np.array(current_lex_vector).reshape(1, -1)
            word_understanding_prediction = vocab_model.predict(current_lex_vector)

            if debug: 
                print("prediction", understanding_importance_list[unit_index][:1],current_lex_vector, word_understanding_prediction)
            un_unit = understanding_importance_list[unit_index]
            if(word_understanding_prediction > 0):
                positive_undertanding += 1
                un_unit.append(word_understanding_prediction)
                positive_understanding_list.append(un_unit)
            else:
                negative_uderstanding += 1
                un_unit.append(abs(word_understanding_prediction))
                negative_understanding_list.append(un_unit)
            if debug:
                print("+",positive_undertanding, "-", negative_uderstanding)
                
            
    positive_understanding_real = []
    positive_vocab_imp = []
    for positive_understaning_el in positive_understanding_list:
        positive_understanding_real.append(positive_understaning_el[3] * positive_understaning_el[1] )
        positive_vocab_imp.append(positive_understaning_el[1])
        
    negative_understanding_real = []
    negative_vocab_imp = []
    for negative_und_el in negative_understanding_list:
        negative_understanding_real.append(negative_und_el[3] * negative_und_el[1] )
        negative_vocab_imp.append(negative_und_el[1])
    if debug:print(sum(positive_vocab_imp), sum(negative_vocab_imp), overall_vocab_imp)
    
    sum_real_understanding = sum(positive_understanding_real) + sum(negative_understanding_real)
    positive_understanding_rate = sum(positive_understanding_real) / sum_real_understanding
    
    confidence_value = 0
    if(sum(positive_vocab_imp) > 0 ):
        if debug:print(sum(positive_understanding_real)/sum(positive_vocab_imp))#отображение уверенности в предсказании
        confidence_value = sum(positive_understanding_real)/sum(positive_vocab_imp)
    if (sum(negative_vocab_imp) > 0):
        if debug: print(sum(negative_understanding_real)/sum(negative_vocab_imp))
        
        
    #85% understanding is best
    #vocab
    confidence_value = confidence_value[0]
    positive_understanding_rate = positive_understanding_rate[0]
    vocab_recommend_diff = positive_understanding_rate - 0.85
    vocab_recommendation = (vocab_recommend_diff, confidence_value)
    
    sentence_recommendation = overall_sentences_understanding_prob - 0.85
    
    predict_answer_rate = predict_answer_rate[0]
    text_recommednation = predict_answer_rate - 0.85 
            
    return vocab_recommendation, sentence_recommendation, text_recommednation

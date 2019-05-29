from text_processing_udpipe_w2v import get_text_map
from ud_class import Model

import random
import numpy as np
from sklearn.model_selection import train_test_split


from collections import OrderedDict
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Perceptron, SGDRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from statistics import mean

from keras.models import Sequential
from keras.layers import Dense

class user_vector:
    def __init__(self,debug = False):
        self.debug = debug
        self.vocab_features =[]
        self.sentence_features =[]
        #coreference_items, negation_items, sent_special_pos, dependencies_length, Y (answer)                   
        self.text_fearues = [] #OrderedDict([("lix",[]),("ttr",[])])
        self.answers_count = OrderedDict([("correct_answers",0),("incorrect_answers",0)])
        self.trigramms_list = []
        
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
        #current_text_features.extend(text_map['vocab_properties'])
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
            
        #update setnence and text features
        if correct_answer == True:
            answer_value = 1
            self.answers_count['correct_answers'] += 1
            if self.debug: print("Answer for this question is correct")
        else:
            answer_value = 0
            self.answers_count['incorrect_answers'] += 1
            if self.debug: print("Answer for this question is incorrect")
        if self.debug:print("check answers count", self.answers_count['correct_answers'], self.answers_count['incorrect_answers'])
        current_sentence_features = []
        current_sentence_features.append(sentence_map['spec_sentence_features']['negation'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['coreference'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['vozvr_verb'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['prich'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['deepr'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['case_complexity'])
        current_sentence_features.append(sentence_map['spec_sentence_features']['mean_depend_length'])
        #current_sentence_features.extend(sentence_map['average_vocabulary'])
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
            understanding_importance = word_w['vocabulary_prop']['tf_idf']
            understanding_importance_sum += understanding_importance
            understanding_importance_list.append([word_w['lemma'], understanding_importance,word_w['lex_vector'],word_w['lex_trigram']])
            
        for un_unit in understanding_importance_list:
            if(understanding_importance_sum > 0):
                un_unit[1] /= understanding_importance_sum
        #if self.debug:print("understanding_importance_list", understanding_importance_list)
                       
        
        for unit_index in range(len(understanding_importance_list) ):
            current_element = understanding_importance_list[unit_index][2]
            """
            left_unit_index = unit_index - 1
            if left_unit_index <0:
                left_element = 300 * [0] 
            else:
                left_element = understanding_importance_list[left_unit_index][2]
            right_unit_index = unit_index + 1
            if right_unit_index >=  len(understanding_importance_list):
                right_element = 300 * [1]
            else:
                right_element = understanding_importance_list[right_unit_index][2]
             """   
            current_lex_vector = []
            #print("current_element", current_element)
            current_lex_vector.extend(current_element)
            
            if (correct_answer): 
                current_lex_vector.append(understanding_importance_list[unit_index][1])
                self.trigramms_list.append(understanding_importance_list[unit_index][3])
                #current_lex_vector.append(understanding_importance_list[unit_index][3])
                self.vocab_features.append(current_lex_vector)
                #print("current_lex_vector", current_lex_vector)
            else:
                current_lex_vector.append(-1 * understanding_importance_list[unit_index][1])
                self.trigramms_list.append(understanding_importance_list[unit_index][3])
                #current_lex_vector.append(understanding_importance_list[unit_index][3])
                self.vocab_features.append(current_lex_vector)
     
    def export_user_db(learner_id, self):
        with open ("trigramm_db.txt", "w", encoding = "utf-8") as f:
            for trig in self.trigramms_list:
                f.write(trig + '\n')
                
        words_db = np.array([np.array(word) for word in self.vocab_features])
        word_db_path = learner_id + '_word_db.csv'
        np.savetxt(word_db_path, words_db, delimiter=',') 
        
        sentence_db = np.array([np.array(sent) for sent in self.sentence_features])
        sentence_db_path = learner_id + '_sentence_db.csv'
        np.savetxt(sentence_db_path, sentence_db, delimiter=',') 
        
        text_db = np.array([np.array(text) for text in self.text_fearues])
        text_db_path = learner_id + '_text_db.csv'
        np.savetxt(text_db_path, text_db, delimiter=',') 
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
        
        model = Sequential()
        model.add(Dense(12, input_dim=300, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=150, batch_size=10)
        y_test = model.predict(X_test)
        keras.metrics.categorical_accuracy(y_true, y_pred)
        
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
        
        
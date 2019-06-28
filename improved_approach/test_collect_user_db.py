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
    
    def update_vector_with_answer_sentence(self, sentence_map, effected_collocations_start_indexes_list, correct_answer):
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
            
        for word_w in sentence_map['collocation_vectors_list']:
            if self.debug:print(word_w[1][0])
            if word_w[0] in effected_collocations_start_indexes_list:
                current_word_features = []
                #current_word_features.append(str("local_freq_") + str(word_w[1][1]))
                current_word_features.append(word_w[1][1])
                try:
                    glob_freq_log = math.log(word_w[1][2])
                except:
                    glob_freq_log = 0
                #current_word_features.append(str("global_freq_mpi_ln_") + glob_freq_log)
                current_word_features.append(glob_freq_log)
                current_word_features.extend(word_w[2][0])
                current_word_features.append(answer_value)
                #print(current_word_features)
                self.vocab_features.append(current_word_features)
    def export_user_db(self, learner_id):
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
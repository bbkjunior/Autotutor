text_map = OrderedDict([("lix", 0), 
                        ("ttr", 0), 
                        ("sent_properties",[]),
                        ("sentences_map",  OrderedDict([("spec_sentence_features",(OrderedDict([("negation", 0),('coreference',0),                                                                         ("vozvr_verb",0),("total_vozvr",0),
                                                                                                 ("prich",0),("total_prich",0),                                                         ("deepr",0),("total_deepr",0),
                                                                                                 ("case_complexity",0),("total_case",0), ("mean_depend_length",0)]))),           
                                                                                                 ("syntax_prop",OrderedDict([('distances_list'])), 
                                                                                                 (, 
                                                                                                                 [OrderedDict([("word", word[1]),("lemma",word[2]), 
                                                                                                                 ("vocabulary_prop",(OrderedDict([("tf_idf", 0),("nominal_index",word[0])]))), 
                                                                                                                                    ("grammar_prop", OrderedDict([('pos',word[3] )])),("lex_vector",None)])])]))])
                                                                                                                                    
                                                                                                                                    
                                                                                                                                    
                                                                                                                                    
 

{   "LIX": value,
    "Type_Token_Ration": value,
    "sentences_map": [{"special_sentence_features": {"percentage_of_difficult_language_objects":value, 
                                                   "percentage_of_difficult_POS":value, 
                                                   "mean_syntax dependncies":value},
                     
                    "sentence_words":[{"original_word": word, "lemma":lemma, "importance" : tf_idf, "grammar_prop": POS, "lex_vector": word2vec}, 
                                      { ... }]},
                       {...}]
  } 
  
  
  
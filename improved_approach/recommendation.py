import os
import pandas as pd
import json
from operator import itemgetter

SIMILARITY_PATH = '/Users/nigula/Autotutor/improved_approach/music_similarity'
USER_NAME = "littel_musician"


def colourise(colour, text):
    if colour == "black":
        return "\033[1;30m" + str(text) + "\033[1;m"
    if colour == "red":
        return "\033[1;31m" + str(text) + "\033[1;m"
    if colour == "green":
        return "\033[1;32m" + str(text) + "\033[1;m"
    if colour == "yellow":
        return "\033[1;33m" + str(text) + "\033[1;m"
    if colour == "blue":
        return "\033[1;34m" + str(text) + "\033[1;m"
    if colour == "magenta":
        return "\033[1;35m" + str(text) + "\033[1;m"
    if colour == "cyan":
        return "\033[1;36m" + str(text) + "\033[1;m"
    if colour == "gray":
        return "\033[1;37m" + str(text) + "\033[1;m"
    return str(text)

def eval_texts_return_top(SIMILARITY_PATH,USER_NAME, debug = False):
    text_db = pd.read_csv(USER_NAME + "_text_db.csv",header = None)
    sent_db = pd.read_csv(USER_NAME + "_sentence_db.csv",header = None)
    word_db = pd.read_csv(USER_NAME +  "_word_db.csv",header = None)
    similarity_maps = os.listdir(SIMILARITY_PATH)
    
    evaluated_texts = []

    for sim_map in similarity_maps:
        if debug:
            print('\n==================\n')
            print(sim_map)
        if sim_map.endswith(".json"):
            map_full_path = os.path.join(SIMILARITY_PATH, sim_map)
            try:
                f =  open(map_full_path, "r") #, encoding = "utf-8"
            except:
                f =  open(map_full_path, "r", encoding = "utf-8") #
            similarity_map = json.load(f)

            #TEXT LEVEL
            collected_stuff = []
            overall_similarity = 0
            for sim_text in similarity_map['similar_texts']:
                if debug:
                    print(sim_text[1]['VECTOR'])
                    print(list(text_db.iloc[sim_text[1]['TEXT_IND_in_marked_db']]))
                assert (sim_text[1]['VECTOR'] == list(text_db.iloc[sim_text[1]['TEXT_IND_in_marked_db']])[:-1])
                if debug:print(sim_text[1]['VECTOR'] == list(text_db.iloc[sim_text[1]['TEXT_IND_in_marked_db']])[:-1])
                target_variable = list(text_db.iloc[sim_text[1]['TEXT_IND_in_marked_db']])[-1]
                similarity = sim_text[0]
                overall_similarity += similarity
                collected_stuff.append([similarity, target_variable])
            if debug:print("NORMALIZED")
            for text in collected_stuff:
                text[0] /= overall_similarity
                if debug:print(text)

            text_understanding_expectation = 0
            for sent in collected_stuff:
                text_understanding_expectation += sent[0] * sent[1]
            if debug: print("TEXT UNDERSTANING EXPECTATION", text_understanding_expectation)

            #SENTENCES LEVEL
            math_exp_list = []
            for sim_sentence in similarity_map['simiar_sentences']:
                collected_similar_sentences = []
                overall_similarity = 0
                for marked_sim_sentence in sim_sentence:
                    assert (marked_sim_sentence[1]['sent_vector'] == list(sent_db.iloc[marked_sim_sentence[1]['sent_ind_in_marked_db']])[:-1])
                    target_variable = list(sent_db.iloc[marked_sim_sentence[1]['sent_ind_in_marked_db']])[-1]
                    similarity = marked_sim_sentence[0]
                    overall_similarity += similarity
                    collected_similar_sentences.append([similarity, target_variable])
                for sent in collected_similar_sentences:
                    sent[0] /= overall_similarity
                    if debug:print(sent)
                math_expectation = 0
                for sent in collected_similar_sentences:
                    math_expectation += sent[0] * sent[1]
                if debug:print("SENTENCE UNDERSTANING EXPECTATION",math_expectation)
                math_exp_list.append(math_expectation)
            sent_understanding_expectation = sum(math_exp_list) / len(math_exp_list)
            if debug:print("OVERALL SENTENCE UNDERSTANING EXPECTATION",sent_understanding_expectation)


            #WORD LEVEL
            words_math_exp_list = []
            words_prediction = []
            for sim_colloc_list in similarity_map['similar_collocations']:
                collected_similar_words = []
                overall_similarity = 0
                if debug:print("WORD FROM EXAMINED TEXT", sim_colloc_list[0])
                if len (sim_colloc_list[1]) > 0:
                    for marked_sim_word in sim_colloc_list[1]:
                        target_variable = list(word_db.iloc[marked_sim_word[0]['colloc_db_index']])[-1]
                        similarity = marked_sim_word[1]
                        overall_similarity += similarity
                        #print("sim word", marked_sim_word[0]['sim_colloc'], target_variable)
                        collected_similar_words.append([similarity, target_variable])     
                    for word in collected_similar_words:
                        word[0] /= overall_similarity
                        #print(word)  
                    math_expectation = 0
                    for word in collected_similar_words:
                        math_expectation += word[0] * word[1]

                    if math_expectation < 0.3:
                        highlighted_text = colourise("red", sim_colloc_list[0])
                    elif math_expectation >= 0.3 and math_expectation < 0.6:
                        highlighted_text = colourise("yellow", sim_colloc_list[0])
                    elif math_expectation >= 0.6:
                        highlighted_text = colourise("green", sim_colloc_list[0])
                    if debug: print("WORD UNDERSTANING EXPECTATION",highlighted_text) 
                    words_math_exp_list.append(math_expectation)   
                    words_prediction.append(highlighted_text)
                else:
                    if debug:print("no similar words found")
                    words_prediction.append(sim_colloc_list[0])
            words_understanding_expectation = sum(words_math_exp_list) / len(words_math_exp_list)
            if debug:
                print("OVERALL WORD UNDERSTANING EXPECTATION",words_understanding_expectation)
                for word_calculated in words_prediction:
                    print(word_calculated)

            
            understanding_vector = [text_understanding_expectation, sent_understanding_expectation, words_understanding_expectation]
            if debug:print(understanding_vector)
            understanding_vector_sdev = [0,0,0]
            for val_ind in range(len(understanding_vector)):
                understanding_vector_sdev[val_ind] = (understanding_vector[val_ind]  - 0.8)**2
            if debug:print(understanding_vector_sdev)
            s_dev = sum(understanding_vector_sdev) / len(understanding_vector_sdev)
            if debug:print(s_dev)
            text_index_in_db = sim_map.split(".")[0]
            evaluated_texts.append((s_dev,{"understanding_vector":understanding_vector, "words_prediction":words_prediction,"text_index_in_db":text_index_in_db}))
            f.close()

    reversed_recommended_texts = sorted(evaluated_texts,key=itemgetter(0))   
    for rec_text in reversed_recommended_texts:
        print('\n==================\n')
        print (rec_text[0])
        
        print (rec_text[1]['understanding_vector'])
        print (rec_text[1]['text_index_in_db'])
        """
        for word in rec_text[1]['words_prediction']:
            print(word)
        """
eval_texts_return_top(SIMILARITY_PATH,USER_NAME)
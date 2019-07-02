from calculate_colocations import get_pos_filtered_colloc_from_corpus_list, get_colloc_from_corpus_list,get_clean_lemm_list
import pandas as pd
from tqdm import tqdm
from collections import Counter

import json

texts_lenta = pd.read_csv("football_lenta.csv")

clean_lemm_list = get_clean_lemm_list(list(texts_lenta['text']),lang = 'rus')

unigramm_freq = Counter(clean_lemm_list)
with open ("football_lenta_unigr_freq.json","w", encoding = "utf-8") as f:
    json.dump(unigramm_freq, f, indent = 4, ensure_ascii = False)
    
bigramFreqTable, trigramFreqTable, quadgram_freq, filtered_bi, filtered_tri, bigramPMITable, trigramPMITable, quadragramPMITable, bigramChiTable, trigramChiTable =get_pos_filtered_colloc_from_corpus_list(list(texts_lenta['text']),"rus")


def cean_pos_tags(df, ngramm_name):
    clean_words = []
    for i in tqdm(range(len(df))):
        posed_ngramm = df.iloc[i][ngramm_name]
        clean_ngramm = []
        for w in posed_ngramm:
            word = w.split("_")[0]
            clean_ngramm.append(word)
        clean_ngramm = tuple(clean_ngramm)
        clean_words.append(clean_ngramm)
    
    data = { "ngramm":clean_words,"freq":df['freq']}
    clean_df = pd.DataFrame(data) 
    return clean_df
bigramFreqTable_clean = cean_pos_tags(bigramFreqTable,'bigram')
trigramFreqTable_clean = cean_pos_tags(trigramFreqTable, "trigram")
quadgram_freq_clean = cean_pos_tags(quadgram_freq, "quadgramF")
filtered_bi_clean = cean_pos_tags(filtered_bi,'bigram')
filtered_tri_clean = cean_pos_tags(filtered_tri,'trigram')

def get_freq_colloc_dict(ngrm_lis):
    freq_colloc_dict = {'2':{},'3':{},'4':{}}
    for ngramm_df in ngrm_lis:
        dct = ngramm_df.to_dict("split")
        words_len = len(dct['data'][0][0])
        for el in dct['data']:
            
            ngramm_raw = ''
            for el_i in el[0]:
                ngramm_raw += el_i + ' '
            ngramm_raw = ngramm_raw.strip()
            #print(ngramm_raw,words_len )
            freq_colloc_dict[str(words_len)][ngramm_raw] = el[1]
        
    return freq_colloc_dict
ngramms_list =[bigramFreqTable_clean, trigramFreqTable_clean, quadgram_freq_clean]
freq_colloc_dict = get_freq_colloc_dict(ngramms_list)

def get_united_numeric_colloc_base (numeric_calc_collocations_list, collocations_by_freq_dict):#все кроме фильтрованных по частям речи
    
    overall_colloc_json = {'2':{},'3':{},'4':{}}
    for colloc in numeric_calc_collocations_list:
        colloc_len = len(colloc)
        for key in list(colloc.keys()):
            if 'gram' in key:
                ngramm_name = key
            else:
                freq_name = key
        n_of_words = str(len(colloc.iloc[0][ngramm_name]))
        print(n_of_words)
        for index in tqdm(range(int(colloc_len))):
            collocation_element = colloc.iloc[index][ngramm_name]
            str_el = ''
            for el in collocation_element:
                str_el += el + ' '
            str_el = str_el.strip()
            if str_el in collocations_by_freq_dict[n_of_words]:
                overall_colloc_json[n_of_words][str_el] = collocations_by_freq_dict[n_of_words][str_el] 
            else:
                overall_colloc_json[n_of_words][str_el] = colloc.iloc[index][freq_name]
            """
            if str_el in overall_colloc_json[n_of_words]:
                overall_colloc_json[n_of_words][str_el] *= int(colloc.iloc[index][freq_name])
            else:
                overall_colloc_json[n_of_words][str_el] = int(colloc.iloc[index][freq_name])
            """
            #print(str_el)
        #print("========")
    return overall_colloc_json
#ngramms_list = [bigramFreqTable_clean, trigramFreqTable_clean, quadgram_freq_clean, bigramPMITable, trigramPMITable, quadragramPMITable]
ngramms_list = [bigramPMITable, trigramPMITable, quadragramPMITable, bigramChiTable, trigramChiTable,filtered_bi_clean,filtered_tri_clean]

log_collocations_vs_freq = get_united_numeric_colloc_base(ngramms_list,freq_colloc_dict)

with open ("football_lenta_smart_colloc_freq.json","w", encoding = "utf-8") as f:
    json.dump(log_collocations_vs_freq, f, indent = 4, ensure_ascii = False)
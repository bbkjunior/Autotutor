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

    "overall_colloc_text": [
        {
            "0": [
                [
                    0,
                    "���� ���������"
                ],

    "collocation_index_list": [ lemma, unigr_freq,lyashevskaya_freq_dict[lemma]
                [
                    0,
                    [
                        "���� ���������",
                        318.5,
                        176.85
                    ]
                ],

    "collocation_vectors_list": [
                [
                    0,
                    [
                        "���� ���������",
                        318.5,
                        176.85
                    ],
                    [
                        [
                            0.06343774311244488,
                            0.06113103777170181,
                            0.11051688343286514,
                            0.019039432518184185,
                            0.0266796313226223,
                
    return text_map


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
                    if len(local_corpora_freq_list) > 0:
                        local_corpora_mean = mean(local_corpora_freq_list)
                    else:
                        local_corpora_mean = None
                    if len(freq_list) > 0:
                        freq_mean = mean(freq_list)
                    else:
                        freq_mean = None
                    handled_words_indexes.extend(sub_ind)
                    sentence_collected_collocation[sub_ind[0]] = (ngramm, local_corpora_mean,freq_mean)
                    if debug:print("sentence_collected_collocation", sentence_collected_collocation)
            if debug:print(ngramm)
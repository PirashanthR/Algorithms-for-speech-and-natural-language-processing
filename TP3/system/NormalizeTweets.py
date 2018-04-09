#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Algorithms for Speech and NLP TD 3 -- Pirashanth Ratnamogan
Fonction qui normalise les tweets
"""

from ContextualInformationComponents import ProposeWordForContext,ReadModelC2Vec
from FormalSimilarityComponents import string_kernel,DamerauLevenshteinDistance,LevenshteinDistance
from GloveEmbedding import ReadGlovePretrained
from Utils import dict_acronym
import re
import string
import numpy as np
import sys 
from sklearn.metrics.pairwise import cosine_similarity

reload(sys)  
sys.setdefaultencoding('utf8')


punctuations = ('['+string.punctuation+']')
unvariant_data = string.punctuation +'RT' +'rt'
list_to_remove = ['http',r"\x"]
VectorizedLevenshteinDistance = np.vectorize(LevenshteinDistance)

def NormalizeTweets(tweets,c2vec_model_param_file,c2vec_nresult = 1000,GLOVE_DIR=None,keep_tweet_specificity = True,subseq_length=3,eps_embedding_cosine_distance_correct=3e-1,decay_param_kernel=0.1):
    '''
    Fonction qui normalise un ensemble de tweets
    Param: @tweets: (list) liste des tweets a normaliser
    @c2vec_model_param_file: (str) chemin vers le modele c2vec pretrained
    @c2vec_nresult: (int) nombre de proposition c2vec
    @GLOVE_DIR: (str) chemin vers le glove twitter pretrained
    @keep_tweet_specificity: (bool) Enlever ou non les # et @
    @eps_embedding_cosine_distance_correct: (float) param pour utiliser les words vectors
    @subseq_length: (int) parametre string kernel
    @decay_param_kernel: (float) parametre string kernel
    Return: Les tweets normalisées
    '''
    normalized_tweets  = []
    model,w,word2index,index2word = ReadModelC2Vec(c2vec_model_param_file)
    if GLOVE_DIR != None:
        embedding_index = ReadGlovePretrained(GLOVE_DIR)
    for tweet in tweets:
        global punctuations
        global unvariant_data
        tweet_inter = tweet.lower()
        tweet_inter= " ".join(filter(lambda x:x[:4]!='http', tweet_inter.split()))
        tweet_inter = tweet_inter.encode('ascii','ignore') #remove emoji
        tokens = re.findall(r"[\w']+|"+ punctuations, tweet_inter)
        tokens_normalized =[]
        next_token_skip= False
        token_for_context2vec= list(tokens)
        for index_token,token in enumerate(tokens):
            if token in dict_acronym:
                tokens_normalized.extend(dict_acronym[token].split())
                token_for_context2vec = token_for_context2vec[:index_token] + dict_acronym[token].split() + token_for_context2vec[index_token+1:]
                continue
            if (token in unvariant_data) or (next_token_skip):
                if next_token_skip:
                    next_token_skip= False
                    if not(keep_tweet_specificity):
                        continue
                if (token=='@') or (token=='#'):
                   next_token_skip=True
                   if not(keep_tweet_specificity):
                        continue
                tokens_normalized.append(token)
                continue
            
            list_tokens_replace = list(token_for_context2vec)
            list_tokens_replace[index_token] = '[]'
            context_token = ' '.join(list_tokens_replace)

            proposed_word_c2vec = list(np.array(ProposeWordForContext(model,w,word2index,index2word,context_token,c2vec_nresult))[:,0])
            score_lev_proposed_word_c2vec = VectorizedLevenshteinDistance(token,proposed_word_c2vec)
            min_lev_dist = np.min(score_lev_proposed_word_c2vec)
            all_indices_lev= np.where(np.array(score_lev_proposed_word_c2vec) == min_lev_dist)
            
            len_words = [len(prop)-1 for prop in proposed_word_c2vec]
            subseq_length_to_use = min([subseq_length]+len_words+[len(token)-1])
            if len(token)<3:
                proposed_word_kernel = []
            elif subseq_length_to_use<2:
                subseq_length_to_use = 2
                word_for_kernel_string = [word for word in proposed_word_c2vec if len(word)>2]
                score_kernel_string = string_kernel([token],subseq_length_to_use,decay_param_kernel,word_for_kernel_string)
                max_kernel_score = np.max(score_kernel_string)
                all_indices_kernel= np.where(np.array(score_kernel_string[0]) == max_kernel_score)
                proposed_word_kernel =list(np.array(word_for_kernel_string)[list(all_indices_kernel[0])])
            else:
                score_kernel_string = string_kernel([token],subseq_length_to_use,decay_param_kernel,proposed_word_c2vec)
                max_kernel_score = np.max(score_kernel_string)
                all_indices_kernel= np.where(np.array(score_kernel_string[0]) == max_kernel_score)
                proposed_word_kernel = list(np.array(proposed_word_c2vec)[list(all_indices_kernel[0])])
            potential_token = list(np.array(proposed_word_c2vec)[list(all_indices_lev[0])]) +  proposed_word_kernel
            
            if ((min_lev_dist<=2)and(GLOVE_DIR != None) and(token in embedding_index)):
                #if not in embedding index likely that the word is mispelled
                vec_1 = embedding_index[token]
                fill_token = False
                for i_pot_token in potential_token:
                    if i_pot_token in embedding_index:
                        vec_2 = embedding_index[i_pot_token]
                        cos_sim = cosine_similarity(vec_1.reshape(1, -1),vec_2.reshape(1, -1))
                        if cos_sim>(1-eps_embedding_cosine_distance_correct):
                            tokens_normalized.append(i_pot_token)
                            token_for_context2vec[index_token]= i_pot_token
                            fill_token = True
                            break
                if not(fill_token):
                    if c2vec_nresult>500: #I suppose that if the number of outcome outputed by c2vec is high
                        #then it is unlikely that it doesn't appear in the context so I put the most
                        #probable one in 
                        tokens_normalized.append(proposed_word_c2vec[np.argmin(score_lev_proposed_word_c2vec)])
                    else:
                        tokens_normalized.append(token)
                    
                            
            else:
                if (min_lev_dist>2): #not in the embedding index matrix so we will take the closest one as correction but
                    token_best_context = token  #if the closest is too far we will keep the original content
                else:
                    token_best_context = proposed_word_c2vec[np.argmin(score_lev_proposed_word_c2vec)]
                    
                tokens_normalized.append(token_best_context)
            
        
        processed_tweet = ' '.join(tokens_normalized)
        processed_tweet=processed_tweet.replace(' @ ',' @')
        processed_tweet=processed_tweet.replace(' # ',' #')
        processed_tweet=processed_tweet.replace(' . . ','.. ')
        processed_tweet=processed_tweet.replace(' . .','..')
        processed_tweet=processed_tweet.replace(' . . .','...')
        processed_tweet=processed_tweet.replace(' . . . ','... ')
        processed_tweet=processed_tweet.replace(' .','.')
        processed_tweet=processed_tweet.replace(' !','!')
        processed_tweet=processed_tweet.replace(' ?','?')
        processed_tweet=processed_tweet.replace(' "','"')
        processed_tweet=processed_tweet.replace(' ]',']')
    
        for punct in punctuations:
            processed_tweet=processed_tweet.replace(' '+punct+' ',punct+' ')
        normalized_tweets.append(processed_tweet)
    return normalized_tweets
    
    
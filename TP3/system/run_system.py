#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Algorithms for Speech and NLP TD 3 -- Pirashanth Ratnamogan
Fonction qui lance tout le systeme
"""
import sys
from Utils import read_corpus,write_corpus
from NormalizeTweets import NormalizeTweets

#Gestion un peu brouillonne des arguments
corpus = sys.argv[1]
model_param_file = sys.argv[2]

if len(sys.argv)>4:
    nb_result_c2vec = int(sys.argv[3])
    output_path = sys.argv[4]
else:
    nb_result_c2vec = 1000
    output_path = None


if len(sys.argv)>6:
    glove_dir = sys.argv[6]
    if glove_dir == 'None':
        glove_dir = None
else:
    glove_dir = None

if len(sys.argv)>5:
    keep_tweet_specificity = bool(sys.argv[5])
else:
    keep_tweet_specificity = True
    
        
full_tweets = read_corpus(corpus)
normalized_tweets = NormalizeTweets(full_tweets,model_param_file,nb_result_c2vec,glove_dir,keep_tweet_specificity)
if output_path==None: 
    print('**********List of normalized tweets**********')
    lines = '\n'.join(normalized_tweets)
    print(lines)
else:
    lines = '\n'.join(normalized_tweets)
    print(lines)
    write_corpus(output_path,normalized_tweets)




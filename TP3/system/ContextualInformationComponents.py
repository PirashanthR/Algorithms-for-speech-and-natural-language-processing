#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Algorithms for Speech and NLP TD 3 -- Pirashanth Ratnamogan
Ce fichier contient toutes les fonctions que l'on doit appeler pour utiliser context2vec 
"""
#!/usr/bin/env python
import numpy
import six
import sys
import traceback
import re

from chainer import cuda
from context2vec.common.context_models import Toks
from context2vec.common.model_reader import ModelReader


target_exp = re.compile('\[.*\]')

gpu = -1 # todo: make this work with gpu

if gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(gpu).use()    
xp = cuda.cupy if gpu >= 0 else numpy


def parse_input(line):
    sent = line.strip().split()
    target_pos = None
    for i, word in enumerate(sent):
        if target_exp.match(word) != None:
            target_pos = i
            if word == '[]':
                word = None
            else:
                word = word[1:-1]
            sent[i] = word
    return sent, target_pos
    

def mult_sim(w, target_v, context_v):
    target_similarity = w.dot(target_v)
    target_similarity[target_similarity<0] = 0.0
    context_similarity = w.dot(context_v)
    context_similarity[context_similarity<0] = 0.0
    return (target_similarity * context_similarity)
#model_param_file = r'/home/ratnamogan/Documents/Speech_and_NLP/MVA_2018_SL/TD_#3/context2vec.ukwac.model.package/context2vec.ukwac.model.params'


def ReadModelC2Vec(model_param_file):
    '''
    Fonction qui lit un modèle context2vec préentrainé
    Param: @model_param_file: (str) chemin vers le .param du modèle préentrainé
    Return: Les objets permettant d'utiliser context2vec
    '''
    model_reader = ModelReader(model_param_file)
    w = model_reader.w
    word2index = model_reader.word2index
    index2word = model_reader.index2word
    model = model_reader.model
    return model,w,word2index,index2word

def ProposeWordForContext(model,w,word2index,index2word,line,n_result):    
    '''
    Fonction qui propose des mots étant donné le context
    Param: @model,w,word2index,index2word: objets du modele C2vec
    @line: ligne avec [] a la place du mot recherche
    @n_result: nombre de tops résultats a renvoyer
    Return: Une liste contenant les mots proposés et le score de compatibilité au contexte associé
    '''
    sent, target_pos = parse_input(line)
    if sent[target_pos] == None:
        target_v = None
    else:
        target_v = w[word2index[sent[target_pos]]]
        
    if len(sent) > 1:
        context_v = model.context2vec(sent, target_pos) 
        context_v = context_v / xp.sqrt((context_v * context_v).sum())
    else:
        context_v = None
        
    if target_v is not None and context_v is not None:
        similarity = mult_sim(w, target_v, context_v)
    else:
        if target_v is not None:
            v = target_v
        elif context_v is not None:
            v = context_v                
        else:
            print("Error")            
        similarity = (w.dot(v)+1.0)/2 # Cosine similarity can be negative, mapping similarity to [0,1]
    
        count = 0
        list_to_return = []
        for i in (-similarity).argsort():
            if numpy.isnan(similarity[i]):
                continue
            list_to_return.append([index2word[i], similarity[i]])
            #print('{0}: {1}'.format(index2word[i], similarity[i]))
            count += 1
            if count == n_result:
                break
    return list_to_return
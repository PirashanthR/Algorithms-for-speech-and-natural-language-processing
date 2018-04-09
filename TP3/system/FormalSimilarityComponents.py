#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Algorithms for Speech and NLP TD 3 -- Pirashanth Ratnamogan
Ce fichier contient toutes les fonctions que l'on utilise pour la "formal similarity"

LevenshteinDistance
DamerauLevenshteinDistance
string_kernel
"""
import numpy as np

def LevenshteinDistance(str1,str2):
    '''
    Fonction qui calcule la levenshtein distance entre str1 et str2
    Param: @(str1): (str) string 1 pour la comparaison
    @(str2): (str) string 2 pour la comparaison
    Return (int) distance
    '''
    len_s1 = len(str1) +1
    len_s2 = len(str2) +1
    m = np.zeros((len_s1,len_s2))
    for i in range(len_s1):
        m[i,0] = i
    
    for j in range(len_s2):
        m[0,j] = j
    
    for i in range(1,len_s1):
        for j in range(1,len_s2):
            if str1[i-1]==str2[j-1]:
                m[i,j]= min(m[i-1,j]+1,m[i,j-1]+1,m[i-1,j-1])
            else:
                m[i,j] =min(m[i-1,j]+1,m[i,j-1]+1,m[i-1,j-1]+1)
    return m[-1,-1]

def DamerauLevenshteinDistance(str1,str2):
    '''
    Fonction qui calcule la Damereau levenshtein distance entre str1 et str2
    Param: @(str1): (str) string 1 pour la comparaison
    @(str2): (str) string 2 pour la comparaison
    Return (int) distance
    '''
    len_s1 = len(str1) +1
    len_s2 = len(str2) +1
    m = np.zeros((len_s1,len_s2))
    for i in range(len_s1):
        m[i,0] = i
    
    for j in range(len_s2):
        m[0,j] = j
    
    for i in range(1,len_s1):
        for j in range(1,len_s2):
            if str1[i-1]==str2[j-1]:
                m[i,j]= min(m[i-1,j]+1,m[i,j-1]+1,m[i-1,j-1])
                if ((i>1)and(j>1)and(str1[i-1]==str2[j-2])and(str1[i-2]==str2[j-1])):
                    m[i,j] = min(m[i,j],m[i-2,j-2])
            else:
                m[i,j] =min(m[i-1,j]+1,m[i,j-1]+1,m[i-1,j-1]+1)
                if ((i>1)and(j>1)and(str1[i-1]==str2[j-2])and(str1[i-2]==str2[j-1])):
                    m[i,j] = min(m[i,j],m[i-2,j-2]+1)

    return m[-1,-1]

#inspired from https://github.com/timshenkao/StringKernelSVM/blob/master/stringSVM.py

def K(n, s, t,decay_param):
    smallest_size= min(len(s), len(t)) 
    if smallest_size< n:
        return 0
    else:
        part_sum = 0
        for j in range(1, len(t)):
            if t[j] == s[-1]:
                #not t[:j-1] as in the article but t[:j] because of Python slicing rules!!!
                part_sum += K1(n - 1, s[:-1], t[:j],decay_param)
        result = K(n, s[:-1], t,decay_param) + decay_param ** 2 * part_sum
        return result


def K1(n, s, t,decay_param):
    if n == 0:
        return 1
    elif min(len(s), len(t)) < n:
        return 0
    else:
        part_sum = 0
        for j in range(1, len(t)):
            if t[j] == s[-1]:
    #not t[:j-1] as in the article but t[:j] because of Python slicing rules!!!
                part_sum += K1(n - 1, s[:-1], t[:j],decay_param) * (decay_param ** (len(t) - (j + 1) + 2))
        result = decay_param * K1(n, s[:-1], t,decay_param) + part_sum
        return result

gram_matrix_elem = lambda str1,str2,sdkval1,sdkval2,subseq_length,decay_param: 1 if str1==str2 else \
K(subseq_length, str1, str2,decay_param) / (sdkval1 * sdkval2) ** 0.5

def string_kernel(X1,subseq_length,decay_param,X2=[]):
    """
    Calcul de la matrice de gram du string kernel 
    param @X1: (list) list of string for the comparison
    param @X2: (list) list of string for the comparison
    @subseq_length: paramètre de subsequence a considérer
    @decay_param:paramètre lambda de l'article 
    voir http://www.jmlr.org/papers/volume2/lodhi02a/lodhi02a.pdf
    return: La matrice de gram
    """
    len_X2= len(X2)
    len_X1 = len(X1)
    sim_docs_kernel_value = {}

    if len_X2 ==0:
        # numpy array of Gram matrix
        gram_matrix = np.zeros((len_X1, len_X1), dtype=np.float32)

        #when lists of documents are identical
        #store K(s,s) values in dictionary to avoid recalculations
        for i in range(len_X1):
            sim_docs_kernel_value[i] = K(subseq_length, X1[i], X1[i],decay_param)
            #calculate Gram matrix
        for i in range(len_X1):
            for j in range(i, len_X1):
                gram_matrix[i, j] = gram_matrix_elem(X1[i], X1[j], sim_docs_kernel_value[i],\
                           sim_docs_kernel_value[j],subseq_length,decay_param)
                #using symmetry
                gram_matrix[j, i] = gram_matrix[i, j]
        return gram_matrix
    else:
        gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)

        sim_docs_kernel_value[1] = {}
        sim_docs_kernel_value[2] = {}
        #store K(s,s) values in dictionary to avoid recalculations
        for i in range(len_X1):
            sim_docs_kernel_value[1][i] = K(subseq_length, X1[i], X1[i],decay_param)
        for i in range(len_X2):
            sim_docs_kernel_value[2][i] = K(subseq_length, X2[i], X2[i],decay_param)
        
        for i in range(len_X1):
            for j in range(len_X2):
                gram_matrix[i, j] = gram_matrix_elem(X1[i], X2[j], sim_docs_kernel_value[1][i],sim_docs_kernel_value[2][j],subseq_length,decay_param)
        
        return gram_matrix


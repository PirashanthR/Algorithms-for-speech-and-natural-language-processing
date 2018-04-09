#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Algorithms for Speech and NLP TD 3 -- Pirashanth Ratnamogan
Contient la lecture du fichier de glove embedding pretrained
"""

import os
import numpy as np

def ReadGlovePretrained(GLOVE_DIR):
    '''
    Lecture du fichier glove
    Param @GLOVE_DIR: contient le chemin vers le fichier glove pretrained twitter
    Return le dictionnaire qui associe mots à leurs representation
    '''
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.twitter.27B.200d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    print('Found %s word vectors.' % len(embeddings_index))
    
    return embeddings_index


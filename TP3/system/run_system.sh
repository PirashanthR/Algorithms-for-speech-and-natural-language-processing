#!/bin/sh

#assume context to vec dir in $CONTEXT2VECDIR

CORPUS=$1
#Corpus dans le paramètre du .sh

#CONTEXT2VECDIR="/home/ratnamogan/Documents/Speech_and_NLP/MVA_2018_SL/TD_#3/context2vec.ukwac.model.package/context2vec.ukwac.model.params"
#On suppose que CONTEXT2VECDIR est assigné préalablement

GLOVE_DIR=$2
#GLOVE_DIR="/home/ratnamogan/Documents/Speech_and_NLP/MVA_2018_SL/TD_#3/glove.twitter.27B"
#On peut mettre le chemin vers le pretrained glove dir pour amélirer le résultat

NB_RES_C2VEC="1000"
OUTPUT_PATH="./normalized_tweets.txt"
KEEP_TWEET_SPEC="True"

#ligne a lancer
python ./run_system.py $CORPUS $CONTEXT2VECDIR $NB_RES_C2VEC $OUTPUT_PATH $KEEP_TWEET_SPEC $GLOVE_DIR





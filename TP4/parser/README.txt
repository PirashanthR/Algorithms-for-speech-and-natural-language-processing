Algorithms for Speech and NLP TD 4: Pirashanth RATNAMOGAN

Ce dossier contient un dossier system qui continent toutes les fonctions qui permettent de lancer mon application qui permet de Parser des phrases.

Exemple: Entrée: 
J'essaye cette phrase.   -------> DEVIENT
( (SENT (VN (CLS J') (V essaye)) (NP (DET cette) (NC phrase))) (PONCT .)))

Le code a été fait en python 3.6.

Pratiquement tout a été codé de zéro, ainsi les librairies présentes nltk et enchant sont là uniquement en complément pour améliorer mon résultat mais le code pourrait facilement tourner sans ces librairies.

Afin de lancer le code les libraires suivantes sont nécessaires: 
re
string
numpy
nltk
enchant
os
sys

Pour lancer l'outil il faut utiliser le script shell
run_parser.sh

Il prend obligatoirement en entrée
Argument 1: Chemin vers le corpus d'entrainement (SEQUOIA)
Argument 2: Chemin vers les phrases à parser 

En option on peut ajouter les arguments suivants: 
Argument 3: True ou False -> utiliser ou non l'outil d'enchant d'autocorrection

Les arguments suivant sont nécessaire si l'on veut s'appuyer sur l'outil de PosTagging de stanford pour gérer les mots inconnues: https://nlp.stanford.edu/software/tagger.shtml#Download.

Les arguments suivants sont également en option.
Argument 4: Chemin vers l'executable java
Argument 5: Chemin vers le jar du stanford postagger full (voir exemple pour que ca soit clair (à télécharger)
Argument 6: Chemin vers le model du stanford  postagger full (voir exemple pour que ca soit clair (à télécharger)

Exemple de lancement complet: 
./run_parser.sh /home/ratnamogan/Documents/Speech_and_NLP/MVA_2018_
SL/TD_#4/sequoia-corpus+fct.mrg_strict /home/ratnamogan/Documents/Speech_and_NLP/MVA_2018_SL/TD_#4/Fichieralire.txt True /usr/bin/java /home/ratnamogan/Documents/Speech_and_NLP/MVA_2018_SL/TD_#4/stanford-postagger-full-2017-06-09/stanford-postagger-3.8.0.jar /home/ratnamogan/Documents/Speech_and_NLP/MVA_2018_SL/TD_#4/stanford-postagger-full-2017-06-09/models/french.tagger 

Exemple de lancement sans les options:
./run_parser.sh /home/ratnamogan/Documents/Speech_and_NLP/MVA_2018_
SL/TD_#4/sequoia-corpus+fct.mrg_strict /home/ratnamogan/Documents/Speech_and_NLP/MVA_2018_SL/TD_#4/Fichieralire.txt



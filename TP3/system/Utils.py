#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Algorithms for Speech and NLP TD 3 -- Pirashanth Ratnamogan
Outils pour le projet
"""

#build the dictionnary manually from this list: http://www.smart-words.org/abbreviations/internet-acronyms.png

dict_acronym = {}

dict_acronym['2f4u'] = 'too fast for you'
dict_acronym['4yeo'] = 'for your eyes only'
dict_acronym['aamof'] = 'as a matter of fact'
dict_acronym['ack'] = 'acknoldgement'
dict_acronym['afaik'] = 'as far as i know'
dict_acronym['afair'] = 'as far as i remember'
dict_acronym['afk'] = 'away from keyboard'
dict_acronym['aka'] = 'also known as'
dict_acronym['asap'] = 'as soon as possible'
dict_acronym['b2k'] = 'back to keyboard'
dict_acronym['btt'] = 'back to topic'
dict_acronym['btw'] = 'by the way'
dict_acronym['b/c'] = 'because'
dict_acronym['c&p'] = 'copy and paste'
dict_acronym['cu'] = 'see you'
dict_acronym['cys'] = 'check your settings'
dict_acronym['diy'] = 'do it yourself'
dict_acronym['eobd'] = 'end of business dayf'
dict_acronym['eod'] = 'end of discussion'
dict_acronym['eom'] = 'end of message'
dict_acronym['eot'] = 'end of thread'
dict_acronym['faq'] = 'frequently asked questions'
dict_acronym['fack'] = 'full acknowledge'
dict_acronym['fka'] = 'formerly known as'
dict_acronym['fwiw'] = 'for what it\'s worth'
dict_acronym['fyi'] = 'for your information'
dict_acronym['hf'] = 'have fun'
dict_acronym['hth'] = 'hope this helps'
dict_acronym['iirc'] = 'if i recall'
dict_acronym['imo'] = 'in my opinion'
dict_acronym['imnsho'] = 'in my not so humble opinon'
dict_acronym['iow'] = 'in other words'
dict_acronym['itt'] = 'in this thread'
dict_acronym['lol'] = 'laughing out loud'
dict_acronym['mmw'] = 'mark my words'
dict_acronym['nan'] = 'not a number'
dict_acronym['nntr'] = 'no need to reply'
dict_acronym['noob'] = 'newbie'
dict_acronym['noyb'] = 'none of your business'
dict_acronym['nrn'] = 'no reply necessary'
dict_acronym['omg'] = 'oh my god'
dict_acronym['op'] = 'original poster'
dict_acronym['ot'] = 'off topic'
dict_acronym['otoh'] = 'on the other hand'
dict_acronym['pebkac'] = 'problem exists between keyboard and chair'
dict_acronym['pov'] = 'point of view'
dict_acronym['ppl'] = 'people'
dict_acronym['rotfl'] = 'rolling on the floor laughing'
dict_acronym['rsvp'] = 'please reply'
dict_acronym['rtfm'] = 'read the fine manual'
dict_acronym['scnr'] = 'sorry could not resist'
dict_acronym['sflr'] = 'sorry for late reply'
dict_acronym['spoc'] = 'single point of contact'
dict_acronym['tba'] = 'to be annonced'
dict_acronym['tbc'] = 'to be continued'
dict_acronym['tia'] = 'thanks in advance'
dict_acronym['thx'] = 'thanks'
dict_acronym['tnx'] = 'thanks'
dict_acronym['tq'] = 'thank you'
dict_acronym['tyvm'] = 'thank you very much'
dict_acronym['tyt'] = 'take your time'
dict_acronym['ttyl'] = 'talk to you later'
dict_acronym['w00t'] = 'whoomp, there it is'
dict_acronym['wfm'] = 'works for me'
dict_acronym['wrt'] = 'with regard to'
dict_acronym['wth'] = 'what the hell'
dict_acronym['ymmd'] = 'you made my day'
dict_acronym['ymmv'] = 'you mileage may vary'
dict_acronym['yam'] = 'yet another meeting'



def read_corpus(Path_corpus):
    with open(Path_corpus)  as f:
        content = f.readlines()
    content = [x.strip() for x in content] 
    return content

def write_corpus(Path_corpus,tweets):
    lines = '\n'.join(tweets)
    with open(Path_corpus,'w')  as f:
        f.write(lines)

    
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 04 20:15:48 2016

@author: Fianna
"""

import os
import random
import collections
import nltk
from itertools import combinations

class Language:
    def __init__(self, tag):
        self.tag = tag
        self.training_sents, self.testing_sents = self.get_texts()
    
    def get_texts(self): 
        t_sents = open(os.path.join("ICNALE_editted",(self.tag+".txt"))).read().split('\n')    
        tagged_sents = [[tuple(w.split('|')) for w in sent.split(' ')] for sent in t_sents]    
        random.shuffle(tagged_sents)
        training = tagged_sents[:2181]
        testing = tagged_sents[2181:2727]
        return training, testing

class LangClassifier(Language):
    def __init__(self, lang1, lang2, pos = True, words = True):
        self.lang1 = Language(lang1)
        self.lang2 = Language(lang2)
        self.feat_pos = pos
        self.feat_words = words
        self.classifier = self.get_classifier()
    
    def features(self, sent):
        feature_dict = collections.defaultdict(float)
        for n in range(1,5):
            if self.feat_pos == True:
                pos = [t for w,t in sent]
                for gram in nltk.ngrams(pos, n):
                    feature_dict['n=%d_pos=%s'%(n,gram)] = True
            else:
                pass
            if self.feat_words == True:
                wor = [w for w,t in sent]
                for gram in nltk.ngrams(wor, n):
                    feature_dict['n=%d_words=%s'%(n,gram)] = True
            else:
                pass
        return feature_dict

    def get_classifier(self):
        def traintag(lang):
            return [(self.features(sent), lang.tag) for sent in lang.training_sents]
        training_set = traintag(self.lang1)+traintag(self.lang2)
        classifier = nltk.classify.NaiveBayesClassifier.train(training_set)
        return classifier
    
    def test(self):
        def testtag(lang):
            return [(self.features(sent), lang.tag) for sent in lang.testing_sents]
        testing_set = testtag(self.lang1)+testtag(self.lang2)
        return nltk.classify.accuracy(self.classifier, testing_set)
        

tag_set = [fname[:3] for fname in os.listdir('ICNALE_editted')]
tag_pairs = list(combinations(tag_set,2))
results = {}
for (l1,l2) in tag_pairs:
    results['%s_%s'%(l1,l2)] = {'words_and_POS':LangClassifier(l1,l2).test(), 'words':LangClassifier(l1,l2, pos=False).test(), 'POS':LangClassifier(l1,l2, words=False).test()}
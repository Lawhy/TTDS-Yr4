import argparse
import re
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import json
import pickle
from nltk.stem import PorterStemmer
from collections import defaultdict, OrderedDict


class Lexicon:
    """ A class that stores a collection of processed documentsl
        and provides the methods of processing"""
    
    def __init__(self, xml_path, stop_words_path, stemmer):
        self.docs_df = pd.DataFrame(columns=['doc'])
        self.stop_words = []  
        self.stemmer = stemmer
        self.index = defaultdict(lambda: defaultdict(list))  # positional inverted index
        self.load_xml(xml_path)
        self.load_stop_words(stop_words_path)
        
        
    def load_xml(self, file):
        """parse the XML file in standard TREC

        Args:
            file (str): path to the xml file

        Returns:

        """

        # Reading file
        with open(file, 'r', encoding='utf-8') as f:
            xml = f.read()

        # Add a root tag
        xml = '<ROOT>' + xml + '</ROOT>'

        # build a dataframe to store each doc
        print('Loading the XML file...')
        for doc in ET.fromstring(xml):
            docID = doc.find('DOCNO').text
            # doc contains both HEADLINE and TEXT
            content = doc.find('HEADLINE').text + doc.find('TEXT').text
            self.docs_df.loc[docID] = content
        print('Finished. ')
            
            
    def load_stop_words(self, file):
        with open(file, 'r', encoding='utf-8-sig') as f:
            # remove \n for each stop word
            self.stop_words = [w.replace('\n', '') for w in f.readlines()]

    
    @staticmethod
    def tokenize(text):
        """tokenize the input text (split at non word character)

        Args:
            text (str): the input text

        Returns:
            tokens

        """
        pa = r'\w+'
        return re.findall(pa, text)
            

    def preprocess(self, text):
        """preprocess the text:
           1. tokenization; 
           2. case folding;
           3. stopwords cleaning
           3. normalisation (stemming).

        Args:
            text (str): the input text 

        Returns:
            [preprocessed tokens]

        """
        # load the Porter Stemmer from nltk
        tokens = [self.stemmer.stem(t.lower()) for t in self.tokenize(text) \
                        if not t.lower() in self.stop_words]
        
        return tokens
    
    
    def pos_inv_ind(self):
        """build the positional inverted index"""
        
        print('Building the Positional Inverted Index...')
        for i, dp in self.docs_df.iterrows():
            doc = dp['doc']
            tokens = self.preprocess(doc)
            for pos in range(len(tokens)):
                """term:
                       docID: pos1, pos2, ...
                       docID: pos1, pos2, ...
                """ 
                self.index[tokens[pos]][int(i)] += [pos + 1]  # position starts from 1
        # sort the index by keys
        self.index = OrderedDict(sorted(self.index.items()))
        print('Finished.') 
        
        
    def export_index(self):
        """export positional inverted index into readable format"""
        
        with open('index.txt', 'w+', encoding='utf-8') as f:
            for term in self.index.keys():
                    f.write(term + ':\n')
                    _dict_ = self.index[term]
                    for docID in _dict_.keys():
                        pos_list = _dict_[docID]
                        f.write('\t' + str(docID) + ': '\
                                + ','.join(str(pos) for pos in pos_list)\
                                + '\n')
                    f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, help="the path to the xml file")
    parser.add_argument("--st", type=str, help="the path to the stop words list")
    args = parser.parse_args()
    
    lexicon = Lexicon(args.xml, args.st, PorterStemmer())
    lexicon.pos_inv_ind()
    with open('index.pkl', 'wb') as f:
        pickle.dump(lexicon.index, f)
    lexicon.export_index()

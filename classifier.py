"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2017

            **  Headline Classifier  **

Auther(s): Amit Nelinger, Omer Dolev, Netai Benaim, Dan Amir

===================================================
"""
import pandas
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from random import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pickle

NUM_WORDS = ['two','three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
NUM_DIGITS = [' 2 ', ' 3 ', ' 4 ', ' 5 ', ' 6 ', ' 7 ', ' 8 ', ' 9 ', ' 10 ']

class Classifier(object):

    def classify(self, X):
        """
        Recieves a list of m unclassified headlines, and predicts for each one which newspaper published it.
        :param X: A list of length m containing the headlines' texts (strings)
        :return: y_hat - a binary vector of length m
        """
        n_chars = np.array(self.num_of_chars(X))
        n_dots = np.array(self.num_of_dots(X))
        n_style = np.array(self.nums_style(X))
        bag_matrix = self.test_bag_of_wordify(X, self.load_obj('vocabulary'))
        bag_matrix = bag_matrix.todense()
        data = np.column_stack((bag_matrix, n_chars, n_dots, n_style))
        clf = joblib.load('svm_model.pkl')
        preds = clf.predict(data).tolist()
        preds = [int(y) for y in preds]
        return preds

    def num_of_chars(self, title_set):
        chars_count = [len(line) for line in title_set]
        return chars_count

    def num_of_dots(self, title_set):
        word_count = [line.count('.') for line in title_set]
        return word_count

    def nums_style(self, title_set):
        nums_list = list()
        for line in title_set:
            nums_words = False
            nums_dig = False
            for num in NUM_WORDS:
                if line.find(num) > -1:
                    nums_words = True
                    break
            for num in NUM_DIGITS:
                if line.find(num) > -1:
                    nums_dig = True
            if nums_words and (not nums_dig):
                nums_list.append(1)
            elif (not nums_words) and nums_dig:
                nums_list.append(-1)
            else:
                nums_list.append(0)
        return nums_list

    def load_obj(self, name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def test_bag_of_wordify(self,test_set, vocab, idx=0, min_d = 0.001, max_d = 1.0, ng_rang = (1,3)):
        count_vect = CountVectorizer(binary='true',
                                 vocabulary = vocab, ngram_range=ng_rang, max_df=max_d, min_df=min_d)
        X_bag = count_vect.transform(test_set)
        return X_bag

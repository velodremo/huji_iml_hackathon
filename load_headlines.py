import pandas
import classifier
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from random import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pickle
import math
import scipy as sp
NUM_WORDS = ['two','three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
NUM_DIGITS = [' 2 ', ' 3 ', ' 4 ', ' 5 ', ' 6 ', ' 7 ', ' 8 ', ' 9 ', ' 10 ']
# Outputs the headlines dataset.
# X is a list of headlines
# y is a list of binary labels, 0 stands for Haaretz and 1 for Israel Hayom
def load_dataset(filenames=['haaretz.csv','israelhayom.csv']):
    cur_y = 0
    X = pandas.DataFrame()
    y = np.empty(0,dtype=np.int32)
    for filename in filenames:
        train_cur = pandas.read_csv(filename, header=None)
        X = pandas.concat([X,train_cur[0]])
        y = np.append(y,cur_y*np.ones(len(train_cur),dtype=np.int32))
        cur_y += 1
    X = [x[0] for x in X.values.tolist()]
    y = y.tolist()
    return X,y

def load_into_datasets():
    POS_h = pandas.read_csv('pos_tags_haarz.csv', header=None)
    POS_h = [x[0] for x in POS_h.values.tolist()]
    POS_i = pandas.read_csv('pos_tags_israel.csv', header=None)
    POS_i = [x[0] for x in POS_i.values.tolist()]
    Xh = pandas.read_csv('haaretz.csv', header=None)
    Xh = [x[0] for x in Xh.values.tolist()]
    Xh = list(zip(Xh, POS_h))
    Xi = pandas.read_csv('israelhayom.csv', header=None)
    Xi = [x[0] for x in Xi.values.tolist()]
    Xi = list(zip(Xi, POS_i))


    shuffle(Xh)
    shuffle(Xi)
    Xh_train = Xh[:2300]
    Xi_train = Xi[:2300]
    Xh_test = Xh[2300:2700]
    Xi_test = Xi[2300:2700]


    Xh_train = [(x, 1) for x in Xh_train]
    Xi_train = [(x, 0) for x in Xi_train]
    Xh_test = [(x, 1) for x in Xh_test]
    Xi_test = [(x, 0) for x in Xi_test]

    X_train = Xh_train + Xi_train
    X_test = Xh_test + Xi_test
    shuffle(X_test)
    shuffle(X_train)
    Y_test = [x[1] for x in X_test]
    X_test = [x[0] for x in X_test]
    Y_train = list(map(lambda x: x[1], X_train))
    X_train = list(map(lambda x: x[0], X_train))




    return np.column_stack((X_train, Y_train)), np.column_stack((X_test,
                                                                 Y_test))

def print_set(dataset):
    # print (dataset[0])
    # print (dataset[1])
    # for x,y in enumerate(dataset):
    #     print(x,y)
    print(dataset)


def bag_of_wordify(headlines, idx=0, min_d = 0.001, max_d = 1.0, ng_rang = (1,3)):

    count_vect = CountVectorizer(binary='true', ngram_range=ng_rang, min_df=min_d, max_df=max_d) # TODO
    #  stop words
    X_bag = count_vect.fit_transform([x[idx] for x in headlines])
    # print(count_vect.vocabulary_)
    return (X_bag, np.array([x[2] for x in headlines])), \
           count_vect.vocabulary_

def test_bag_of_wordify(test_set, vocab, idx=0, min_d = 0.001, max_d = 1.0, ng_rang = (1,3)):
    count_vect = CountVectorizer(binary='true',
                                 vocabulary = vocab, ngram_range=ng_rang, max_df=max_d, min_df=min_d)
    X_bag = count_vect.transform([x[0] for x in test_set])
    return X_bag, np.array([x[1] for x in test_set])

def num_of_words(title_set):
    word_count = [len(line[0]) for line in title_set]
    return word_count



def classify(train_bag, test_bag):

    # classifier = svm.SVC(cache_size=1000, gamma=0.01, C=10)
    # clf = GridSearchCV(classifier, parameters)
    clf = svm.SVC(cache_size=1000, C=10.0, gamma=0.01)
    clf.fit(train_bag[0], train_bag[1])
    # print(clf.cv_results_)
    # success_count = 0
    # for idx, sample in enumerate(test_bag[0]):
    #     pred = clf.predict(sample[0])
    #     success_count += int(pred == test_bag[1][idx])
    # print ("SUCCESS COUNT =", success_count)
    success_rate = clf.score(test_bag[0], test_bag[1])
    joblib.dump(clf, filename='svm_model.pkl')

    print("SUCCESS RATE =", success_rate)
def num_of_dots(title_set):
    word_count = [line[0].count('.') for line in title_set]
    # word_count = [0 if x < 70 else 1 for x in word_count]
    return word_count

def nums_style(title_set):
    nums_list = list()
    for line in [x[0] for x in title_set]:
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

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

train_set, test_set = load_into_datasets()

train_bag, vocab = bag_of_wordify(train_set)
save_obj(vocab, 'vocabulary')
train_pos, vocab_pos = bag_of_wordify(train_set, idx=1, max_d=1.0, min_d=0.05, ng_rang=(3,5))
train_pos = train_pos[0].todense()
train_num = num_of_words(train_set)
train_dots = num_of_dots(train_set)
train_digits = np.array(nums_style(train_set))


test_num = num_of_words(test_set)
test_digits = np.array(nums_style(test_set))
test_pos = (test_bag_of_wordify(test_set,vocab_pos, idx=1, max_d=1.0, min_d=0.05, ng_rang=(2,5)))
test_pos = test_pos[0].todense()
print("POS: ", train_pos.shape)


# print (train_bag[0].shape, np.array(train_num).shape)
train_y = np.array([x[2] for x in train_set])
test_y = np.array([x[2] for x in test_set])
train_bag = train_bag[0].todense()
print("bag shape: ", train_bag.shape)
train_num = np.asmatrix(train_num).T
# train_bag = (np.column_stack((train_bag, train_num))), train_y

# train_features = (np.column_stack((train_bag[0], np.array(train_num))),
#                   train_bag[1])
train_num = np.array(train_num)
train_dot = np.array(train_dots)
train_data = np.column_stack((train_bag, train_num, train_dot, train_digits))
# train_data = np.column_stack((train_bag, train_data))
# train_data = np.column_stack((train_pos, train_data))
# train_features = (train_num, train_bag[1])

test_bag = test_bag_of_wordify(test_set, vocab)
test_num = np.asmatrix(test_num).T
test_dot = num_of_dots(test_set)
test_data = np.column_stack((test_bag[0].todense(), test_num, test_dot,test_digits))
# test_data = np.column_stack((test_bag[0].todense(), test_data))

# test_bag = np.column_stack((test_bag[0].todense(), test_num)), test_bag[1]
# test_data = np.column_stack((test_pos, test_data))


# test_features = (np.column_stack((test_bag[0], np.array(test_num))),
#                  test_bag[1])

# test_features = (test_num, test_bag[1])
print(len(train_bag[1]))

print("train shape: ",train_data.shape)
print("test shape: ",test_data.shape)
#
classify((train_data, train_y), (test_data, test_y))



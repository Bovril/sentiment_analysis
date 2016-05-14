import os
from collections import Counter
import time

from tqdm import tqdm

from pymongo import MongoClient

from reader.reader import Read
from metrics.metrics import test
from filter.prepocessing import Filter, REGEX_PATTERNS, stemmatize
from classifier.naive_bayes import NB
from train.nb_train import train

client = MongoClient()
db = client.sent
items = db.my_collection

try:
    import cPickle as pickle
except ImportError:
    import pickle

# GLOBAL VARIABLES
# --------------------------------------------------------------------

# Current directory

CWD = os.path.dirname(os.path.realpath(__file__))

test_data = CWD + "/trainingandtestdata/testdata.csv"
tr_data = Read.load_data(CWD + "/trainingandtestdata/training.1600000_posneg.csv")

stop_words = set()

with open(CWD + '/english.stop', 'rb') as stop_w:
    for word in stop_w:
        stop_words.add(word.decode('utf-8').rstrip())

tweet_filter = Filter(
    ngram_combo=[1, 2, 3],
    stop_words=stop_words,
    patterns=REGEX_PATTERNS,
    func=stemmatize
)

START_TIME = time.time()
lista_fajlova = ['kombo_fajl', 'all_words_fajl', 'kategorije_fajl']


def timex():
    """Function for timing between
    :rtype : None
    """
    print (
        "--- {0:.2f} minutes ---".format((time.time() - START_TIME)))


# training_data = CWD + "/semeval2014/semeval_train_dev.tsv"
# test_data = CWD + "/semeval2014/semeval_test.tsv"

# training_data = CWD + "/trainingandtestdata/training.50000.processed.posneg.csv"


class Serializer:
    def __init__(self, lista_fajlova, lista_objekata):
        self.fajlovi = lista_fajlova
        self.objekti = lista_objekata()

    def serialize(self):
        fajlovi_za_serailizaciju = [
            open(file, 'wb') for file in self.fajlovi]
        for podatak, fajl in zip(self.objekti, fajlovi_za_serailizaciju):
            pickle.dump(podatak, fajl)
            fajl.close()

class Deserialize:
    def __init__(self, lista_fajlova):
        self.fajlovi = lista_fajlova

    def deserialize(self):
        fajlovi_za_des = [open(file, 'rb') for file in self.fajlovi]
        return [pickle.load(o) for o in fajlovi_za_des]


def feat_reduction(word_frequency_in_class, all_words, n):
    for word in all_words.keys():
        if sum(word_frequency_in_class[word].values()) <= n:
            del word_frequency_in_class[word]
            del all_words[word]
    print ("uklonjene su reci sa manje od %d ukupno pojavljivanja u svim klasama" % n)
    print ("ukupno unikatnih reci koriscenjih za klasifikaciu: {}".format(len(all_words.keys())))
    return word_frequency_in_class, all_words


timex()

if __name__ == '__main__':
    # print (30*'#')
    # print '############ TRAINING ################'
    # print '######################################'
    # print ngram_broj
    # print " bez lema i stema  "
    # print "semeval2014"

    # ser = Serializer(lista_fajlova, train)
    # timex()
    # ser.serialize()
    # timex()

    des = Deserialize(lista_fajlova)
    word_frequency_in_class, all_words, categories = train(tr_data, tweet_filter) # tqdm(des.deserialize())
    # word_frequency_in_class, all_words, categories = tqdm(des.deserialize())
    print type(categories)
    timex()
    total_words_in_category = Counter()
    ukupno_reci_u_dokumentu = sum(all_words.values())

    word_vector = word_frequency_in_class.keys()
    # print [i for i in word_frequency_in_class.keys()] == word_frequency_in_class.keys()
    for i in tqdm(word_vector):
        for j in categories.keys():
            total_words_in_category[j] += word_frequency_in_class[i][j]

    # print "Ukupno reci po kategoriji {}".format(total_words_in_category)

    # test(test_data,
    #      NB,
    #      word_frequency_in_class,
    #      total_words_in_category,
    #      categories,
    #      tweet_filter)

    for i in range(0, 30, 1):
        word_frequency_in_class1, all_words1 = feat_reduction(
            word_frequency_in_class, all_words, i)
        test_nb_3 = test(test_data,
                         NB,
                         word_frequency_in_class1,
                         total_words_in_category,
                         categories,
                         tweet_filter)
        timex()

    timex()

from collections import Counter
from math import log

class NaiveBayes:
    def __init__(self, tokneized_tweet, categories):
        self.tokenized_tweet = tokneized_tweet
        self.categories = categories

    def classify(self):
        pass


def NB(categories, tokenized_tweet, word_frequency_in_class, total_words_in_category):
    verovatnoce_kategorija_nb = Counter()
    for category in categories.keys():
        verovatnoca_kategorije = log(
            float(categories[category]) / sum(categories.values()))  # prior za klasu
        for word in tokenized_tweet:
            verovatnoca_kategorije += log(
                max(4E-5, (float(word_frequency_in_class[word][category]) / total_words_in_category[category])))
        verovatnoce_kategorija_nb[category] = verovatnoca_kategorije
    # print 'prior za kategoriju {0} je '.format(category) +
    # str(float(categories[category])/sum(categories.values()))

    return max(verovatnoce_kategorija_nb, key=lambda x: verovatnoce_kategorija_nb[x])


def NB_3(categories, tokenized_tweet, total_words_in_category):
    verovatnoce_kategorija_NB = Counter()
    for category in categories.keys():
        verovatnoca_kategorije = log(
            float(categories[category]) / sum(categories.values()))  # prior za klasu
        for word in tokenized_tweet:
            try:
                verovatnoca_kategorije += log(max(4E-5, (float(items.find_one(
                    {"word": word})['count'][category]) / total_words_in_category[category])))
            except TypeError:
                verovatnoca_kategorije += log(4E-5)
        verovatnoce_kategorija_NB[category] = verovatnoca_kategorije
    # print 'prior za kategoriju {0} je '.format(category) +
    # str(float(categories[category])/sum(categories.values()))

    return max(verovatnoce_kategorija_NB, key=lambda x: verovatnoce_kategorija_NB[x])


def NB_2(categories, word_vector, tokenized_tweet):
    verovatnoce_kategorija_NB = Counter()
    for category in categories.keys():
        uslovna_verv = 0
        verovatnoca_kategorije = log(
            float(categories[category]) / sum(categories.values()))  # prior za klasu
        for word in tokenized_tweet:
            uslovna_verv += log(
                float(word_frequency_in_class[word][category] + 1) / (
                    len(word_vector) + total_words_in_category[category]))
        verovatnoca_kategorije += uslovna_verv
        verovatnoce_kategorija_NB[category] = verovatnoca_kategorije

    return max(verovatnoce_kategorija_NB, key=lambda x: verovatnoce_kategorija_NB[x])

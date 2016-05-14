__author__ = 'dudulez'
from collections import defaultdict, Counter
from tqdm import tqdm

def train(tr_data, tweet_filter):
    word_frequency_in_class = defaultdict(Counter)
    # Counts number of times a word shows up in a class, eg.
    # 'good':{positive:456, negative:23, neutral:34}
    all_words = Counter()  # counts occurrence of all words in training set
    categories = Counter()
    all_words_list = []

    for row in tqdm(tr_data):
        tweet = row[5]
        categories[row[0]] += 1

        for word in tweet_filter(tweet):
            all_words_list.append(word)
            all_words[word] += 1
            word_frequency_in_class[word][row[0]] += 1

    print "ukupno reci u listi svih reci: {} ".format(len(all_words_list))
    return word_frequency_in_class, all_words, categories

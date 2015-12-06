# /usr/lib/python
import time
import re
from sets import Set

import numpy as np
import pandas as pd
import sklearn.cross_validation
import sklearn.feature_extraction.text
import sklearn.metrics
import sklearn.naive_bayes
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from nltk.tokenize import PunktWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import PunktWordTokenizer, WordPunctTokenizer

start_time = time.time()

lemmatizer = WordNetLemmatizer()
stemmatizer = PorterStemmer()
tokenizer = WordPunctTokenizer()


def my_token(s):
    my_tokenizer = PunktWordTokenizer()
    return my_tokenizer.tokenize(s)


stop_words = Set()

with open('english.stop', 'rb') as stop_w:
    for word in stop_w:
        stop_words.add(word.decode('utf-8').rstrip())


def sanitizer(tweet):
    tweet_trimed = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url',
                          tweet)
    tweet_trimed_hash_at = re.sub(':\)|:-\)|:D|:-\]', 'good good', tweet_trimed)
    tweet_trimed_hash_at = re.sub(':\(|:-\(|:\[|:\|', 'bad bad', tweet_trimed_hash_at)
    # tweet_trimed_hash_at = re.sub('\d+.\d+|\d+|\d+th', 'number', tweet_trimed_hash_at)
    tweet_trimed_hash_at = re.sub('/^\d*\.?\d*$/', 'number', tweet_trimed_hash_at)
    tweet_trimed_hash_at = re.sub('/^\d*\,?\d*$/', 'number', tweet_trimed_hash_at)
    tweet_trimed_hash_at = re.sub('/^\d*\-?\d*$/', 'number', tweet_trimed_hash_at)
    # tweet_trimed_hash_at = re.sub("[\(\)&,.:!?`~-]+", '', tweet_trimed_hash_at)
    # tweet_trimed_hash_at = re.sub("(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)", '', tweet_trimed_hash_at)
    tweet_trimed_hash_at = re.sub('[\s]+', ' ', tweet_trimed_hash_at)
    tweet_token = tokenizer.tokenize(tweet_trimed_hash_at.decode('utf-8'))

    tweet_token = tokenizer.tokenize(tweet_trimed_hash_at.decode('ISO-8859-1'))

    tweet_sanitized = [word.lower() for word in tweet_token]
    tweet_lemmatized = [lemmatizer.lemmatize(word) for word in tweet_sanitized]
    # print tweet_lemmatized
    # tweet_stemmatized = [stemmatizer.stem(word) for word in tweet_sanitized if word not in stop_words]
    # tweet_replaced = [replacer.replace(word) for word in tweet_stemmatized]
    # print tweet_stemmatized

    # no_punctuation = lowers.translate(None, string.punctuation)
    tweet_obradjen = " ".join(tweet_lemmatized)

    return tweet_obradjen


categories = ['positive', 'negative', 'neutral']
names = ['id', 'broj', 'label', 'text']

data_train = pd.read_table('/home/dudulez/sentiment_analysis/semeval2014/semeval_train_dev.tsv', sep='\t', names=names)
# data_train  = pd.read_table('/home/dudulez/sentiment_analysis/tweeti_train_neutral.tsv', sep = '\t', names = names)

data_test = pd.read_table('/home/dudulez/sentiment_analysis/semeval2014/semeval_test.tsv', sep='\t', names=names)
# data_test = pd.read_table('/home/dudulez/sentiment_analysis/tweeti_test_neutral.tsv', sep = '\t', names = names)

# split data into traininig and testing sets. Default is 75% train

# )

# after the split, we have two arrays of arrays

# train, test = data_train,data_test
train = np.array(
    data_train.ix[:, names])  # , train1 = sklearn.cross_validation.train_test_split(data_train, train_size= .99999999)
# test, test1 = sklearn.cross_validation.train_test_split(data_test, train_size= .99999999)
test = np.array(data_test.ix[:, names])

train_data, test_data = pd.DataFrame(train, columns=names), pd.DataFrame(test, columns=names)
# vidi da li ovde moze da se promeni nesto


# vectorization is the process of converting all names into a binary vector

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words=stop_words,
                             ngram_range=(1, 2),
                             lowercase=True,
                             tokenizer=my_token,
                             preprocessor=sanitizer
                             )



# vectorizer = sklearn.feature_extraction.CountVectorizer()#encoding=u'utf-8',lowercase=True,
# tokenizer = None, ngram_range(2,3), stop_words = 'english', min_df = 0, max_df = 0.1)



train_matrix = vectorizer.fit_transform(train_data['text'])
test_matrix = vectorizer.transform(test_data['text'])

positive_cases_train = (train_data['label'] == 'positive')
positive_cases_test = (test_data['label'] == 'positive')
negative_cases_train = (train_data['label'] == 'negative')
negative_cases_test = (test_data['label'] == 'negative')
neutral_cases_train = (train_data['label'] == 'neutral')
neutral_cases_test = (test_data['label'] == 'neutral')



# ratio imbalance

# train

classifier = OneVsRestClassifier(LinearSVC())
# classifier = OneVsRestClassifier(sklearn.naive_bayes.MultinomialNB())
# classifier = OneVsRestClassifier(LogisticRegression(penalty='l2', C=1.0))
# classifier = SGDClassifier(alpha = 0.00001, l1_ratio=0.015)


classifier.fit(train_matrix, negative_cases_train)

predict_sentiment = classifier.predict(test_matrix)
# predict_probs = classifier.predict_proba(test_matrix)

accuracy = classifier.score(test_matrix, negative_cases_test)
precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
    negative_cases_test, predict_sentiment)

print(" LogisticRegression, no preprocesing, unigram only")
print ("accuracy = ", accuracy)
print (" precision =", precision)
print ("recal = ", recall)
print ("f1 score = ", f1)

end_time = time.time()
print 'Iterations took %f seconds.' % (end_time - start_time)

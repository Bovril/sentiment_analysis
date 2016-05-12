import re

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from nltk.util import ngrams as nltk_ngrams
import sys
import types

REGEX_PATTERNS = [
    ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
     'url'),
    (':\)|:-\)|:D|:-\]', 'good good'),
    (':\(|:\[|:\|', 'bad bad'),
    ('\d+.\d+|\d+|\d+th', 'number'),
    ('/^\d*\.?\d*$/', 'number'),
    ('/^\d*\-?\d*$/', 'number'),
    ('@[^\s]+', 'atuser'),
    ("[\(\)&,.:!?`~-]+", ''),
    ("(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)", ''),
    ("#[a-zA-Z]+", '')
]

if sys.version_info[0] > 2:  # Python 3+
    create_bound_method = types.MethodType
else:
    def create_bound_method(func, obj):
        return types.MethodType(func, obj, obj.__class__)

class Filter:
    tokenizer = WordPunctTokenizer()

    def __init__(self,
                 tweet,
                 ngram_combo=[],
                 stop_words=[],
                 patterns=[],
                 func=None
                 ):
        self.tweet = tweet
        self.ngram_combo = ngram_combo
        self.stop_words = stop_words
        self.patterns = [(re.compile(pattern), replacer)
                         for (pattern, replacer) in patterns]
        if func:
            self.word_base_method = create_bound_method(func, self)

    def get_tweet(self):
        return self.tweet

    def ngramize(self, o_tweet):
        ngram = []
        for i in self.ngram_combo:
            ngrams = nltk_ngrams(o_tweet.split(), i)
            for grams in ngrams:
                ngram.append(" ".join(grams))
        return ngram

    def regex_replace(self, tweet):
        for (pattern, replacer) in self.patterns:
            (tweet, count) = re.subn(pattern, replacer, tweet)

        return tweet

    def get_sub_tweet(self):
        return self.regex_replace(self.get_tweet())

    def tokenize(self):
        """

        :return: tokenized tweet
        """

        tokenized_tweet = Filter.tokenizer.tokenize(
            self.get_sub_tweet().decode('utf-8'))
        return [word.lower()
                for word in tokenized_tweet if word.lower() not in self.stop_words]

    def word_base_method(self):
        return self.tokenize()

    def __call__(self):
        """
        :return: a list of n-grams
        """
        if len(self.ngram_combo) == 1 and sorted(self.ngram_combo)[0] == 1:
            return self.word_base_method()

        print 'this'
        return self.ngramize(" ".join(self.word_base_method()))


def lemmatize(self):
    lemma = WordNetLemmatizer()
    return [lemma.lemmatize(word)
            for word in self.tokenize()]

def stemmatize(self):
    stemm = PorterStemmer()
    return [stemm.stem(word)
            for word in self.tokenize()]


if __name__ == "__main__":
    stop_words = ['the', 'a', 'is', 'this']
    tweet_ = """This is a random tweet, the 123-12, #HashTags,
    @bovril, http://github.com, GOOOOOOO! LOL, we are cooking, """
    filter_ = Filter(tweet=tweet_,
                     ngram_combo=[1, 2, 3],
                     stop_words=stop_words,
                     patterns=REGEX_PATTERNS,
                     func=stemmatize,
                     )
    print filter_()

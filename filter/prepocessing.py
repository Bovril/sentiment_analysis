import csv

from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
from nltk.util import ngrams as nltk_ngrams

class Read:
    def __init__(self, filename, delimiter=','):
        self.filename = filename
        self.delimiter = delimiter

    @property
    def open_and_read(self):
        try:
            opened = csv.reader(open(self.filename, 'rb'), delimiter='\t')
            return opened

        except IOError as e:
            print e, 1



class Filter:

    def __init__(self,
                 root_reduction=None,
                 ngram_combo=[1],
                 stop_words=True,
                 ):
        self.root_reduction = root_reduction
        self.ngram_combo = ngram_combo
        self.stop_word = stop_words
        self.root_reduct_options = {'stemm':self.stemmer, 'lemma': self.lemmatize}

    def tokenize(self):
        tokenizer = WordPunctTokenizer()
        return tokenized_tweet  # return tweet_tokenized

    def lemmatize(self):
        lemma = WordNetLemmatizer()
        return lemma
    def stemmer(self):
        stemm = PorterStemmer()

    def root_reduct(self):
        action = self.root_reduct_options.get(self.root_reduction)
        return action()



    def ngramize(self):
        ngram = []
        for i in self.ngram_combo:
            ngrams = nltk_ngrams(tvit.split(), i)
            for grams in ngrams:
                ngram.append(" ".join(grams))
        return ngram

    def regex_strip(self, tweet):
        tweet_url_remove = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                                  'url',
                                  tweet)
        tweet_pos_emo = re.sub(':\)|:-\)|:D|:-\]', 'good good', tweet_url_remove)
        tweet_neg_emo = re.sub(':\(|:\[|:\|', 'bad bad', tweet_pos_emo)

        tweet_trimmed = re.sub('\d+.\d+|\d+|\d+th', 'number', tweet_neg_emo)
        tweet_trimmed = re.sub('/^\d*\.?\d*$/', 'number', tweet_trimmed)
        tweet_trimmed = re.sub('/^\d*\,?\d*$/', 'number', tweet_trimmed)
        tweet_trimmed = re.sub('/^\d*\-?\d*$/', 'number', tweet_trimmed)
        tweet_trimmed = re.sub('@[^\s]+', 'atuser', tweet_trimmed)

        # tweet_trimmed = re.sub("[\(\)&,.:!?`~-]+", '', tweet_trimmed)
        # tweet_trimmed = re.sub("(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)", '', tweet_trimmed)
        # tweet_trimmed = tweet_trimmed.translate(None, string.punctuation)
        return tweet_trimmed

        tweet_reducted = self.tokenize().tokenize(tweet_trimmed.decode('utf-8'))

        tweet_sanitized = [word.lower() for word in tweet_token if word not in stop_words]
        tweet_sanitized = [tag.strip("#") for tag in tweet_sanitized]
        # tweet_lemmatized = [lemmatizer.lemmatize(word) for word in tweet_sanitized if word not in stop_words]
        # print tweet_lemmatized
        tweet_stemmatized = [stemmatizer.stem(word) for word in tweet_sanitized if word not in stop_words]
        # print tweet_stemmatized
        tweet_obradjen = " ".join(tweet_stemmatized)

        ngram_tvit = ngramz(tweet_obradjen, ngram_broj)
        return self.ngramize(tweet_obradjen, ngram_broj)

    def filter(self, tweet):
        self.regex_strip(tweet)
        self.tokenize()
        self.root_reduct()
        self.ngramize()
        return ngramized_tweet
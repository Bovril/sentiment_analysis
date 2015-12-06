import csv
import re
from sets import Set
import sys
from collections import Counter, defaultdict

reload(sys)
sys.setdefaultencoding('utf8')
from nltk.tokenize import PunktWordTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
import time
from math import log
from nltk.util import ngrams
import string

start_time = time.time()


def word_feats(words):
    return dict([(word, True) for word in words])


def read_from_file(filename):
    with open(filename, 'rb') as opened_file:
        opened_file = csv.reader(opened_file, delimiter='\t')
    return opened_file

# print read_from_file('probni_tvitovi.tsv')


# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

ngram_broj = [1, 2]


def ngramz(tvit, n):
    ngram = []
    for i in n:
        sixgrams = ngrams(tvit.split(), i)
        for grams in sixgrams:
            ngram.append(" ".join(grams))
    return ngram


stop_words = Set()

with open('english.stop', 'rb') as stop_w:
    for word in stop_w:
        stop_words.add(word.decode('utf-8').rstrip())

stemmatizer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
tokenizer = PunktWordTokenizer()

total_tweets = 0


def sanitizer(tweet, stemmatizer):
    tweet_trimed = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url',
                          tweet)
    tweet_trimed_hash_at = re.sub(':\)|:-\)|:D|:-\]', 'good good', tweet_trimed)
    tweet_trimed_hash_at = re.sub(':\(|:-\(|:\[|:\|', 'bad bad', tweet_trimed_hash_at)

    tweet_trimed_hash_at = re.sub('\d+.\d+|\d+|\d+th', 'number', tweet_trimed_hash_at)
    tweet_trimed_hash_at = re.sub('/^\d*\.?\d*$/', 'number', tweet_trimed_hash_at)
    tweet_trimed_hash_at = re.sub('/^\d*\,?\d*$/', 'number', tweet_trimed_hash_at)
    tweet_trimed_hash_at = re.sub('/^\d*\-?\d*$/', 'number', tweet_trimed_hash_at)
    tweet_trimed_hash_at = re.sub('@[^\s]+', 'atuser', tweet_trimed_hash_at)

    # tweet_trimed_hash_at = re.sub("[\(\)&,.:!?`~-]+", '', tweet_trimed_hash_at)
    # tweet_trimed_hash_at = re.sub("(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)", '', tweet_trimed_hash_at)
    tweet_trimed_hash_at = tweet_trimed_hash_at.translate(None, string.punctuation)

    tweet_token = tokenizer.tokenize(tweet_trimed_hash_at.decode('ISO-8859-1'))


    is_upper = []
    tweet_sanitized = [word.lower() for word in tweet_token if word not in stop_words]
    tweet_sanitized = [tag.strip("#") for tag in tweet_sanitized]

    # tweet_stemmatized = [stemmatizer.stem(word) for word in tweet_sanitized if word not in stop_words]
    tweet_lemmatized = [lemmatizer.lemmatize(word) for word in tweet_sanitized if word not in stop_words]

    tweet_obradjen = " ".join(tweet_lemmatized)

    ngram_tvit = ngramz(tweet_obradjen, ngram_broj)
    return ngram_tvit


sentence = "this is a foo bar sentences, and i want to ngramize it! don't worry! :) http://bit.ly/da"

print sanitizer(sentence, stemmatizer)

print '######################################'
print '############ TRAINING ################'
print '######################################'
print ngram_broj
print "  # dodat, lema,  "

training_data = '/home/dudulez/sentiment_analysis/trainingandtestdata/training.1600000_posneg.csv'
# training_data = '/home/dudulez/sentiment_analysis/trainingandtestdata/training.50000.processed.posneg.csv'
test_data = '/home/dudulez/sentiment_analysis/trainingandtestdata/testdata.csv'
i = 0
tw_id = 0
list_of_rows = []


def train():
    kombo_reci_kategorije = defaultdict(Counter)  # broji koliko se svaka rec pojavljuje u nekoj kategoriji pr
    # 'good':{positive:456, negative:23, neutral:34}
    all_words = Counter()
    categories = Counter()  # broj tvitova u svakoj kategoriji

    with open(training_data, 'rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter=',')

        for row in tsvin:

            tweet = row[5]
            categories[row[0]] += 1

            tweet_token = sanitizer(tweet, stemmatizer)
            for word in tweet_token:
                # vector.add(word)
                all_words[word] += 1
                kombo_reci_kategorije[word][row[0]] += 1
    return kombo_reci_kategorije, all_words, categories


kombo_reci_kategorije, all_words, categories = train()


def feat_reduction(kombo_reci_kategorije, all_words, n):
    for word in all_words.keys():
        if sum(kombo_reci_kategorije[word].values()) <= n:
            del kombo_reci_kategorije[word]
            del all_words[word]
    print "uklonjene su reci sa manje od %d ukupno pojavljivanja u svim klasama" % n
    print len(all_words.keys())
    return kombo_reci_kategorije, all_words, n, len(all_words.keys())


ukupno_reci_po_kategoriji = Counter()

print 'pojavljivanje reci moth u svakoj kategoriji' + str(kombo_reci_kategorije['moth'])
print categories

vector = [i for i in kombo_reci_kategorije.keys()]  # moze i vector umesto ovoga
for i in vector:
    for j in categories.keys():
        ukupno_reci_po_kategoriji[j] += kombo_reci_kategorije[i][j]

print 'Reci koje se najcesce pojavljuju ' + str(all_words.most_common(30))

print '*** Ukupno reci u svim tvitovima je ' + str(sum(all_words.values()))

ukupno_reci_u_dokumentu = sum(all_words.values())
rec = 'moth'

print 'verovatnoca da ce se rec {0} pojaviti u celom dokumentu je {1}'.format(rec, float(
    all_words[rec]) / ukupno_reci_u_dokumentu)

# test tacnosti 
# za svaki twit proveriti da li je klasa ista kao i u fajlu, ako jeste dodajemo 1 i na kraju podelimo broj
# sa ukupnim brojem

dummy = 'the team lost the game last night defeat was expected'.split(' ')
dummy_lst = [f for f in dummy if f not in stop_words]
dummy_2 = 'Lunch from my new Lil spot 33 gooooooood ...THE COTTON BOWL ....pretty\
 good#1st#time#will be going back# http://instagr.am/p/RX9939CIv8/'

print ukupno_reci_po_kategoriji


def NB(categories, vector, tokenized_tweet):
    verovatnoce_kategorija_NB = Counter()
    for category in categories.keys():
        verovatnoca_kategorije = log(float(categories[category]) / sum(categories.values()))  # prior za klasu
        for word in tokenized_tweet:
            verovatnoca_kategorije += log(
                max(9E-6, (float(kombo_reci_kategorije[word][category]) / ukupno_reci_po_kategoriji[category])))
        verovatnoce_kategorija_NB[category] = verovatnoca_kategorije
    # print 'prior za kategoriju {0} je '.format(category) + str(float(categories[category])/sum(categories.values()))

    return max(verovatnoce_kategorija_NB, key=lambda x: verovatnoce_kategorija_NB[x])


def NB_2(categories, vector, tokenized_tweet):
    verovatnoce_kategorija_NB = Counter()
    for category in categories.keys():
        uslovna_verv = 0
        verovatnoca_kategorije = log(float(categories[category]) / sum(categories.values()))  # prior za klasu
        for word in tokenized_tweet:
            uslovna_verv += log(
                float(kombo_reci_kategorije[word][category] + 1) / (len(vector) + ukupno_reci_po_kategoriji[category]))
        verovatnoca_kategorije += uslovna_verv
        verovatnoce_kategorija_NB[category] = verovatnoca_kategorije
    # print 'prior za kategoriju {0} je '.format(category) + str(float(categories[category])/sum(categories.values()))

    return max(verovatnoce_kategorija_NB, key=lambda x: verovatnoce_kategorija_NB[x])


def fisher(tokenized_tweet):
    freqSum = 0
    freq_feat_in_cat = defaultdict(Counter)
    for category in categories:
        for word in tokenized_tweet:
            freq_feat_in_cat[word][category] = float(kombo_reci_kategorije[word][category]) / categories[category]
            freqSum = sum(freq_feat_in_cat[word].values())
            print freqSum
    return freq_feat_in_cat


def test(test_file, classifier, categories, vector):
    """Uzima test fajl i za svaki red radi tokenizaciju, zatim provlaci listu kroz NB klasifikator,
    izlaz je kategorija tweeta, uporedjujemo je sa stvarnom kategorijom tweeta (nalazi se u nekoj koloni testa),
    vraca
    """
    vector = all_words.keys()
    total_test_tweets = 0
    correct_tweets = 0
    sent = ''
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    with open(test_file, 'rb') as opened_file:
        opened_file = csv.reader(opened_file, delimiter=',')
        for line in opened_file:
            total_test_tweets += 1
            sent = classifier(categories, vector, sanitizer(line[5], stemmatizer))
            if sent == line[0] and sent == "positive":
                tp += 1
                correct_tweets += 1
            elif sent == line[0] and sent == "negative":
                tn += 1
                correct_tweets += 1
            elif sent != line[0] and sent == "positive":
                fp += 1
            elif sent != line[0] and sent == "negative":
                fn += 1

        prec_poz = tp / (tp + fp)
        prec_neg = tn / (tn + fn)
        rec_poz = tp / (tp + fn)
        rec_neg = tn / (tn + fp)
        f_poz = (2 * prec_poz * rec_poz) / (prec_poz + rec_poz)
        f_neg = (2 * prec_neg * rec_neg) / (prec_neg + rec_neg)

    print '********* Ukupan broj tweetova u testu je {0}, broj tacnih odgovora je {1}'.format(total_test_tweets,
                                                                                              correct_tweets)
    print '********* Procenat tacnosti je {0:.2f} % '.format((correct_tweets / float(total_test_tweets)) * 100)
    print "-------------------------"
    print " | " + str(tp) + " | " + str(fp) + " | "

    print "--------------------------"
    print " | " + str(fn) + " | " + str(tn) + " | "
    print "--------------------------"
    print "Preciznost pozitivne klase je " + str(prec_poz)
    print "Preciznost negativne klase je " + str(prec_neg)
    print "Recal pozitivne klase je " + str(rec_poz)
    print "Recal negativne klase je " + str(rec_neg)
    print "F1 pozitivne klase je " + str(f_poz) + " a negativne je " + str(f_neg)
    return (correct_tweets / float(total_test_tweets)) * 100


def MI(kombo_reci_kategorije, ukupno_reci_u_dokumentu, categories):
    """izlaz: reci sortirane po mi
    -uzimamo max MI od svih klasa
    -
    """
    pass

# test_1 = NB(categories, vector, dummy_lst)
sani = sanitizer(dummy_2, stemmatizer)
print sani
# print test_1
print ("--- {0} seconds ---".format(time.time() - start_time))
print '######################################'
print '############## TESTING ###############'
print '######################################'

test_nb = test(test_data, NB, categories, vector)
# test_nb_4 = test(test_data, NB_2, categories, vector)

# test_nb_1 = test('/home/dudulez/sentiment_analysis/trainingandtestdata/test2.csv', NB, categories, vector)
# test_nb_2 = test('/home/dudulez/sentiment_analysis/trainingandtestdata/test2.csv',NB_2,  categories, vector)

tac = []
removed = []
feat = []
temp_removed = 0
temp_tac = 0
temp_feat = 0

for i in range(0, 500, 2):
    kombo_reci_kategorije, all_words, temp_removed, temp_feat = feat_reduction(kombo_reci_kategorije, all_words, i)
    test_nb_3 = test(test_data, NB, categories, vector)
    # test_nb_1 = test('/home/dudulez/sentiment_analysis/trainingandtestdata/test2.csv', NB, categories, vector)
    tac.append(test_nb_3)
    removed.append(temp_removed)
    feat.append(temp_feat)

print "removed"
print removed
print "tac"
print tac
print "feat"
print feat




# print categories

# preciznost
# provuci svaki tweet iz testa kroz petlju, videti da li je kategorija(klasa) ista koja se dobije ista kao i fajlu

# to do: lemma, stema i jos bolje filtriranje
# feature selection
# probati formulu sa onih slajdova
print ("--- {0} seconds ---".format(time.time() - start_time))

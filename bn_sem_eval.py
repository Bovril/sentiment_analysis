import csv
import re
import nltk.data
from sets import Set
import sys
from collections import Counter, defaultdict
import string
from math import log
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import FreqDist
from nltk.stem import PorterStemmer, WordNetLemmatizer
import time
from nltk.tokenize import PunktWordTokenizer, WordPunctTokenizer
from nltk.util import ngrams

reload(sys)
sys.setdefaultencoding('utf8')

start_time = time.time()

tokenizer = WordPunctTokenizer()


def word_feats(words):
    return dict([(word, True) for word in words])


def read_from_file(filename):
    with open(filename, 'rb') as opened_file:
        opened_file = csv.reader(opened_file, delimiter='\t')
    return opened_file

# print read_from_file('probni_tvitovi.tsv')

# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')



stop_words = Set()

with open('english.stop', 'rb') as stop_w:
    for word in stop_w:
        stop_words.add(word.decode('utf-8').rstrip())

ngram_broj = [1, 2]


def ngramz(tvit, n):
    ngram = []
    for i in n:
        sixgrams = ngrams(tvit.split(), i)
        for grams in sixgrams:
            ngram.append(" ".join(grams))
    return ngram


lemmatizer = WordNetLemmatizer()
stemmatizer = PorterStemmer()
tokenizer = WordPunctTokenizer()
total_tweets = 0


def sanitizer(tweet, stemmatizer):
    tweet_url_remove = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url',
                              tweet)
    tweet_pos_emo = re.sub(':\)|:-\)|:D|:-\]', 'good good', tweet_url_remove)
    tweet_neg_emo = re.sub(':\(|:\[|:\|', 'bad bad', tweet_pos_emo)

    tweet_trimed_hash_at = re.sub('\d+.\d+|\d+|\d+th', 'number', tweet_neg_emo)
    tweet_trimed_hash_at = re.sub('/^\d*\.?\d*$/', 'number', tweet_trimed_hash_at)
    tweet_trimed_hash_at = re.sub('/^\d*\,?\d*$/', 'number', tweet_trimed_hash_at)
    tweet_trimed_hash_at = re.sub('/^\d*\-?\d*$/', 'number', tweet_trimed_hash_at)
    tweet_trimed_hash_at = re.sub('@[^\s]+', 'atuser', tweet_trimed_hash_at)

    # tweet_trimed_hash_at = re.sub("[\(\)&,.:!?`~-]+", '', tweet_trimed_hash_at)
    # tweet_trimed_hash_at = re.sub("(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)", '', tweet_trimed_hash_at)
    # tweet_trimed_hash_at = tweet_trimed_hash_at.translate(None, string.punctuation)

    tweet_token = tokenizer.tokenize(tweet_trimed_hash_at.decode('utf-8'))

    tweet_sanitized = [word.lower() for word in tweet_token if word not in stop_words]
    tweet_sanitized = [tag.strip("#") for tag in tweet_sanitized]
    # tweet_lemmatized = [lemmatizer.lemmatize(word) for word in tweet_sanitized if word not in stop_words]
    # print tweet_lemmatized
    tweet_stemmatized = [stemmatizer.stem(word) for word in tweet_sanitized if word not in stop_words]
    # print tweet_stemmatized
    tweet_obradjen = " ".join(tweet_stemmatized)

    ngram_tvit = ngramz(tweet_obradjen, ngram_broj)
    return ngramz(tweet_obradjen, ngram_broj)

# sentence = "this is a foo bar sentences, and i want to ngramize it! don't worry! :) http://bit.ly/da"

# print sanitizer(sentence, stemmatizer)

print '######################################'
print '############ TRAINING ################'
print '######################################'
print ngram_broj
print " bez lema i stema  "
print "semeval2014"

training_data = "/semeval2014/semeval_train_dev.tsv"
test_data = "/semeval2014/semeval_test.tsv"

# training_data = "/tweeti_train.tsv"
# test_data = "/tweeti_test.tsv"

i = 0
tw_id = 0
list_of_rows = []


def train():
    kombo_reci_kategorije = defaultdict(Counter)
    ##broji koliko se svaka rec pojavljuje u nekoj kategoriji pr. :
    # 'good':{positive:456, negative:23, neutral:34}
    all_words = Counter()
    categories = Counter()

    with open(training_data, 'rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t')

        for row in tsvin:
            tweet = row[3]
            categories[row[2]] += 1

            tweet_token = sanitizer(tweet, stemmatizer)
            for word in tweet_token:
                all_words[word] += 1
                kombo_reci_kategorije[word][row[2]] += 1
    return kombo_reci_kategorije, all_words, categories


kombo_reci_kategorije, all_words, categories = train()


def feat_reduction(kombo_reci_kategorije, all_words, n):
    for word in all_words.keys():
        if sum(kombo_reci_kategorije[word].values()) <= n:
            del kombo_reci_kategorije[word]
            del all_words[word]
    print "uklonjene su reci sa manje od %d ukupno pojavljivanja u svim klasama" % n
    print len(all_words.keys())
    return kombo_reci_kategorije, all_words


ukupno_reci_po_kategoriji = Counter()

# print 'pojavljivanje reci moth u svakoj kategoriji' + str(kombo_reci_kategorije['moth'])
# print categories

vector = [i for i in kombo_reci_kategorije.keys()]  # moze i vector umesto ovoga
for i in vector:
    for j in categories.keys():
        ukupno_reci_po_kategoriji[j] += kombo_reci_kategorije[i][j]


# print 'Reci koje se najcesce pojavljuju ' +  str(all_words.most_common(30))

# print '*** Ukupno reci u svim tvitovima je ' + str(sum(all_words.values()))

ukupno_reci_u_dokumentu = sum(all_words.values())
# rec = 'moth'

# print 'verovatnoca da ce se rec {0} pojaviti u celom dokumentu je {1}'.format(rec, float\
#	(all_words[rec])/ukupno_reci_u_dokumentu ) 

# test tacnosti 
# za svaki twit proveriti da li je klasa ista kao i u fajlu, ako jeste dodajemo 1 i na kraju podelimo broj
# sa ukupnim brojem

dummy = 'the team lost the game last night defeat was expected'.split(' ')
dummy_lst = [f for f in dummy if f not in stop_words]
dummy_2 = 'Lunch from my new Lil spot 33 gooooooood ...THE COTTON BOWL ....pretty\
 good#1st#time#will be going back# http://instagr.am/p/RX9939CIv8/'


# print ukupno_reci_po_kategoriji

def NB(categories, vector, tokenized_tweet):
    verovatnoce_kategorija_NB = Counter()
    for category in categories.keys():
        verovatnoca_kategorije = log(float(categories[category]) / sum(categories.values()))  # prior za klasu
        for word in tokenized_tweet:
            verovatnoca_kategorije += log(
                max(4E-5, (float(kombo_reci_kategorije[word][category]) / ukupno_reci_po_kategoriji[category])))
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


def test(test_file, classifier, categories, vector):
    """Uzima test fajl i za svaki red radi tokenizaciju, zatim provlaci listu kroz NB klasifikator,
    izlaz je kategorija tweeta, uporedjujemo je sa stvarnom kategorijom tweeta (nalazi se u nekoj koloni testa),
    vraca
    """
    vector = all_words.keys()
    total_test_tweets = 0
    correct_tweets = 0
    sent = ''
    conf = defaultdict(Counter)

    with open(test_file, 'rb') as opened_file:
        opened_file = csv.reader(opened_file, delimiter='\t')
        for line in opened_file:
            total_test_tweets += 1
            sent = classifier(categories, vector, sanitizer(line[3], stemmatizer))
            if sent == line[2]:
                correct_tweets += 1
            conf[line[2]][sent] += 1
    print conf
    print '********* Ukupan broj tweetova u testu je {0}, broj tacnih odgovora je {1}"'.format(total_test_tweets,
                                                                                               correct_tweets)
    print '********* Procenat tacnosti je {0} '.format(correct_tweets / float(total_test_tweets))

    prec = 0
    recall = 0
    micro_f = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    list_i = ['positive', 'negative', 'neutral']
    list_j = ['positive', 'negative', 'neutral']

    # for t in range(len(list_i)):
    # 	tp, tn, fp, fn = 0,0,0,0
    # 	temp = t
    # 	print '*************'
    # 	for i in list_i:
    # 		for j in list_j:
    # 			if (i == j) and (list_i.index(i) == t):
    # 				tp=conf[i][j]
    # 				# print i, j
    # 				print ' ii  TP   ',tp
    # 			elif (i != j) and (list_i.index(i) == t):
    # 				fp+= conf[i][j]
    # 				# print i, j
    # 				print ' ij   FP  ',fp
    # 			elif (list_i.index(i) != t) and (list_i.index(j) == t):
    # 				fn += conf[i][j]
    # 				# print i, j
    # 				print '  ji  FN  ',fn
    # 			else:
    # 				tn+=conf[i][j]
    # 				# print i, j
    # 				print ' ost   TN  ',tn

    # 	prec += (float(tp))/(float(tp+fp))
    # 	print prec
    # 	recall += (float(tp))/(float(tp+fn))
    # 	print recall
    # 	micro_f += (float(2*prec*recall)) / float(prec+recall)
    # 	print micro_f

    # print "prec: " + str(prec/3) + "; recall: " + str(recall/3) + "; fscore: " + str(micro_f/3)

    TPpositive = conf["positive"]["positive"]
    FPpositive = conf["positive"]["neutral"] + conf["positive"]["negative"]
    TNpositive = conf['negative']['negative'] + conf['neutral']['neutral'] + conf['negative']['neutral'] + \
                 conf['neutral']['negative']
    FNpositive = conf["negative"]["positive"] + conf["neutral"]["positive"]

    prec_positive = float(TPpositive) / float(TPpositive + FPpositive)
    recall_positive = float(TPpositive) / float(TPpositive + FNpositive)

    f_positive = (2 * prec_positive * recall_positive) / (prec_positive + recall_positive)
    acc_positive = float(TPpositive + TNpositive) / float(TPpositive + TNpositive + FPpositive + FNpositive)

    print "prec_positive; recall_positive; f_positive; acc_positive"
    print prec_positive, recall_positive, f_positive, acc_positive
    print ""

    TPnegative = conf["negative"]["negative"]
    FPnegative = conf['negative']['positive'] + conf['negative']['neutral']
    TNnegative = conf["positive"]["positive"] + conf["positive"]["neutral"] + conf["neutral"]["positive"] + \
                 conf["neutral"]["neutral"]
    FNnegative = conf['positive']['negative'] + conf['neutral']['negative']

    prec_negative = float(TPnegative) / float(TPnegative + FPnegative)
    recall_negative = float(TPnegative) / float(TPnegative + FNnegative)

    f_negative = (2 * prec_negative * recall_negative) / (prec_negative + recall_negative)
    acc_negative = float(TPnegative + TNnegative) / float(TPnegative + TNnegative + FPnegative + FNnegative)

    print "prec_negative, recall_negative, f_negative, acc_negative"
    print prec_negative, recall_negative, f_negative, acc_negative
    print ""

    TPneutral = conf["neutral"]["neutral"]
    FPneutral = conf["neutral"]["positive"] + conf["neutral"]["negative"]
    TNneutral = conf["positive"]["positive"] + conf["negative"]["negative"] + conf["positive"]["negative"] + \
                conf["negative"]["positive"]
    FNneutral = conf['positive']['neutral'] + conf['negative']['neutral']

    prec_neutral = float(TPneutral) / float(TPneutral + FPneutral)
    recall_neutral = float(TPneutral) / float(TPneutral + FNneutral)

    f_neutral = (2 * prec_neutral * recall_neutral) / (prec_neutral + recall_neutral)
    acc_neutral = float(TPneutral + TNneutral) / float(TPneutral + TNneutral + FNneutral + FPneutral)

    print "prec_neutral, recall_neutral, f_neutral, acc_neutral"
    print prec_neutral, recall_neutral, f_neutral, acc_neutral
    print ""

    tp = TPpositive + TPnegative + TPneutral
    fp = FPpositive + FPnegative + FPneutral
    tn = TNpositive + TNnegative + TNneutral
    fn = FNpositive + FNnegative + FNneutral

    prec = float(tp) / float(tp + fp)
    recall = float(tp) / float(tp + fn)
    micro_f = (float(2 * prec * recall)) / float(prec + recall)
    acc = float(tp + tn) / float(tp + tn + fp + fn)

    print prec, recall, micro_f, acc

    print "ukupna preciznost: " + str((prec_positive + prec_neutral + prec_negative) / 3)
    print "total recall: " + str((recall_neutral + recall_negative + recall_positive) / 3)
    print "f mera total: " + str((f_neutral + f_negative + f_positive) / 3)
    print "ukupna tacnost: " + str((acc_neutral + acc_positive + acc_negative) / 3)

# # test_1 = NB(categories, ukupno_reci_u_kategoriji, dummy_lst)
# sani = sanitizer(dummy_2, stemmatizer)
# print sani
# # print test_1




test_nb = test(test_data, NB, categories, vector)
# test_nb1 = test(test_data, NB_2, categories, vector)

for i in range(0, 15, 1):
    kombo_reci_kategorije, all_words = feat_reduction(kombo_reci_kategorije, all_words, i)
    test_nb_3 = test(test_data, NB, categories, vector)
# #test_nb_1 = test('/home/dudulez/sentiment_analysis/trainingandtestdata/test2.csv', NB, categories, vector)


# print categories

# preciznost
# provuci svaki tweet iz testa kroz petlju, videti da li je kategorija(klasa) ista koja se dobije ista kao i fajlu

# to do: lemma, stema i jos bolje filtriranje
# feature selection
# probati formulu sa onih slajdova
print ("--- {0} seconds ---".format(time.time() - start_time))

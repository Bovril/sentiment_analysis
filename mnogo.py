from pymongo import MongoClient
from bn_sem_eval import sanitizer, NB_3
from nltk.stem import PorterStemmer

stemmatizer = PorterStemmer()

""" MongoDB stuff"""
client = MongoClient()
db = client.sent
items = db.my_collection



total_words_in_category = {'positive': 11485968, 'negative': 11403178}
KAT = {'positive': 800000, 'negative': 800000}
sentence = """this is a foo bar sentences, and i want to ngramize it!
    don't worry! :) http://bit.ly/da"""

# Fajlovi za serijalizaciju i deserijalizaciju
lista_fajlova = ['kombo_fajl', 'all_words_fajl', 'kategorije_fajl']

# timex()
# print lista_fajlova[2]

# des = Deserialize([lista_fajlova[2]])

# print type(des)
# kat = des.deserialize()


sentence = """this is a foo bar sentences, and i want to ngramize it!
    don't worry! :) http://bit.ly/da"""


vector = db.command("collstats", "my_collection")['count']

# timex()

def insert_to_mongo():
    """Upis svake reci i njenog pojavljivanja u svakoj klasi u bazu"""
    pass


def classify(tweet):
    sl = sanitizer(tweet, stemmatizer)
    return NB_3(KAT, vector, sl, total_words_in_category)


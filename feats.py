__author__ = 'dudulez'

import collections
import itertools
import nltk.classify.util
import nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, precision, recall
from nltk.probability import FreqDist, ConditionalFreqDist
#
# def evaluate_classifier(featx):
#
#     negids = movie_reviews.fileids('neg')
#     posids = movie_reviews.fileids('pos')
#
#     negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg')
#                 for f in negids]
#     posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos')
#                 for f in posids]
#
#     negcutoff = len(negfeats) * 3 / 4
#     poscutoff = len(posfeats) * 3 / 4
#
#     trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
#     testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
#
#     classifier = NaiveBayesClassifier.train(trainfeats)
#     refsets = collections.defaultdict(set)
#     testsets = collections.defaultdict(set)
#
#     for i, (feats, label) in enumerate(testfeats):
#         refsets[label].add(i)
#         observed = classifier.classify(feats)
#         testsets[observed].add(i)
#
#     print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
#     print 'pos precision:', precision(refsets['pos'], testsets['pos'])
#     print 'pos recall:', recall(refsets['pos'], testsets['pos'])
#     print 'neg precision:', precision(refsets['neg'], testsets['neg'])
#     print 'neg recall:', recall(refsets['neg'], testsets['neg'])
#     classifier.show_most_informative_features()
#
#
# def word_feats(words):
#     return dict([(word, True) for word in words])
#
# # print 'evaluating single word features'
# # evaluate_classifier(word_feats)
#
#
# word_fd = FreqDist()  # dict type object
# label_word_fd = ConditionalFreqDist()
#
# for word in movie_reviews.words(categories=['pos']):
#     word_fd[word.lower()] += 1
#     label_word_fd['pos'][word.lower()] += 1
#
# for word in movie_reviews.words(categories=['neg']):
#     word_fd[word.lower()] += 1
#     label_word_fd['neg'][word.lower()] += 1
#
# # # n_ii = label_word_fd[label][word]
# # # n_ix = word_fd[word]
# # # n_xi = label_word_fd[label].N()
# # # n_xx = label_word_fd.N()
#
# print type(word_fd)
# print (label_word_fd.keys()[:1])
#
#
# pos_word_count = label_word_fd['pos'].N()
# neg_word_count = label_word_fd['neg'].N()
# total_word_count = pos_word_count + neg_word_count
#
# word_scores = {}
#
# for word, freq in word_fd.iteritems():
#     pos_score = BigramAssocMeasures.chi_sq(label_word_fd['pos'][word],
#                                            (freq, pos_word_count), total_word_count)
#     neg_score = BigramAssocMeasures.chi_sq(label_word_fd['neg'][word],
#                                            (freq, neg_word_count), total_word_count)
#     word_scores[word] = pos_score + neg_score
#
# best = sorted(
#     word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:10000]
# bestwords = set([w for w, s in best])
#
#
# def best_word_feats(words):
#     return dict([(word, True) for word in words if word in bestwords])
#
# # print 'evaluating best word features'
# # evaluate_classifier(best_word_feats)
#
#
# # def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
# #     bigram_finder = BigramCollocationFinder.from_words(words)
# #     bigrams = bigram_finder.nbest(score_fn, n)
# #     d = dict([(bigram, True) for bigram in bigrams])
# #     d.update(best_word_feats(words))
# #     return d
#
# # print 'evaluating best words + bigram chi_sq word features'
# # evaluate_classifier(best_bigram_word_feats)
#
# negids = movie_reviews.fileids('neg')
# # print negids[:10]
#
#
# posids = movie_reviews.fileids('pos')
# words_n = (movie_reviews.words(fileids=['neg/cv000_29416.txt']), 'neg')
# # print words_n
#
# negfeats = [(best_word_feats(movie_reviews.words(fileids=[f])), 'neg')
#             for f in negids]
#
# posfeats = [(best_word_feats(movie_reviews.words(fileids=[f])), 'pos')
#             for f in posids]
# # print posfeats[:10]
#
# negcutoff = len(negfeats) * 3 / 4
# poscutoff = len(posfeats) * 3 / 4
#
# trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
# testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
#
# classifier = NaiveBayesClassifier.train(trainfeats)
# refsets = collections.defaultdict(set)
# testsets = collections.defaultdict(set)
# documents = [(list(movie_reviews.words(fileid)), category)
#              for category in movie_reviews.categories()
#              for fileid in movie_reviews.fileids(category)]



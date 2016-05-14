__author__ = 'dudulez'

from collections import Counter, defaultdict

from reader.reader import Read


def test(test_file,
         classifier,
         word_frequency_in_class,
         total_words_in_category,
         categories,
         filter_):
    """Uzima test fajl i za svaki red radi tokenizaciju, zatim provlaci listu kroz NB klasifikator,
    izlaz je kategorija tweeta, uporedjujemo je sa stvarnom kategorijom tweeta (nalazi se u nekoj koloni testa),
    vraca
    """
    total_test_tweets = 0
    correct_tweets = 0
    conf = defaultdict(Counter)
    test_data = Read.load_data(test_file)

    for line in test_data:
        total_test_tweets += 1
        sent = classifier(
            categories,
            filter_(line[5]),
            word_frequency_in_class,
            total_words_in_category
        )
        if sent == line[0]:
            correct_tweets += 1
        conf[line[0]][sent] += 1
    # print (conf)
    print ('********* Ukupan broj tweetova u testu je {0}, broj tacnih odgovora je {1}"'.format(total_test_tweets,
                                                                                                correct_tweets))
    print ('********* Procenat tacnosti je {0} '.format(correct_tweets / float(total_test_tweets)))

    # prec = 0
    # recall = 0
    # micro_f = 0
    # tp, tn, fp, fn = 0, 0, 0, 0
    # list_i = ['positive', 'negative']
    # list_j = ['positive', 'negative']

    # for t in range(len(list_i)):
    # tp, tn, fp, fn = 0,0,0,0
    # temp = t
    # print '*************'
    # for i in list_i:
    # for j in list_j:
    # if (i == j) and (list_i.index(i) == t):
    # tp=conf[i][j]
    # print i, j
    # print ' ii  TP   ',tp
    # elif (i != j) and (list_i.index(i) == t):
    # fp+= conf[i][j]
    # print i, j
    # print ' ij   FP  ',fp
    # elif (list_i.index(i) != t) and (list_i.index(j) == t):
    # fn += conf[i][j]
    # print i, j
    # print '  ji  FN  ',fn
    # else:
    # tn+=conf[i][j]
    # print i, j
    # print ' ost   TN  ',tn

    # prec += (float(tp))/(float(tp+fp))
    # print prec
    # recall += (float(tp))/(float(tp+fn))
    # print recall
    # micro_f += (float(2*prec*recall)) / float(prec+recall)
    # print micro_f

    # print "prec: " + str(prec/3) + "; recall: " + str(recall/3) + "; fscore:
    # " + str(micro_f/3)

    # TPpositive = conf["positive"]["positive"]
    # FPpositive = conf["positive"]["neutral"] + conf["positive"]["negative"]
    # TNpositive = conf['negative']['negative'] + conf['neutral']['neutral'] + conf['negative']['neutral'] + \
    #     conf['neutral']['negative']
    # FNpositive = conf["negative"]["positive"] + conf["neutral"]["positive"]

    # prec_positive = float(TPpositive) / float(TPpositive + FPpositive)
    # recall_positive = float(TPpositive) / float(TPpositive + FNpositive)

    # f_positive = (2 * prec_positive * recall_positive) / \
    #     (prec_positive + recall_positive)
    # acc_positive = float(TPpositive + TNpositive) / \
    #     float(TPpositive + TNpositive + FPpositive + FNpositive)

    # print "prec_positive; recall_positive; f_positive; acc_positive"
    # print prec_positive, recall_positive, f_positive, acc_positive
    # print ""

    # TPnegative = conf["negative"]["negative"]
    # FPnegative = conf['negative']['positive'] + conf['negative']['neutral']
    # TNnegative = conf["positive"]["positive"] + conf["positive"]["neutral"] + conf["neutral"]["positive"] + \
    #     conf["neutral"]["neutral"]
    # FNnegative = conf['positive']['negative'] + conf['neutral']['negative']

    # prec_negative = float(TPnegative) / float(TPnegative + FPnegative)
    # recall_negative = float(TPnegative) / float(TPnegative + FNnegative)

    # f_negative = (2 * prec_negative * recall_negative) / \
    #     (prec_negative + recall_negative)
    # acc_negative = float(TPnegative + TNnegative) / \
    #     float(TPnegative + TNnegative + FPnegative + FNnegative)

    # print "prec_negative, recall_negative, f_negative, acc_negative"
    # print prec_negative, recall_negative, f_negative, acc_negative
    # print ""

    # TPneutral = conf["neutral"]["neutral"]
    # FPneutral = conf["neutral"]["positive"] + conf["neutral"]["negative"]
    # TNneutral = conf["positive"]["positive"] + conf["negative"]["negative"] + conf["positive"]["negative"] + \
    #     conf["negative"]["positive"]
    # FNneutral = conf['positive']['neutral'] + conf['negative']['neutral']

    # prec_neutral = float(TPneutral) / float(TPneutral + FPneutral)
    # recall_neutral = float(TPneutral) / float(TPneutral + FNneutral)

    # f_neutral = (2 * prec_neutral * recall_neutral) / \
    #     (prec_neutral + recall_neutral)
    # acc_neutral = float(TPneutral + TNneutral) / \
    #     float(TPneutral + TNneutral + FNneutral + FPneutral)

    # print "prec_neutral, recall_neutral, f_neutral, acc_neutral"
    # print prec_neutral, recall_neutral, f_neutral, acc_neutral
    # print ""

    # tp = TPpositive + TPnegative + TPneutral
    # fp = FPpositive + FPnegative + FPneutral
    # tn = TNpositive + TNnegative + TNneutral
    # fn = FNpositive + FNnegative + FNneutral

    # prec = float(tp) / float(tp + fp)
    # recall = float(tp) / float(tp + fn)
    # micro_f = (float(2 * prec * recall)) / float(prec + recall)
    # acc = float(tp + tn) / float(tp + tn + fp + fn)

    # print prec, recall, micro_f, acc

    # print "ukupna preciznost: " + str((prec_positive + prec_neutral + prec_negative) / 3)
    # print "total recall: " + str((recall_neutral + recall_negative + recall_positive) / 3)
    # print "f mera total: " + str((f_neutral + f_negative + f_positive) / 3)
    # print "ukupna tacnost: " + str((acc_neutral + acc_positive +
    # acc_negative) / 3)

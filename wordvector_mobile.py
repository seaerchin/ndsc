import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from wordfunc import *
import pickle
import re
train = pd.read_csv("mobile_data_info_train_competition.csv")
eval = pd.read_csv("mobile_data_info_val_competition.csv")

count_vec_train = CountVectorizer(analyzer='word',
                            ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)
count_vec_eval = CountVectorizer(analyzer='word',
                            ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)

count_train = count_vec_train.fit(train.title)
count_eval = count_vec_eval.fit(eval.title)

trainwords = pd.DataFrame([count_train.vocabulary_], columns=count_train.vocabulary_.keys()).T.sort_values(by = [0])
evalwords = pd.DataFrame([count_eval.vocabulary_], columns=count_eval.vocabulary_.keys()).T.sort_values(by = [0])

count_train = list(count_train.vocabulary_.keys())
count_eval = list(count_eval.vocabulary_.keys())
worddifference = ([item for item in count_eval if item not in count_train])



word_add = []
for i in count_train:
    if len(i) >=3:
        word_add.append(i)

wording = words()
wording.addword(word_add)

counter = 0
for item in worddifference:
    counter += 1
    if len(item) > 20:
        print("Takes too long to process")
        print(item)
    else:
        result = combinelist(wording.segment(item))
        if len(result) > 1:
            #print(counter)
            regex = (r'\b%s\b' % item)
            eval['title'] = eval['title'].str.replace(regex, " ".join([x for x in result if len(x)>1]), regex=True)
            #eval['title'] = eval['title'].str.replace(item, " ".join([x for x in result if len(x)>1]))
            #replacedf(item, " ".join([x for x in result if len(x)>1]), eval)

regex = (r'\b%s\b' % "blackgoldpurplewhitesilver")
eval['title'] = eval['title'].str.replace(regex, "black gold purple white silver", regex=True)



#recalibrate the eval
count_vec_eval = CountVectorizer(analyzer='word',
                            ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)
count_eval = count_vec_eval.fit(eval.title)
evalwords = pd.DataFrame([count_eval.vocabulary_], columns=count_eval.vocabulary_.keys()).T.sort_values(by = [0])
count_eval = list(count_eval.vocabulary_.keys())
worddifference = ([item for item in count_eval if item not in count_train])

len(worddifference)

eval_save = []
train_save = []
counter = 0
for eva_item in worddifference:
    counter += 1
    count = []
    for train_item in count_train:
        count.append(levenshteinDistance(eva_item, train_item))
    eval_save.append(eva_item)
    train_save.append(count_train[count.index(max(count))])
    regex = (r'\b%s\b' % eva_item)
    eval['title'] = eval['title'].str.replace(regex, count_train[count.index(max(count))], regex=True)
    #eval['title'] = eval['title'].str.replace(eva_item, count_train[count.index(max(count))])
    print(counter)
    #replacedf(eva_item, count_train[count.index(max(count))], eval)


count_vec_eval = CountVectorizer(analyzer='word',
                            ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)
count_eval = count_vec_eval.fit(eval.title)
evalwords = pd.DataFrame([count_eval.vocabulary_], columns=count_eval.vocabulary_.keys()).T.sort_values(by = [0])
count_eval = list(count_eval.vocabulary_.keys())
worddifference = ([item for item in count_eval if item not in count_train])
len(worddifference)

eval.to_csv("mobile_new.csv", index = False)

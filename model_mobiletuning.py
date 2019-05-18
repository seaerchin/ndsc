from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import json

#data = ['Benefits', 'Brand', 'Colour_group', 'Product_texture', 'Skin_type']


json_data=open('mobile_profile_train.json').read()
data_j = json.loads(json_data)
df = pd.read_csv("mobile_data_info_train_competition.csv")
column = "Operating System"
#Phone Screen Size
res = dict((v,k) for k,v in data_j[column].items())
df = df.loc[df[column].notnull()]
df[column] = df[column].map(res)
df = df.sample(n = 500, random_state = 1)


add = df.shape[0]

df[column+"_id"], ylabel = df[column].factorize()
category_id_df = df[[column, column+"_id"]].drop_duplicates().sort_values(column+"_id")
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[[column, column+"_id"]].values)
labels = df[column+"_id"]

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.title).toarray()

features.shape
labels.shape
percent = 20
testno = int(features.shape[0]/100*(100 - percent))
train_X = features[:testno]
train_Y= labels[:testno]
test_X = features[testno:]
test_Y = labels[testno:]






#With Hyper Parameters Tuning
#2-3,SVM
#importing modules
from sklearn.model_selection import GridSearchCV
from sklearn import svm
#making the instance
model=svm.SVC()
#Hyper Parameters Set
params = {'C': [0.5, 1, 1.5, 2, 3],
          'kernel': ['rbf', 'poly', 'linear'],
          'gamma':['auto', 'scale'],
          'degree':[1,2],
          }
#Making models with hyper parameters sets
model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)
#Learning
model1.fit(train_X,train_Y)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction=model1.predict(test_X)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,test_Y))



model2 = LinearSVC()
model2.fit(train_X,train_Y)
prediction2 = model2.predict(test_X)
print("Accuracy:",metrics.accuracy_score(prediction2,test_Y))

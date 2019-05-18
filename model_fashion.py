from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import json
from sklearn import svm

json_data=open('beauty_profile_train.json').read()
data_j = json.loads(json_data)
df = pd.read_csv("beauty_data_info_train_competition.csv")
#Phone Screen Size

column = "Benefits"

# Index(['itemid', 'title', 'image_path', 'Benefits', 'Brand', 'Colour_group',
#        'Product_texture', 'Skin_type'],
#       dtype='object')

res = dict((v,k) for k,v in data_j[column].items())
df = df.loc[df[column].notnull()]
df[column] = df[column].map(res)

eval = pd.read_csv("beauty_new.csv")

df.shape
eval.shape
#df = df.sample(n=138000, random_state=1)
add = df.shape[0]
#add = 138000
df[column+"_id"], ylabel = df[column].factorize()
category_id_df = df[[column, column+"_id"]].drop_duplicates().sort_values(column+"_id")
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[[column, column+"_id"]].values)
labels = df[column+"_id"]

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.title.append(eval.title)).toarray()

features.shape

testfory = features[add:]
features = features[:add]

print(features.shape)
print(testfory.shape)

models = [
    svm.SVC(C = 1.3, kernel = 'rbf', gamma = 'scale', degree = 2)
]


#X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
for model in models:
    print("running")
    model_name = type(model).__name__
    model.fit(features, labels)
    y_pred = model.predict(testfory)
    pred_label = ylabel[y_pred]
    predicted = pred_label.map(data_j[column])
    predicted.shape
    index = eval['itemid'].astype(str)+"_"+str(column)
    result = pd.DataFrame(dict(s1 = index, s2 = predicted))
    result.to_csv(column+model_name+str(".csv"), index = False)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import LinearSVC
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
import json

#data = ['Benefits', 'Brand', 'Colour_group', 'Product_texture', 'Skin_type']

json_data=open('mobile_profile_train.json').read()
data_j = json.loads(json_data)
df = pd.read_csv("mobile_data_info_train_competition.csv")
column = "Network Connections"
#Phone Screen Size
res = dict((v,k) for k,v in data_j[column].items())
df = df.loc[df[column].notnull()]
df[column] = df[column].map(res)

eval = pd.read_csv("mobile_new.csv")

add = df.shape[0]
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
#Operating System, svm.SVC(C=2, kernel="linear", gamma='auto', degree=1)
#Feature default linear
#Network connections, svm.SVC(C=1, kernel="linear")
#memoryRam use default
#brand default
#warenty period svm.SVC(C=1.8, kernel="rbf", gamma = 'scale',degree=2)
#storage use linear
#color use linear
#phonemodel default
#camera, rbf, scale,1.4,2
#phone screen size, rbf, scale,2,2
    models = [
        LinearSVC()
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
    result.to_csv(column+model_name+str("new.csv"), index = False)

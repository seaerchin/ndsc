import pandas as pd
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
import os
import re
import collections
from sklearn.model_selection import train_test_split
from word_cnn import WordCNN
import tensorflow as tf

path="C:\\Users\\KangYu\\Desktop\\"
trainmobile = pd.read_csv(path+"/mobile_data_info_train_competition.csv",encoding = "ISO-8859-1")
TRAIN_PATH=path+"mobile_data_info_train_competition.csv"
TEST_PATH=path+"mobile_data_info_val_competition.csv"
WORD_MAX_LEN = 30
labely="Camera"

def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text

def build_word_dict():
    if not os.path.exists("word_dict.pickle"):
        train_df = pd.read_csv(TRAIN_PATH)
        contents = train_df["title"]

        words = list()
        for content in contents:
            for word in word_tokenize(clean_str(content)):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<eos>"] = 2
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    else:
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    return word_dict

def build_word_dataset(step, word_dict, document_max_len):
    if step == "train":
        df = pd.read_csv(TRAIN_PATH)
        df = df.loc[~np.isnan(df[labely])]
    else:
        df = pd.read_csv(TEST_PATH)

    # Shuffle dataframe
    df = df.sample(frac=1)
    x = list(map(lambda d: word_tokenize(clean_str(d)), df["title"]))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d + [word_dict["<eos>"]], x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))
    if step == "train":
        y = list(map(lambda d: d , list(df[labely])))
        return x,y
    else:
        ind = list(map(lambda d: d , list(df["itemid"])))
        return x,ind

word_dict = build_word_dict()
vocabulary_size = len(word_dict)
vocabulary_size
x, y = build_word_dataset("train", word_dict, WORD_MAX_LEN)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.20)
NUM_CLASS = 15
BATCH_SIZE = 8
NUM_EPOCHS = 100

def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]

def batch_iter_test(inputs, ind, batch_size, num_epochs):
    inputs = np.array(inputs)
    input2 = np.array(ind)
    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index],input2[start_index:end_index]


tf.reset_default_graph()
with tf.Session() as sess:
    model = WordCNN(vocabulary_size, WORD_MAX_LEN, NUM_CLASS)

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())

    train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)
    num_batches_per_epoch = (len(train_x) - 1) // BATCH_SIZE + 1
    max_accuracy = 0

    for x_batch, y_batch in train_batches:
        train_feed_dict = {
            model.x: x_batch,
            model.y: y_batch,
            model.is_training: True
        }
        print("running")
        _, step, loss = sess.run([model.optimizer, model.global_step, model.loss], feed_dict=train_feed_dict)
        print("stopped")
        if step % 100 == 0:
            print("step {0}: loss = {1}".format(step, loss))

        if step % 1000 == 0:
            # Test accuracy with validation data for each epoch.
            valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1)
            sum_accuracy, cnt = 0, 0

            for valid_x_batch, valid_y_batch in valid_batches:
                valid_feed_dict = {
                    model.x: valid_x_batch,
                    model.y: valid_y_batch,
                    model.is_training: False
                }
                accuracy = sess.run(model.accuracy, feed_dict=valid_feed_dict)
                sum_accuracy += accuracy
                cnt += 1
            valid_accuracy = sum_accuracy / cnt

            print("\nValidation Accuracy = {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))

            # Save model
            if valid_accuracy > max_accuracy:
                max_accuracy = valid_accuracy
                saver.save(sess, "{0}/{1}.ckpt".format("wordcnn", "wordcnn"), global_step=step)
                print("Model is saved.\n")


word_dict=build_word_dict()
test_x, ind= build_word_dataset("test", word_dict, WORD_MAX_LEN)
tf.reset_default_graph()
graph = tf.Graph()
prediction=[]
indlist=[]
with graph.as_default():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('C:/Users/leech/PycharmProjects/Intro/wordcnn/wordcnn.ckpt-6000.meta')
        saver.restore(sess, "C:/Users/leech/PycharmProjects/Intro/wordcnn/wordcnn.ckpt-6000")
        x = graph.get_operation_by_name("x").outputs[0]
        y = graph.get_operation_by_name("y").outputs[0]
        is_training = graph.get_operation_by_name("is_training").outputs[0]
        accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
        predictions = graph.get_operation_by_name("output/ArgMax").outputs[0]
        batches = batch_iter_test(test_x, ind, BATCH_SIZE, 1)
        for batch_x, ind in batches:
            feed_dict = {
                x: batch_x,
                is_training: False
            }

            prediction=np.concatenate([prediction,sess.run(predictions, feed_dict=feed_dict)])
            indlist=np.concatenate([indlist,ind])


print(prediction)
len(prediction)
len(indlist)
indlist
dataset = pd.DataFrame({'id':indlist,'tagging':prediction})
dataset['id']=dataset['id'].apply(lambda x: str(int(x))+"_"+labely.replace(" ", "_"))
dataset['tagging']=dataset['tagging'].astype(int)
path="C:/Users/leech/OneDrive/Desktop/NDSC/Results/"
dataset.to_csv(path+labely+".csv", index=False)

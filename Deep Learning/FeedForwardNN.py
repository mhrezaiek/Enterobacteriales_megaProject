import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, concatenate
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import csv
from random import random
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras.backend as K
from keras.initializers import RandomUniform, TruncatedNormal
import matplotlib as mpl
mpl.use('Agg')
import matplotlib . pyplot as plt
import gc
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

def genome_id_creator(path):
        gene_list = path
        genome_id = []
        with open(gene_list) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                genome_id.append(row)
        return genome_id
def dataset_creator(data, lable_path):
    dataset = data

    lable = genome_id_creator(lable_path)
    lable = pd.DataFrame(lable[1:], columns=['genome_id', lable_name, 'Gentamicin_val'])
    lable['genome_id'] = lable['genome_id'].astype(str)
    print(lable_name + "added", flush=True)

    merged = dataset.merge(lable, left_on=['id'], right_on=lable['genome_id'], how='right')
    merged = pd.DataFrame(merged)
    merged = merged.dropna()
    dataset = np.asmatrix(merged.iloc[0:, 1:merged.shape[1] - 3])
    dataset = np.asarray(dataset)
    y = np.asmatrix(merged.iloc[0:, merged.shape[1] - 2])
    y = np.asarray(y, dtype=np.float)
    y = np.reshape(y, (merged.shape[0], 1))

    return dataset, y
def model(input_shape, Nlayer):
    n = 7000
    input1 = Input(shape=(input_shape,))
    x = Dense(7000, kernel_initializer=RandomUniform(), bias_initializer=TruncatedNormal(), activation='relu' )(input1)
    for i in range(Nlayer):
        n = n / 2
        x =Dense(n, kernel_initializer=RandomUniform(), bias_initializer=TruncatedNormal(), activation='relu')(x)
        x = tf.keras.layers.Dropout(.2)(x)

    out = tf.keras.layers.Dense(1, kernel_initializer = RandomUniform(), bias_initializer = TruncatedNormal(), activation = 'sigmoid')(x)

    model = tf.keras.models.Model(inputs=input1, outputs=out)
    return model
def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)
def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.sum(K.cast(K.not_equal(y_true, -1), dtype))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
    return correct / total
def plot(drugs, method_name):
    plt.ioff()
    lw = 2
    i = 0
    plot_name = str(method_name)+".pdf"
    for drug in drugs:
        lable = drug
        plt.plot(fpr[i], tpr[i], lw=lw, label=str(lable) + '(area = %0.2f)' % roc_auc[i])
        i = i+1
    plt.legend(loc="lower right", prop={'size': 7})
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.savefig(plot_name)
    plt.close()


print("commencing", flush = True)
drugs = ["amikacin","Amoxicillin","Ampicillin","Aztreonam","cefalotin","Cefepime","Cefotaxime","Cefoxitin","Ceftazidime","ceftriaxone","Cefuroxime","Ciprofloxacin","ertapenem", "Gentamicin", "imipenem", "meropenem", "tetracycline"]

fpr = []
tpr = []
roc_auc = []

data = genome_id_creator("../ML/sub_dataset.csv")
data = pd.DataFrame(data[1:],columns=data[0] )
data['id'] = data['id'].astype(str)

for drug in drugs:
    gc.collect()
    lable_name = "dataset_perdrugs/"+drug + "_final.csv"
#    lable_name = "../ML/dataset_perdrugs/" + drug + "_final.csv"
    dataset, y = dataset_creator(data, lable_name)
    dataset , y = dataset.astype(float), y.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.3, random_state=23)

    nn = model(99,3)
    Adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    nn.compile(loss=masked_loss_function, optimizer=Adam, metrics=[masked_accuracy])
    print('model complied')
    CIP = nn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64, verbose=1,shuffle=True)
    #print(dataset)

    y_score = nn.predict(X_test).ravel()
    fpr1, tpr1, _ = roc_curve(y_test, y_score)
    roc_aucc = auc(fpr1, tpr1)
    fpr.append(fpr1)
    tpr.append(tpr1)
    roc_auc.append(roc_aucc)


plot(drugs, "deep")








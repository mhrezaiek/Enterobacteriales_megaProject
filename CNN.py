import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, concatenate, Conv1D
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.initializers import RandomUniform, TruncatedNormal
import matplotlib
#matplotlib.use('Agg')
import matplotlib. pyplot as plt
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
def model(input_shape, Nlayer):
    n = 100
    input1 = Input(shape=(input_shape,))
    x = Dense(100, kernel_initializer=RandomUniform(), bias_initializer=TruncatedNormal(), activation='relu' )(input1)
    x = tf.keras.layers.Reshape([100, 1])(x)
    for i in range(Nlayer):
        n = int(n / 2)
        x =Conv1D(n,strides=2,kernel_size=2,padding='same', kernel_initializer=RandomUniform(), bias_initializer=TruncatedNormal(), activation='relu')(x)
        x = tf.keras.layers.Dropout(.2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = Dense(50, kernel_initializer=RandomUniform(), bias_initializer=TruncatedNormal(), activation='relu')(x)
    out = tf.keras.layers.Dense(17, kernel_initializer = RandomUniform(), bias_initializer = TruncatedNormal(), activation = 'sigmoid')(x)

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

data = genome_id_creator("../../../Desktop/sample.csv")
data = pd.DataFrame(data[1:],columns=data[0] )

gc.collect()
#dataset = np.asmatrix(data.iloc[0:,2:378969])
dataset = np.asmatrix(data.iloc[0:,3:6])
dataset = np.asarray(dataset)
dataset = np.reshape(dataset, (-1,3,1))
y = np.asmatrix(data.iloc[0:,6:])

y = np.asarray(y, dtype=np.float)
y = np.reshape(y, (2020, 17))
dataset , y = dataset.astype(float), y.astype(float)

X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.3, random_state=23)

nn = model((3,1),1)
Adam = tf.keras.optimizers.Adam(learning_rate=0.001)
nn.compile(loss=masked_loss_function, optimizer=Adam, metrics=[masked_accuracy])
print('model complied',flush = True)
CIP = nn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16, verbose=2,shuffle=True)
print("Done" , flush = True)

y_score = nn.predict(X_test).ravel()
y_score = np.reshape(y_score, (-1,17))





for i in range(17):
    temp_y_test = []
    temp_y_score = []
    for j in range(y_score.shape[0]):
        if  y_test[j,i] != -1:
            temp_y_test.append(y_test[j,i])
            temp_y_score.append(y_score[j,i])


    fpr1, tpr1, _ = roc_curve(temp_y_test, temp_y_score)
    roc_aucc = auc(fpr1, tpr1)
    fpr.append(fpr1)
    tpr.append(tpr1)
    roc_auc.append(roc_aucc)
plot(drugs, "CNN")
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

def genome_id_creator(path):
        gene_list = path
        genome_id = []
        with open(gene_list) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            for row in readCSV:
                genome_id.append(row)
        return genome_id

fpr = []
tpr = []
roc_auc = []


def dataset_creator(data, lable_path):
    dataset = data

    lable = genome_id_creator(lable_path)
    lable = pd.DataFrame(lable[1:], columns=['genome_id', lable_name, 'Gentamicin_val'])
    lable['genome_id'] = lable['genome_id'].astype(str)
    print(lable_name + "added", flush=True)

    merged = dataset.merge(lable, left_on=['id'], right_on=lable['genome_id'], how='right')
    merged = pd.DataFrame(merged)
    merged = merged.dropna()
    dataset = np.asmatrix(merged.iloc[0:, 0:merged.shape[1] - 3])
    dataset = np.asarray(dataset)
    y = np.asmatrix(merged.iloc[0:, merged.shape[1] - 2])
    y = np.asarray(y, dtype=np.float)
    y = np.reshape(y, (merged.shape[0], 1))

    return dataset, y


print("commencing", flush = True)


drugs = ["amikacin","Amoxicillin","Ampicillin","Aztreonam","cefalotin","Cefepime","Cefotaxime","Cefoxitin","Ceftazidime","ceftriaxone","Cefuroxime","Ciprofloxacin","ertapenem", "Gentamicin", "imipenem", "meropenem", "tetracycline"]



data = genome_id_creator("../ML/sub_dataset.csv")
data = pd.DataFrame(data[1:],columns=data[0] )
data['id'] = data['id'].astype(str)

for drug in drugs:
    gc.collect()
#    lable_name = "dataset_perdrugs/"+drug + "_final.csv"
    lable_name = "../ML/dataset_perdrugs/" + drug + "_final.csv"
    dataset, y = dataset_creator(data, lable_name)

    X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.3, random_state=23)
    
    print(dataset,y)








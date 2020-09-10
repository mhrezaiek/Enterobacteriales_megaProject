import pandas as pd
import numpy as np
import csv
import os
import math
import gc

from sklearn.model_selection import train_test_split
from sklearn import svm
from os import listdir
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import matplotlib as mpl
mpl.use('Agg')
import matplotlib . pyplot as plt

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

drugs = ["amikacin","Amoxicillin","Ampicillin","Aztreonam","cefalotin","Cefepime","Cefotaxime","Cefoxitin","Ceftazidime","ceftriaxone","Cefuroxime","Ciprofloxacin","ertapenem", "Gentamicin", "imipenem", "meropenem", "tetracycline"]

print("commencing", flush = True)



data = genome_id_creator("../snp_dataset.csv")
#data = genome_id_creator("../snp_dataset.csv")
data = pd.DataFrame(data[1:],columns=data[0] )
data['id'] = data['id'].astype(str)




for drug in drugs:
    gc.collect()
    dataset = data
    lable_name = "dataset_perdrugs/"+drug + "_final.csv"
    print(lable_name)
    lable = genome_id_creator(lable_name)
    lable = pd.DataFrame(lable[1:],columns=['genome_id',lable_name,'Gentamicin_val'] )
    lable['genome_id'] = lable['genome_id'].astype(str)
    print(lable_name +"added", flush = True)


    merged = dataset.merge(lable, left_on=['id'] , right_on=lable['genome_id'], how='right')
    merged = pd.DataFrame(merged)
    merged = merged.dropna()
    dataset = np.asmatrix(merged.iloc[0:,0:merged.shape[1]-3])
    dataset = np.asarray(dataset)
    y = np.asmatrix(merged.iloc[0:,merged.shape[1]-2])
    y = np.asarray(y, dtype=np.float)
    y =np.reshape(y,(merged.shape[0],1))
    X_train, X_test, y_train, y_test= train_test_split(dataset, y, test_size=0.3, random_state=23)

    
    
    
    
    from sklearn.linear_model import LogisticRegression
    model= LogisticRegression()
    model.fit(X_train,y_train)
    y_pred= model.predict(X_test)
    y_score = model.predict_proba(X_test)
    y_score = y_score[:,1]
    
    
    fpr1, tpr1, _ = roc_curve(y_test, y_score)
    roc_aucc = auc(fpr1, tpr1)
    
    fpr.append(fpr1)
    tpr.append(tpr1)
    roc_auc.append(roc_aucc)
    del dataset


plt.ioff()
lw = 2
plt.plot(fpr[0], tpr[0] , lw=lw, label='amikacin (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1] , lw=lw, label='Amoxicillin (area = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2] , lw=lw, label='Ampicillin (area = %0.2f)' % roc_auc[2])
plt.plot(fpr[3], tpr[3] , lw=lw, label='Aztreonam (area = %0.2f)' % roc_auc[3])
plt.plot(fpr[4], tpr[4] , lw=lw, label='cefalotin (area = %0.2f)' % roc_auc[4])
plt.plot(fpr[5], tpr[5] , lw=lw, label='Cefepime (area = %0.2f)' % roc_auc[5])
plt.plot(fpr[6], tpr[6] , lw=lw, label='Cefotaxime (area = %0.2f)' % roc_auc[6])
plt.plot(fpr[7], tpr[7] , lw=lw, label='Cefoxitin (area = %0.2f)' % roc_auc[7])
plt.plot(fpr[8], tpr[8] , lw=lw, label='Ceftazidime (area = %0.2f)' % roc_auc[8])
plt.plot(fpr[9], tpr[9] , lw=lw, label='ceftriaxone (area = %0.2f)' % roc_auc[9])
plt.plot(fpr[10], tpr[10] , lw=lw, label= 'Cefuroxime (area = %0.2f)' % roc_auc[10])
plt.plot(fpr[11], tpr[11] , lw=lw, label='Ciprofloxacin (area = %0.2f)' % roc_auc[11])
plt.plot(fpr[12], tpr[12] , lw=lw, label='ertapenem (area = %0.2f)' % roc_auc[12])
plt.plot(fpr[13], tpr[13] , lw=lw, label='Gentamicin (area = %0.2f)' % roc_auc[13])
plt.plot(fpr[14], tpr[14] , lw=lw, label='imipenem (area = %0.2f)' % roc_auc[14])
plt.plot(fpr[15], tpr[15] , lw=lw, label='meropenem (area = %0.2f)' % roc_auc[15])
plt.plot(fpr[16], tpr[16] , lw=lw, label='tetracycline (area = %0.2f)' % roc_auc[16])

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig("Overall_ROC_LR.pdf")
plt.close()


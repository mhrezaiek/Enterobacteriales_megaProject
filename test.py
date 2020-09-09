import pandas as pd
import numpy as np
import csv
import os
import math

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn import svm
from os import listdir
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt



def genome_id_creator(path):
    gene_list = path
    genome_id = []
    with open(gene_list) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            genome_id.append(row)
    return genome_id


dataset = genome_id_creator("../../../Desktop/snp_dataset.csv")
dataset = pd.DataFrame(dataset[1:],columns=dataset[0] )
dataset['id'] = dataset['id'].astype(str)


lable = genome_id_creator("datasets_perdrug/Gentamicin_1.csv")
lable = pd.DataFrame(lable[1:],columns=['genome_id','Gentamicin','Gentamicin_val'] )
lable['genome_id'] = lable['genome_id'].astype(str)



merged = dataset.merge(lable, left_on=['id'] , right_on=lable['genome_id'], how='right')


merged = pd.DataFrame(merged)
merged = merged.dropna()
merged = merged.drop(['genome_id'], axis=1)
print(merged)


dataset = np.asmatrix(merged.iloc[0:,0:merged.shape[1]-2])
dataset = np.asarray(dataset)

y = np.asmatrix(merged.iloc[0:,merged.shape[1]-2])
y = np.asarray(y, dtype=np.float)
y =np.reshape(y,(merged.shape[0],1))

X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size=0.3, random_state=23)
supportVectore = svm.SVC()
y_score = supportVectore.fit(X_train, y_train).decision_function(X_test)

fpr = []
tpr = []
roc_auc = []

fpr1, tpr1, _ = roc_curve(y_test, y_score)
roc_aucc = auc(fpr1, tpr1)
fpr.append(fpr1)
tpr.append(tpr1)
roc_auc.append(roc_aucc)

fig1= plt.gcf()

lw = 2
plt.plot(fpr[0], tpr[0],
         lw=lw, label='CTZ (area = %0.2f)' % roc_auc[0])

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve ')
plt.legend(loc="lower right")
plt.show()

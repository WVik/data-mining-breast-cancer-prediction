__author__ = 'Vikram'
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd


#Uncomment this and comment the lines after that for a web-csv read
"""dataFrame = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
                       names = ['id','clump_thickness', "unif_cell_size", "unif_cell_shape", "marg_adhesion", "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"])
"""

dataFrame = pd.read_csv('/Users/Vikram/Documents/Projects/Data-Mining-Breast-Cancer/data/breast-cancer-wisconsin.data',
                       names = ['id','clump_thickness', "unif_cell_size", "unif_cell_shape", "marg_adhesion", "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"])

dataFrame.replace('?','-9999999', inplace=True)
dataFrame.drop(['id'],1,inplace=True)

#print(dataFile['clump_thickness'])

X = np.array(dataFrame.drop('class',1))
y = np.array(dataFrame['class'])

kf = model_selection.KFold(n_splits = 10);
globalAccuracy = 0.0

#print(kf.split(X))

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(train_index)
    print("--------")
    clf = neighbors.KNeighborsClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)
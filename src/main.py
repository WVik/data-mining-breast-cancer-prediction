__author__ = 'Vikram'
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

dataFrame = pd.read_csv('/Users/Vikram/Documents/Projects/Data-Mining-Breast-Cancer/data/breast-cancer-wisconsin.data',
                       names = ['id','clump_thickness', "unif_cell_size", "unif_cell_shape", "marg_adhesion", "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"])
dataFrame.replace('?',-99999, inplace=True)
dataFrame.drop(['id'],1,inplace=True)

#print(dataFile['clump_thickness'])

X = np.array(dataFrame.drop('class',1))
y = np.array(dataFrame['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
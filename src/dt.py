import numpy as np
from sklearn import preprocessing, model_selection, tree, metrics
import pandas as pd
import plotly
plotly.tools.set_credentials_file(username = 'WVik', api_key='MyiOZkG5iObapIYvwKVT')
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from subprocess import check_call


import time
start_time = time.time()

dataFrame = pd.read_csv('/Users/Vikram/Documents/Projects/Data-Mining-Breast-Cancer/data/breast-cancer-wisconsin.data',
                       names = ['id','clump_thickness', "unif_cell_size", "unif_cell_shape", "marg_adhesion", "single_epith_cell_size", "bare_nuclei", "bland_chrom", "norm_nucleoli", "mitoses", "class"])


#Drop missing data values
dataFrame.replace('?',np.nan, inplace=True)
dataFrame.dropna(0,'any',inplace=True)


#Drop the ID column
dataFrame.drop(['id'],1,inplace=True)


#Separate attributes and class
X = np.array(dataFrame.drop('class',1))
y = np.array(dataFrame['class'])

#Create one decision tree

X_train_temp,X_test_temp,y_train_temp,y_test_temp = model_selection.train_test_split(X,y,test_size=0.3)

clf = tree.DecisionTreeClassifier(criterion="gini",max_depth=15,min_samples_leaf=10).fit(X_train_temp,y_train_temp)
tree.export_graphviz(clf,
    out_file='tree.dot')

check_call(['dot','-Tpng','tree.dot','-o','OutputFile.png'])



#Cross Validation
#Set up 10 fold cross-validation
kf = model_selection.KFold(n_splits = 10);
globalAccuracy = 0.0


for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(train_index)
    print("--------")
    clf_gini = tree.DecisionTreeClassifier(criterion="gini",max_depth=30,min_samples_leaf=20)
    clf_gini.fit(X_train, y_train)
    accuracy = clf_gini.score(X_test, y_test)
    metrics.confusion_matrix(y_test, clf.predict(X_test))
    print(metrics.classification_report(y_test, clf_gini.predict(X_test)))
    print(accuracy)
    globalAccuracy += accuracy

print()
print()

print(globalAccuracy/10)

print(time.time() - start_time)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
import seaborn as sns

###Add Data to Pandas DataFrame
df = pd.read_csv('../Data/breast-cancer-wisconsin.data', header=None, names=['ID', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class'])
df = df.set_index('ID')
df['Class'].replace(2, 0, inplace=True)
df['Class'].replace(4, 1, inplace=True)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
print('--> Added Data into Pandas Data Frame')

#Split into training and test set
y = df['Class']
df.drop('Class', axis=1, inplace=True)
X = df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('--> Split Train Data to Trainign and Test Sets')


from sklearn.ensemble import RandomForestClassifier

sns.set(style='whitegrid')

feat_labels = df.columns
clf = RandomForestClassifier(n_estimators = 100, random_state=0, n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)


importances = clf.feature_importances_
indicies = np.argsort(importances)[::-1]
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]),
        importances[indicies],
        color='skyblue',
        align='center')
plt.xticks(range(X_train.shape[1]),
           feat_labels[indicies], rotation = 90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
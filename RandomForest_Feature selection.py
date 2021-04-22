"""FEATURE SELECTION RANDOMFOREST"""
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

#import data
df = pd.read_csv('6m3.csv', encoding='utf-8')
#df = pd.DataFrame(df, dtype='float')
#data = df.convert_objects(convert_numeric=True)
df.columns = [ 'mrs6m','Sex','Age','Onset 4.5h','Hypertension','Diabetes mellitusry disease','Hyperlipidemia',
                   'Coronary artery disease','Atrial fibrillation', 'Previous stroke', 'Drinking', 'Smoking', 'NIHSS',
                   'Weight', 'height','BMI','SBP','DBP','PLT count','INR','CREA','FBG',
              'TC','TG','HDL','LDL','HbA1c','Previous antiplatelet','IV thrombolysis']

print('class labels:', np.unique(df['mrs6m']))
# Split training set & test set
#x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
seed=7
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=seed)

# Eigenvalue scaling - normalization, decision tree model does not rely on feature scaling
stdsc=StandardScaler()
x_train_std=stdsc.fit_transform(x_train)
x_test_std=stdsc.fit_transform(x_test)

# The importance of features in random forest assessment
feat_labels = df.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, n_jobs=-1, random_state=0)
forest.fit(x_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(x_train.shape[1]):
    #print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[[indices[f]]], importances[indices[f]]))
#  feature importance visualization -decay based on average impurity
plt.title('Feature Importance-RandomForest')
plt.bar(range(x_train.shape[1]), importances[indices], color='lightblue', align='center')
#plt.xticks(range(x_train.shape[1]), feat_labels, rotation=90)
plt.xticks(range(x_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
plt.show()

# variable importance measures

x_selected = forest.transform(x_train, threshold=0.05)  #
print(x_selected.shape)
threshold = 0.05
x_selected = x_train[:, importances > threshold]
print(x_selected.shape)



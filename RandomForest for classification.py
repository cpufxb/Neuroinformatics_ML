"""RANDOMFOREST FOR CLASSIFICATION"""
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
#loda data
data = pd.read_csv("stroke6.csv").values
x = data[:, :-1] #
y = data[:, -1] #

seed=25
X_train, x_test,Y_train, y_test = train_test_split(x, y, test_size=0.25,random_state=seed)

# over-sampling
from imblearn.over_sampling import SMOTE
# definition of SMOTE
smo = SMOTE(random_state=15) #sampling_strategy={1: 188 },
#smo=SMOTEENN(random_state=15)
x, y = smo.fit_sample(x, y)
#x_train, y_train = smo.fit_sample(x_train, y_train)


x_train, x_1,y_train, y_1 = train_test_split(x, y, test_size=0.47,random_state=seed)
from collections import Counter
# View the category distribution of the generated samples
print(Counter(y))
print(Counter(y_train))
print(Counter(y_test))

minmax = MinMaxScaler()
x_train = minmax.fit_transform(x_train)
x_test = minmax.fit_transform(x_test)

#fit model
RF = RandomForestClassifier(max_depth=7,n_estimators=100) #n_jobs=-1
RF.fit(x_train, y_train)

y_score=RF.fit(x_train,y_train).predict_proba(x_test)

y_pred = RF.predict(x_test)
predictions = [round(value) for value in y_pred]
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Compute ROC curve and ROC area for each class
print('AUC：')
fpr,tpr,threshold = roc_curve(y_test, y_score[:, 1] ) ###
roc_auc = auc(fpr,tpr) ###calculation the value of auc
print(roc_auc)

#plt
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %.2f)' % roc_auc)##
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic RF')
plt.legend(loc="lower right")
plt.show()




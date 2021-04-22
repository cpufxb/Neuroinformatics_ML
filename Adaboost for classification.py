"""ADABOOST FOR CLASSIFICATION"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
#load data
data = pd.read_csv("6m3.csv").values
x = data[:, :-1] #
y = data[:, -1] #

seed=20
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

# fit model
model=AdaBoostClassifier(DecisionTreeClassifier (max_depth=25,min_samples_split=2,min_samples_leaf=5),
                         algorithm="SAMME.R",
                         n_estimators=100,learning_rate=0.01)#
# train model
model.fit(x_train, y_train)

y_score=model.fit(x_train,y_train).predict_proba(x_test)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Compute ROC curve and ROC area for each class
print('AUC：')
fpr,tpr,threshold = roc_curve(y_test,y_score[:,1])
roc_auc = auc(fpr,tpr) #calculate the value of auc
print(roc_auc)

lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %.2f)' % roc_auc)#
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of adaboost')
plt.legend(loc="lower right")
plt.show()


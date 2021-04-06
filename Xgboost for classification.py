"""XGBOOST FOR CLASSIFICATION"""
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report,confusion_matrix
#load data
data = pd.read_csv("6m3.csv").values
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

#build model
model = XGBClassifier(
silent=0 ,
#nthread=4,#
learning_rate= 0.3, # lr
min_child_weight=1,
max_depth=5, #
gamma=0.1,  # 0.1 0.2
subsample=0.8,
max_delta_step=0,
colsample_bytree=0.6,
reg_lambda=1,  # L2 Regularization
#reg_alpha=0, # L1 Regularization
scale_pos_weight=1,
#objective= 'multi:softmax', #
#num_class=10, #
n_estimators=500, #number of trees
seed=1000, #
 )


eval_set = [(x_test, y_test)]
model.fit(x_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=False)

y_score=model.fit(x_train,y_train).predict_proba(x_test)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
print("Accuracy: %.2f%%" % (accuracy * 100.0))
##

fpr, tpr, threshold1 = roc_curve(y_test, y_score[:, 1])  ###
roc_auc = auc(fpr, tpr)  #calculation the value of auc
print(roc_auc)

lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %.2f)' % roc_auc)##
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic of xgboost')
plt.legend(loc="lower right")
plt.show()


#feature importance visualization
from xgboost import plot_importance
from matplotlib import pyplot
model.fit(x, y)
plot_importance(model)
pyplot.show()

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.3,0.1, 0.01, 0.001]}
grid = GridSearchCV(model, param_grid, refit=True, verbose=2)
grid.fit(x_train, y_train)
# predictions
grid_predictions = grid.predict(x_test)
# classification_report
print(classification_report(y_test, grid_predictions))

import numpy as np
import pandas as pd
import math
import glob
import matplotlib.pyplot as plt
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble.partial_dependence import partial_dependence
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
fig = plt.figure()
target_feature = (1, 3)
pdp, axes = partial_dependence(model, target_feature, X=X_train, grid_resolution=50)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel([target_feature[0]])
ax.set_ylabel([target_feature[1]])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.subplots_adjust(top=0.9)
plt.show()
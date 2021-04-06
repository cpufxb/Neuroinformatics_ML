"""PDP FOR XGBCLASSIFICER"""
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
import eli5   # permutation importance
from eli5.sklearn import PermutationImportance
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

path = 'E:\\8.csv'
data = pd.read_csv(path)
#
cols_to_use = ['Age','DM','FBG','NIHSS', 'LDL', 'HDL','CREA','INR']
y = data.mrs6m
x= data[cols_to_use]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 400)

# XGBoost
model = xgb.XGBClassifier(
                        learning_rate =0.05,
                         n_estimators=100,
                         max_depth=3,
                         min_child_weight=1,
                         gamma=0.3,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         objective= 'multi:softprob',
                         nthread=4,
                         scale_pos_weight=1,
                         num_class=2,
                         seed=27
                    ).fit(X_train, y_train)

perm = PermutationImportance(model, random_state = 1).fit(X_test, y_test) #
eli5.show_weights(perm, feature_names = X_test.columns.tolist())
#plt.show(perm)

from pdpbox import pdp

feature = 'HDL'
pdp_goals = pdp.pdp_isolate(model, X_train, x.columns, feature)
# draw partial dependence plot of feature
pdp.pdp_plot(pdp_goals, feature)
plt.show()


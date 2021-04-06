"""PDP FIGURES"""
# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter
import joblib
#from sklearn.externals import joblib
import warnings
warnings.filterwarnings("ignore")

cols_to_use = ['Age','DM','FBG','NIHSS', 'LDL', 'HDL','CREA','INR']

def get_some_data():
    data = pd.read_csv('8_6m.csv')
    y = data.mrs6m
    X = data[cols_to_use]
    my_imputer = SimpleImputer()
    imputed_X = my_imputer.fit_transform(X)
    return imputed_X, y

X, y = get_some_data()
my_model = GradientBoostingRegressor()
my_model.fit(X, y)
myplots  = plot_partial_dependence(my_model,
                                   features=[0],
                                   X=X,
                                   feature_names=cols_to_use,
                                   grid_resolution=10)
plt.show(myplots)

GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.1, loss='ls', max_depth=3, max_features=None,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=1,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=100, n_iter_no_change=None, presort='auto',
             random_state=None, subsample=1.0, tol=0.0001,
             validation_fraction=0.1, verbose=0, warm_start=False)
#
from sklearn.ensemble.partial_dependence import plot_partial_dependence

my_plots = plot_partial_dependence(my_model,
                                   feature_names=cols_to_use,
                                   features=[0],
                                   X=X)
plt.show(my_plots)


#3D
names = cols_to_use
fig = plt.figure()
target_feature = (0, 6)
pdp, axes = partial_dependence(my_model, target_feature, X=X, grid_resolution=50)
XX, YY = np.meshgrid(axes[0], axes[1])
Z = pdp[0].reshape(list(map(np.size, axes))).T
ax = Axes3D(fig)
surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap=plt.cm.BuPu, edgecolor='k')
ax.set_xlabel(names[target_feature[0]])
ax.set_ylabel(names[target_feature[1]])
ax.set_zlabel('Partial dependence')
#  pretty init view
ax.view_init(elev=22, azim=122)
plt.colorbar(surf)
plt.subplots_adjust(top=0.9)
plt.show()





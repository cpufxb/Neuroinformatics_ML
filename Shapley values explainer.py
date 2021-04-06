"""SHAPLEY VALUES EXPLAINER"""
# -*- coding:utf-8 -*-
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('seaborn')

#load dadta
data = pd.read_csv('8_6m.csv')
cols = ['Age','DM','FBG','NIHSS', 'LDL', 'HDL','CREA','INR']

model = xgb.XGBRegressor(max_depth=4, learning_rate=0.05, n_estimators=150)
model.fit(data[cols], data['amount'].values)


importance_frame = pd.DataFrame({'Importance': list(model.feature_importances_), 'Feature': list(cols)})
importance_frame.sort_values(by='Importance', inplace=True)
#importance_frame['rela_imp'] = importance_frame['Importance'] / sum(importance_frame['Importance'])
importance_frame.plot(kind='barh', x='Feature', figsize=(8, 8), color='orange')
plt.show()

import xgboost
import shap
#shap.initjs()
explainer = shap.TreeExplainer(model)

#demension of shap_values
shap_values = explainer.shap_values(data[cols])
print(shap_values.shape)

#The base value is the mean value of the fitted value of the target variable in the training set
y_base = explainer.expected_value
print(y_base)

data['pred'] = model.predict(data[cols])
print(data['pred'].mean())

#The SHAP value of a single sample；j=j+2
j = 18   #The location of the selected sample
j=j-1
player_explainer = pd.DataFrame()
player_explainer['feature'] = cols
player_explainer['feature_value'] = data[cols].iloc[j].values
player_explainer['shap_value'] = shap_values[j]
print(player_explainer)
#
print('y_base + sum_of_shap_values: %.2f'%(y_base + player_explainer['shap_value'].sum()))
print('y_pred: %.2f'%(data['pred'].iloc[j]))

shap.initjs()
#matplotlib=True
shap.force_plot(explainer.expected_value, shap_values[j], data[cols].iloc[j],matplotlib=True)
#plt.show(shap.force_plot)

# shap.force_plot(explainer.expected_value, shap_values,data[cols])
# plt.show(shap.force_plot)

#Overall analysis of features
shap.summary_plot(shap_values, data[cols])

"""Because the calculation methods of SHAP and feature_importance are different, 
we also get a different importance ranking from the previous one."""
shap.summary_plot(shap_values, data[cols], plot_type="bar", color='orange')

#Partial Dependence Plot
shap.dependence_plot('Age', shap_values, data[cols], interaction_index=None, show=False)
plt.show(shap.dependence_plot)

#inreraction analysis of multivariates
shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(data[cols])
shap.summary_plot(shap_interaction_values, data[cols], max_display=4)

#Dependence_plot is used to describe the influence of two variables on the target value
shap.dependence_plot('Age', shap_values, data[cols], interaction_index='HDL', show=False)
plt.show(shap.dependence_plot)
import pandas as pd
import numpy as np
import graphviz.backend as be
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn import tree
import dtreeviz as dt

health_df = pd.read_csv("D:\PythonData\Heart_attack.csv")
y = health_df['Heart_Disease']

health_features = ['Chest_pain','Blocked_Arteries','Patient_Weight']
X = health_df[health_features]
for feature in ['Chest_pain','Blocked_Arteries']:
        if feature=='Chest_pain':
           X['Chest_pain'] = X['Chest_pain'].map({'Yes':1,'No':0})
        if feature=='Chest_pain':
           X['Blocked_Arteries'] = X['Blocked_Arteries'].map({'Yes':1,'No':0})
print(X)
print(y)
health_model = DecisionTreeClassifier(random_state=1,criterion='gini',max_depth=4)
health_model.fit(X, y)
plt.figure(figsize=(18,18))
tree.plot_tree(health_model, fontsize=9)
plt.show()
"""
viz = dt.model(health_model, X_train=X, y_train=y, feature_names=health_features, target_name='Heart_Disease',class_names=['Yes','No'])
print(type(viz))
v = viz.view(scale=2,fancy=True, orientation='LR',max_X_features_LR=10, max_X_features_TD=5)
v.show()
"""
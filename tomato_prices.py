import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
from sklearn import tree

tomato_df = pd.read_csv('D:/pythondata/tomato_prices.csv')
y = tomato_df['Price']
price_features = ['Supply','Demand']
X = tomato_df[price_features]

tomato_price_model = DecisionTreeRegressor(random_state=1,criterion='squared_error',max_depth=3)
tomato_price_model.fit(X, y)
plt.figure(figsize=(18,18))
tree.plot_tree(tomato_price_model, fontsize=9, feature_names=['Sup','Dem'], class_names=True)

pred_1_dict = {'Supply':[50],
               'Demand':[150]}
pred_1_X = pd.DataFrame(pred_1_dict)
pred_1_y = tomato_price_model.predict(pred_1_X)
print('Prediction 1 - Supply 50, Demand 150 ', pred_1_y)

pred_2_dict = {'Supply':[50],
               'Demand':[60]}
pred_2_X = pd.DataFrame(pred_2_dict)
pred_2_y = tomato_price_model.predict(pred_2_X)
print('Prediction 2 - Supply 50, Demand 60 ', pred_2_y)

plt.show()
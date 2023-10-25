import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

comedian_df = pd.read_csv("comedian.csv")
country_dict = {"UK":0, "USA":1, "N":2}
see_not_see_dict = {"NO":0, "YES":1}
comedian_df["Nationality"] = comedian_df["Nationality"].map(country_dict)
comedian_df["Go"] = comedian_df["Go"].map(see_not_see_dict)

independent_features = ["Age","Experience","Rank","Nationality"]
dependent_feature = ["Go"]

X = comedian_df[independent_features]
y = comedian_df[dependent_feature]

dtree = DecisionTreeClassifier()
dtree.fit(X,y)
y1 = dtree.predict([[43, 21, 8, 0]])
txt = 'The prediction for someone with Age 40, Exp 10, Rank 6 and from Netherlands is {} '
print(txt.format(y1))

tree.plot_tree(dtree, feature_names=independent_features)
plt.show()


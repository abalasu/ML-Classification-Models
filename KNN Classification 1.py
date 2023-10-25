import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt


att_df = pd.read_csv("D:\PythonData\KNN_Classification.csv")
# print(att_df.describe())
y = att_df['Attrition']
att_features = ['Age', 'Salary']
X = att_df[att_features]
# n_neighbors - number of neighbors to look at weight - uniform weight for all neighbors or distance - actual distance is considered
# p - 1 for manhattan distance and 2 for euclidean distance
att_model = KNeighborsClassifier(n_neighbors=4, p=2, weights='distance')
att_model.fit(X.values, y.values)

test_X_mstr = [(37,25),(21,12),(45,30),(30,27)]

plt_x = att_df['Age']
plt_y = att_df['Salary']
plt.scatter(plt_x,plt_y)
i = 0
while i < len(plt_y):
    x1 = plt_x[i]
    y1 = plt_y[i]
    print(y[i])
    if y[i] == 'No':
        plt.plot(x1, y1, marker='d', ms=6, c='b')
    else:
        plt.plot(x1, y1, marker='*', ms=6, c='r')
    i += 1
plt.title('Attrition Prediction')
plt.xlabel('Age')
plt.ylabel('Salary in Lakhs')
plt.xlim(20,60)
plt.ylim(0,50)

for test_X in test_X_mstr:
    test_list = []
    test_list.append(test_X)
    att_score_pred = att_model.predict(test_list)
    print("Prediction For ", test_X, ' is ', att_score_pred)
    x1 = test_X[0]
    y1 = test_X[1]

    if att_score_pred == 'Yes':
        plt.plot(x1, y1, marker='+', ms=8, c='r')
    else:
        plt.plot(x1, y1, marker='+', ms=8, c='g')

plt.show()

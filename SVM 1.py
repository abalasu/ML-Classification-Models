import pandas as pd
import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt

att_df = pd.read_csv("D:\PythonData\KNN_Classification.csv")
y = att_df['Attrition']
att_features = ['Age', 'Salary']
X = att_df[att_features]
X_array = X.to_numpy()

plt_x = att_df['Age']
plt_y = att_df['Salary']
plt.scatter(plt_x,plt_y)
i = 0
while i < len(plt_y):
    x1 = plt_x[i]
    y1 = plt_y[i]
    print(y[i])
    if y[i] == 'No':
        plt.plot(x1, y1, marker='d', ms=8, c='b')
    else:
        plt.plot(x1, y1, marker='*', ms=8, c='y')
    i += 1
plt.title('Attrition Prediction')
plt.xlabel('Age')
plt.ylabel('Salary in Lakhs')
att_model_lin = SVC(C=10, kernel="linear")
att_model_lin.fit(X.values, y.values)

test_X_mstr = [(37,25),(21,12),(45,30),(53,100)]

for test_X in test_X_mstr:
    test_list = []
    test_list.append(test_X)
    att_score_pred = att_model_lin.predict(test_list)
    print("Prediction For Poly ", test_X, ' is ', att_score_pred)
    x1 = test_X[0]
    y1 = test_X[1]
    if att_score_pred == 'Yes':
        plt.plot(x1, y1, marker='d', ms=10, c='r')
    else:
        plt.plot(x1, y1, marker='*', ms=10, c='g')
    w = att_model_lin.coef_[0]           # w consists of 2 elements
    b = att_model_lin.intercept_[0]      # b consists of 1 element
    x_points = np.linspace(0, 60)    # generating x-points from -1 to 1
    y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
# Plotting a red hyperplane
    plt.plot(x_points, y_points, c='r');
    
plt.show()
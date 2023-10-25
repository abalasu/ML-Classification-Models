from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
raw_data = {'Age': [20, 22, 21, 23, 24, 25, 26, 27, 28, 29],
            'Years': [2, 4, 3, 5, 6, 7, 8, 9, 10, 11],
            'Salary': [40.0, 43.0, 42.0, 46.0, 48.0, 47.0, 52.0, 54.0, 55.0, 54.0]}

data = pd.DataFrame(raw_data)
x_features = ['Age', 'Years']
y_label = ['Salary']

X = data[x_features]
y = data[y_label]
test_raw_X = {'Age': [20, 21, 22],
            'Years': [12, 13, 14]}
test_raw_y = {'Salary': [62.0, 55.0, 80.0]}

test_X = pd.DataFrame(test_raw_X)
test_y = pd.DataFrame(test_raw_y)
i=1
while i<=10:
    knn_model = KNeighborsRegressor(n_neighbors=i)
    knn_model.fit(X, y)
    knn_pred = knn_model.predict(test_X)
    print(test_X, knn_pred)
    score = mean_squared_error(test_y, knn_pred)
    print(i, score)
    i += 1
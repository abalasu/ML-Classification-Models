from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

raw_data = {'LDL_Count': [180, 160, 170, 200, 210, 140, 300, 115, 120, 155, 186],
            'Heart_disease': [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]}
heart_df = pd.DataFrame(raw_data)

y = heart_df['Heart_disease']
features = ['LDL_Count']
X = heart_df[features]
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 1, train_size=0.70)
heart_model = LogisticRegression()
heart_model.fit(train_X, train_y)

pred_y = heart_model.predict(test_X)
print(test_X, pred_y)
plt.scatter(test_X, pred_y)
plt.scatter(train_X, train_y, c='red', marker='d')
plt.show()
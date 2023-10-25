import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

stu_df = pd.read_csv("D:\PythonData\student_pass_fail.csv")
stu_df_stat = stu_df.describe()

y = stu_df['Pass_Fail']
stu_features = ['Study_Hrs', 'Staff_Rank']
X = stu_df[stu_features]
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0, train_size=20)

student_model = linear_model.LogisticRegression()
student_model.fit(train_X, train_y)
student_score_pred = student_model.predict(test_X)
print("Confusion Matrix")
cm = confusion_matrix(test_y, student_score_pred)
print(cm)
print("Accuracy Score ", accuracy_score(test_y, student_score_pred))
print("Recall ", recall_score(test_y, student_score_pred))
print("Precision ", precision_score(test_y, student_score_pred))
print("F1 Score ", f1_score(test_y, student_score_pred))
print("Classification Report ")
print(classification_report(test_y, student_score_pred))

input("Press Any Key ")

#actual_data = ['orange','apple', 'apple', 'orange', 'orange', 'apple', 'apple', 'apple', 'apple', 'apple', 'apple', 'orange', 'orange']
#pred_data   = ['orange','orange', 'apple', 'apple', 'orange', 'apple', 'apple', 'apple', 'apple', 'apple', 'orange', 'apple', 'apple']

actual_data = np.array(['Dog','Dog',    'Dog','Not Dog','Dog',    'Not Dog','Dog','Dog','Not Dog','Not Dog'])
pred_data   = np.array(['Dog','Not Dog','Dog','Not Dog','Not Dog','Dog',    'Dog','Dog','Not Dog','Not Dog'])
# TP = 1, 3, 7, 8
# FP = 2, 5
# FN = 6
# TN = 4, 9, 10
print("Confusion Matrix")
cm = confusion_matrix(actual_data, pred_data)
print(cm)

cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Dog', 'Not Dog'])

print("Accuracy Score ", accuracy_score(actual_data, pred_data)) # (TP+TN)/(TP+FP+FN+TN)
print("Recall ", recall_score(actual_data, pred_data, pos_label='Dog')) # (TP)/(TP+FN)
print("Precision ", precision_score(actual_data, pred_data, pos_label='Dog'))
print("F1 Score ", f1_score(actual_data, pred_data, pos_label='Dog'))
print("Classification Report ")
print(classification_report(actual_data, pred_data))
cm_display.plot()
plt.show()

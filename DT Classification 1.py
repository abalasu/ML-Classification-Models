from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, classification_report

stu_df = pd.read_csv("D:\PythonData\student_pass_fail_rf.csv")
stu_df_stat = stu_df.describe()

y = stu_df['Pass_Fail']
stu_features = ['Student_int','Access_to_lib', 'Study_Hrs', 'Staff_Rank']
X = stu_df[stu_features]
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0, train_size=0.75)
print(train_X)
student_model = DecisionTreeClassifier(random_state=1,criterion='entropy')
student_model.fit(train_X, train_y)
student_score_pred = student_model.predict(test_X)
plt.figure(figsize=(18,18))
tree.plot_tree(student_model, fontsize=9)
plt.show()
print("Confusion Matrix")
cm = confusion_matrix(test_y, student_score_pred)
print(cm)
print("Accuracy Score ", accuracy_score(test_y, student_score_pred))
print("Recall ", recall_score(test_y, student_score_pred))
print("Precision ", precision_score(test_y, student_score_pred))
print("F1 Score ", f1_score(test_y, student_score_pred))
print("Classification Report ")
print(classification_report(test_y, student_score_pred))

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, classification_report

loan_df = pd.read_csv("D:\PythonData\Gini_Entropy.csv")
y = loan_df['Loan_Status']
loan_features = ['Cust_Id','Loan_Amt']
X = loan_df[loan_features]
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state = 0, train_size=0.75)
print(train_X)
loan_model = DecisionTreeClassifier(random_state=1,criterion='gini')
loan_model.fit(X, y)
print(test_X)
loan_score_pred = loan_model.predict(test_X)
plt.figure(figsize=(18,18))
tree.plot_tree(loan_model, fontsize=9)
plt.show()
print("Confusion Matrix")
cm = confusion_matrix(test_y, loan_score_pred)
print(cm)
print("Accuracy Score ", accuracy_score(test_y, loan_score_pred))
print("Recall ", recall_score(test_y, loan_score_pred))
print("Precision ", precision_score(test_y, loan_score_pred))
print("F1 Score ", f1_score(test_y, loan_score_pred))
print("Classification Report ")
print(classification_report(test_y, loan_score_pred))
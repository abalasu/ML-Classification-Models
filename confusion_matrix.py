from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
import pandas as pd
from seaborn import heatmap

actual = ['Dog', 'Dog', 'Dog', 'Cat', 'Dog', 'Cat', 'Dog', 'Dog', 'Cat', 'Cat']
predicted = ['Dog', 'Cat', 'Dog', 'Cat', 'Cat', 'Dog', 'Dog', 'Dog', 'Cat', 'Cat']
actual_df = pd.DataFrame(actual)
predicted_df = pd.DataFrame(predicted)
actual_df.replace(to_replace='Dog',value=0,inplace=True)
actual_df.replace(to_replace='Cat',value=1,inplace=True)
predicted_df.replace(to_replace='Dog',value=0,inplace=True)
predicted_df.replace(to_replace='Cat',value=1,inplace=True)
cm = confusion_matrix(actual_df, predicted_df, labels=[0,1])
print("Classification Report ", classification_report(actual, predicted))
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Dog', 'Cat'])
cm_display.plot()
plt.show()

a = input("Press enter to continue... ")
y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "bird"] 
cm = confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
cm_df = pd.DataFrame(cm, columns=["ant", "bird", "cat"], index=["ant", "bird", "cat"])

plt.figure(figsize=(5,4))
heatmap(cm_df, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

"""
print('Accuracy_Score ', accuracy_score(actual_df, predicted_df))
print('Recall_Score ', recall_score(actual_df, predicted_df))
print('Precision_Score ', precision_score(actual_df, predicted_df))
print('F1_Score ', f1_score(actual_df, predicted_df, labels=['Dog','Cat']))
"""
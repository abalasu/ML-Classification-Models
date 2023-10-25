from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

digits_data = pd.read_csv('d:/pythondata/training_num.csv')
features = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36',
            'C37','C38','C39','C40','C41','C42','C43','C44','C45','C46','C47','C48','C49','C50','C51','C52','C53','C54','C55','C56','C57','C58','C59','C60','C61','C62','C63','C64']
m, n = digits_data.shape
X = digits_data[features]
y = [1,3,0,1,3,1,5,6,8]
i = 1
j = 0
while j < m:
    record = X.loc[j]
    my_array = record.to_numpy()
    my_resized_array = my_array.reshape(8,8)
    plt.subplot(3,3,i)
    i += 1
    j += 1
    plt.imshow(my_resized_array, cmap=plt.cm.gray)
plt.show()

test_data = pd.read_csv('d:/pythondata/Numbers.csv')
test_X = test_data[features]

m, n = test_X.shape
i = 1
j = 0
while j < m:
    record = test_X.loc[j]
    my_array = record.to_numpy()
    my_resized_array = my_array.reshape(8,8)
    plt.subplot(2, 5, i)
    i += 1
    j += 1
    plt.imshow(my_resized_array, cmap=plt.cm.gray)


act_results = np.array([0,0,0,0,1,1,1,1,3,3])
print("Actual images are        ", act_results)

digit_classifier = KNeighborsClassifier(n_neighbors=3, weights='uniform', metric='euclidean')
digit_classifier.fit(X,y)
KNN_Results = digit_classifier.predict(test_X)
print("The Digit's from KNN are ", KNN_Results)

digit_classifier = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_split=2, max_features=None, random_state=1)
digit_classifier.fit(X,y)
DT_Results = digit_classifier.predict(test_X)
print("The Digit's from DT are  ", DT_Results)

digit_classifier = SVC()
digit_classifier.fit(X,y)
SVM_Results = digit_classifier.predict(test_X)
print("The Digit's from SVM are ", SVM_Results)

plt.show()
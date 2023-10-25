from sklearn.linear_model import LogisticRegression
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

digits_data = pd.read_csv('d:/pythondata/Numbers.csv')
features = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35','C36',
            'C37','C38','C39','C40','C41','C42','C43','C44','C45','C46','C47','C48','C49','C50','C51','C52','C53','C54','C55','C56','C57','C58','C59','C60','C61','C62','C63','C64']
label = ['Result']
X = digits_data[features]
y = digits_data[label]
i = 1
j = 0
while j < 8:
    record = X.loc[j]
    my_array = record.to_numpy()
    my_resized_array = my_array.reshape(8,8)
    plt.subplot(2,4,i)
    i += 1
    j += 1
    plt.imshow(my_resized_array, cmap=plt.cm.gray)
plt.show()

digit_classifier = LogisticRegression()
digit_classifier.fit(X,y.values.ravel())
test_data = pd.read_csv('d:/pythondata/testnum0.csv')
test_x = test_data[features]
i = 1
j = 0
while j < 4:
    record = test_x.loc[j]
    my_array = record.to_numpy()
    my_resized_array = my_array.reshape(8,8)
    plt.subplot(2,2,i)
    i += 1
    j += 1
    plt.imshow(my_resized_array, cmap=plt.cm.gray)
print("The Digit's are ", digit_classifier.predict(test_x))
plt.show()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

data = { 
    "age":[21,19,30,27,29,42,35,37,41,26],
    "salary":[8,6,16,14,17,27,25,25,28,18],
    "resignation":['y','y','n','n','n','y','n','y','n','n']
    }
emp_df = pd.DataFrame(data)
y = emp_df['resignation']
indep_features = ['age','salary']
X = emp_df[indep_features]
class AgeIncrementer(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.key+1]

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
# pipe = Pipeline([('transformation', MinMaxScaler(copy=True))], verbose=True)
pipe = Pipeline([('transformation', MinMaxScaler(copy=False,clip=True)), ('model', KNeighborsClassifier(n_neighbors=3, weights='distance'))], verbose=True)
# The pipeline can be used as any other estimator
# and avoids leaking the test set into the train set
print("Before")
pipe.fit(X_train, y_train)
print("After")
print(X_train)
print(pipe.score(X_test, y_test))

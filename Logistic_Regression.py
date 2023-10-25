""" 
Logistic regression aims to solve classification problems. It does this by predicting categorical outcomes, unlike linear regression 
that predicts a continuous outcome. In the simplest case there are two outcomes, which is called binomial, an example of which is 
predicting if a tumor is malignant or benign. Other cases have more than two outcomes to classify, in this case it is called 
multinomial. A common example for multinomial logistic regression would be predicting the class of an iris flower between 3 
different species. 
"""
from sklearn import linear_model
import numpy as np
# when the number of elements is unknown you give -1 in the reshape parameter for numpy to consider the required value automatically
X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
# print(X)
# X = np.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(3,-1)
# print(X)

y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
logr = linear_model.LogisticRegression()
logr.fit(X,y)

predicted = logr.predict(np.array([6.46]).reshape(-1,1))
print(predicted)

# The above array is set of Tumor sizes and the model predicts if it is really cancer. The coef tells the relationship between X and y.
# The odds here tells that for an increase of the tumor by 1mm the odds of it being Cancer increases 4 times
log_odds = logr.coef_
# The exp function converts the lograthimic number to a regular number
odds = np.exp(log_odds)
print(odds)

# Finding the actual probability based on our model
def logit2prob(logr, X):
  log_odds = logr.coef_ * X + logr.intercept_
  odds = np.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)
# The probability of a tumor of size 3.78cm is 61%
print(logit2prob(logr, X), y)

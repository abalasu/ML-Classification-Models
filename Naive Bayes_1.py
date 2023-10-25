from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
import nltk.data
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
raw_data = {'type': ['Good','Spam','Good','Good','Spam','Good','Spam','Good','Good','Spam'],
            'mail':['Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Friend Friend Friend Friend Friend Friend Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch',
                          'Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money',
                          'Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Friend Friend Friend Friend Friend Friend Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch',
                          'Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Friend Friend Friend Friend Friend Friend Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Money Money Money Money Money',
                          'Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear',
                          'Friend Friend Friend Friend Friend',
                          'Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend',
                          'Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Family Family Family Family Family Family Family Family Family Family',
                          'Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Family Family Family Family Family Family Family Family Family Family Money Money Money Money Money',
                          'Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money']
            }

message = pd.DataFrame(raw_data)
string = ""
for msg in message['mail']:
    string = string + msg + ' '
tokens = word_tokenize(string.lower())
voc = []
[voc.append(word) for word in tokens if word not in voc]
vectorizer = CountVectorizer(vocabulary=voc)
X = vectorizer.fit_transform(message['mail'])
print('X', X)
X_dense = X.todense()
print(" Term Frequency ")
print(voc)
voc.sort()
print(voc)
print('X_Dense')
print(X_dense)
pipe = Pipeline([('count', CountVectorizer(vocabulary=voc)),
                 ('tfid', TfidfTransformer(sublinear_tf=False, norm='l1')), ('nbmodel', MultinomialNB(alpha=1, force_alpha=True))])
nbmodel = pipe.fit(message['mail'],message['type'])
test_data = ['Dear Friend Lunch', 'Dear Friend Money', 'Dear family money money money']
a = nbmodel.predict(test_data)
i = 0
while i<len(test_data):
    print("'",test_data[i],"'",'is a ', a[i], 'e-mail')
    i = i+1

"""
tfid = TfidfTransformer(smooth_idf=False)
tfid_trans = tfid.fit_transform(X)
tfid_dense = tfid_trans.todense()
print(tfid_dense)
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pandas as pd
from sklearn.pipeline import Pipeline
raw_data = {'mail':['Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Friend Friend Friend Friend Friend Friend Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch',
                    'Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money',
                    'Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Friend Friend Friend Friend Friend Friend Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch',
                    'Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Friend Friend Friend Friend Friend Friend Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Lunch Money Money Money Money Money',
                    'Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear Dear',
                    'Friend Friend Friend Friend Friend',
                    'Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend',
                    'Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Family Family Family Family Family Family Family Family Family Family',
                    'Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Friend Family Family Family Family Family Family Family Family Family Family Money Money Money Money Money',
                    'Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money Money'],
            'type': ['Good','Spam','Good','Good','Spam','Good','Spam','Good','Good','Spam']
            }

message = pd.DataFrame(raw_data)
m,n = message.shape
i = 0
while i < m:
    msg = message['mail'][i]
    message['mail'][i] = msg.lower()
    i += 1
#message['mail'] = message['mail'].to_string().lower()
#print(message['mail'])
vectorizer = CountVectorizer()
# fit - method creates the vocabolary
# transform - method creates the document term matrix
X = vectorizer.fit_transform(message['mail'])
print('X')
print(X)
voc = vectorizer.vocabulary_
voc_list = [keys for keys in voc.keys()]
X_dense = X.todense()
voc_list.sort()
print(" Count Matrix ")
print('X_Dense')
print(voc_list)
print(X_dense)

# Normalizes the data
tfidtransform = TfidfTransformer(sublinear_tf=False, norm='l1', use_idf=False)
Y = tfidtransform.fit_transform(X) 
print(" Normalized Term Frequency ")
print('Y')
print(Y)

# Actual Model
nbmodel = MultinomialNB(alpha=1, force_alpha=True)
nbmodel.fit(Y, message['type'])

test_data = ['Dear Friend Lunch', 'Dear Friend Money', 'Dear family money money money']
test_data = [tdata.lower() for tdata in test_data]
X = vectorizer.fit_transform(test_data)
Y = tfidtransform.fit_transform(X)
a = nbmodel.predict(Y)
i = 0
while i<len(test_data):
    print("'",test_data[i],"'",'is a ', a[i], 'e-mail')
    i = i+1

"""
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
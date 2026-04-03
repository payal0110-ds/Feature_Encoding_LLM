from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re

messages=pd.read_csv("SMSSpamCollection",sep="\t",names=["label","message"])

corpus=[]

for i in range(len(messages)):
    review=re.sub("[^a-zA-Z]"," ",messages["message"][i])
    review=review.lower()
    review=review.split()
    review=[WordNetLemmatizer().lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=" ".join(review)
    corpus.append(review)

# print(corpus)
print(len(corpus))

cv=CountVectorizer(max_features=100,binary=True)
"""
Using 'CountVectorizer' also, we can lower case and use stop words directly.
"""
x=cv.fit_transform(corpus).toarray()
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))

print(x)
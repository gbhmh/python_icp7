from sklearn.datasets import fetch_20newsgroups
import  pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
categories = ['alt.atheism', 'talk.religion.misc',
             'comp.graphics', 'rec.motorcycles', 'sci.space']

twenty_train = fetch_20newsgroups(subset='train', shuffle=True, categories=categories)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = KNeighborsClassifier()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True, categories=categories)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)


tfidf_Vect1 = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf1 = tfidf_Vect1.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf1 = KNeighborsClassifier()
clf1.fit(X_train_tfidf1, twenty_train.target)
X_test_tfidf1 = tfidf_Vect1.transform(twenty_test.data)
predicted1 = clf1.predict(X_test_tfidf1)

tfidf_Vect2 = TfidfVectorizer(stop_words='english')
X_train_tfidf2 = tfidf_Vect2.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf2 = KNeighborsClassifier()
clf2.fit(X_train_tfidf2, twenty_train.target)
X_test_tfidf2 = tfidf_Vect2.transform(twenty_test.data)
predicted2 = clf2.predict(X_test_tfidf2)

score = metrics.accuracy_score(twenty_test.target, predicted)
print('knn score is ' + str(score))

score1 = metrics.accuracy_score(twenty_test.target, predicted1)
print('using npgram score is ' + str(score1))

score2 = metrics.accuracy_score(twenty_test.target, predicted2)
print('using stop words score is ' + str(score2))
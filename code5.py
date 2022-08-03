# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 14:39:23 2022

@author: vikas
"""

from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)
clf.predict([[2., 2.], [-1., -2.]])







import pandas as pd

df = pd.read_csv('pima-indians-diabetes.csv')

df.columns

X= df.drop('Result', axis=1).values

Y = df['Result'].values

from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.1)


from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(trainX, trainY)
pred = clf.predict(testX)

from sklearn.metrics import classification_report

print(classification_report(testY, pred))


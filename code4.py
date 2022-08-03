# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 14:33:10 2022

@author: vikas
"""


import pandas as pd

df = pd.read_csv('pima-indians-diabetes.csv')

df.columns

X= df.drop('Result', axis=1).values

Y = df['Result'].values

from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.1)



from perceptrons import Perceptron

p = Perceptron(weights=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3,0.3, 0.3, 0.3],
               learning_rate=0.8)

for sample, label in zip(trainX, trainY):
    p.adjust(label,
             sample)

evaluation = p.evaluate(testX, testY)
print(evaluation)




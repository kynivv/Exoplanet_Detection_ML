import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score as asc

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Data Import
data_train = pd.read_csv('exoTrain.csv')
data_test = pd.read_csv('exoTest.csv')


# Data Preparation
X_train = data_train.drop('LABEL', axis= 1)
Y_train = data_train['LABEL']

X_test = data_test.drop('LABEL', axis= 1)
Y_test = data_test['LABEL']


# Model Training & Accuracy Results
models = [DecisionTreeClassifier(random_state= 42),
          RandomForestClassifier(random_state= 42)]

for m in models:
    print(m)
    
    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy is : {asc(Y_train, pred_train, normalize= True)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy is : {asc(Y_test, pred_test, normalize= True)}\n')
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 02:47:43 2022

@author: dhiren
"""

"""+-----------+----------------+
| Estimator | Model Accuracy |
+-----------+----------------+
|     10    |    77.0492     |
|     20    |    81.9672     |
|     30    |    85.2459     |
|     40    |    83.6066     |
|     50    |    81.9672     |
|     60    |    83.6066     |
|     70    |    78.6885     |
|     80    |    83.6066     |
|     90    |    81.9672     |
|    100    |    81.9672     |
+-----------+----------------+"""

import pandas as pd 
import numpy as np 
heart_disease = pd.read_csv("heart.csv") 
X = heart_disease.drop(['target'] , axis=1)  
Y = heart_disease['target'] 
from sklearn.ensemble import RandomForestClassifier  
clf = RandomForestClassifier(n_estimators=10) 
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)                                               
clf.fit(X_train, Y_train) 

np.random.seed(42)

max_value=float()
maxi=0;

from prettytable import PrettyTable
myTable = PrettyTable(["Estimator", "Model Accuracy"])

for i in range(10,101,10):
    print("Trying model with",{i} ,"estimators...")
    clf = RandomForestClassifier(n_estimators=i).fit(X_train,Y_train)
    check=clf.score(X_test,Y_test)*100
    check=round(check,4)
    print("Model accuracy on test set= ",check)
    if (check>max_value):
        max_value=check
        maxi=i
    myTable.add_row([i,check])

print(myTable)

print('The max accuracy value is',max_value,' which is given by estimator value of',maxi)

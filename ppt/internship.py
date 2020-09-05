import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Bankloan.csv')
X = dataset.iloc[:, [1,2,3,5,6,7,8,10,11,12,13]].values
Y = dataset.iloc[:, -5].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=0)

from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression()
training_set = regressor.fit(x_train,y_train)
y_pred = .predict(X_test)

from sklearn.metrics import confusion_matrix
con_metrics = confusion_matrix(y_test, y_train)
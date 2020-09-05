import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

impor=pd.read_csv('Salary_Data.csv')
x=impor.iloc[:,:-1]
y=impor.iloc[:,1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#fit and train the data now
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)#here I have trained my data


#now, predicting means testing 
y_pred=regressor.predict(x_test)
plt.title("training set vs test set")
plt.xlabel("year")
plt.ylabel("Salary")
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.show()



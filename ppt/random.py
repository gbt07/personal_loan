import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values   #Here, I learnt one thing that if u are using any column that you have to use proper slicing else it will give u error
y=dataset.iloc[:,2:3].values

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(x,y)

xgrid=np.arange(min(x),max(x),0.01)
xgrid=xgrid.reshape((len(xgrid),1))
"""
plt.scatter(x,y,color='red')
plt.plot(xgrid,regressor.predict(xgrid),color='black')
plt.show()
"""
plt.scatter(x,y,color='red')
plt.plot(xgrid,regressor.predict(xgrid),color='black')
plt.show()
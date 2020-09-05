import matplotlib.pyplot  as plt
import pandas as pd
import numpy as np


dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values   #Here, I learnt one thing that if u are using any column that you have to use proper slicing else it will give u error
y=dataset.iloc[:,2].values



from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x,y)


from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=6)
polyx=poly.fit_transform(x)

#poly.fit(polyx,y)
polyreg=LinearRegression()
polyreg.fit(polyx, y)

#plt.scatter(x,y,color='blue')
#plt.plot(x,reg.predict(x),color='red')
#plt.show()

plt.scatter(x,y,color='blue')
plt.plot(x,polyreg.predict(polyx),color='red')
plt.show()




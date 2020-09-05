import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values   #Here, I learnt one thing that if u are using any column that you have to use proper slicing else it will give u error
y=dataset.iloc[:,2:3].values



from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalary=StandardScaler()
x=scalar.fit_transform(x)

#y=column_or_1d(y,warn=True)
y=scalary.fit_transform(y)


from sklearn.svm import SVR
reg=SVR()
reg.fit(x,y)


plt.scatter(x,y,color='red')
plt.plot(x,reg.predict(x),color='black')
plt.show()
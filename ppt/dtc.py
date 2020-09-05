

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


dataset=pd.read_csv("logisticdata.csv")
x=dataset.iloc[:,[2,3]].values   #Here, I learnt one thing that if u are using any column that you have to use proper slicing else it will give u error
y=dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test,=train_test_split(x,y,test_size=0.25,random_state=0)


from sklearn.tree import DecisionTreeClassifier
regressor=DecisionTreeClassifier(criterion='entropy',random_state=0)
regressor.fit(x_train,y_train)

from matplotlib.colors import ListedColormap
x_set,y_set=x_test,y_test
x1,x2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,regressor.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.5,cmap=ListedColormap(('red','green')))
gbt=np.unique(y_set)
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
    

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("50_Startups.csv")

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
encode_x=ColumnTransformer([("State",OneHotEncoder(),[3])],remainder='passthrough')
x=encode_x.fit_transform(x)
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test,=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)


y_pred=reg.predict(x_test)

import statsmodels.api as sm
x=np.append(arr=np.ones((50,1)).astype(int), values=x, axis=1)
X_opt = np.array(x[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = np.array(x[:, [0, 2, 3,4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = np.array(x[:, [0,  2, 3]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = np.array(x[:, [0,  3]], dtype=float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

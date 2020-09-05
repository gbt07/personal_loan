#impoerting data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

impor=pd.read_csv('Bank.csv')
"""
x=impor.iloc[:,:-1]
y=impor.iloc[:,3]
#my name is a gaurav tiwari

# filling missing values
from sklearn.impute import SimpleImputer
misvalue=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
misvalue=misvalue.fit(x.iloc[:,1:3])
#x.iloc[:,1:3]=misvalue.transform(x.iloc[:,1:3])
x.iloc[:,1:3]=misvalue.fit_transform(x.iloc[:,1:3])


#categorical data
from sklearn.preprocessing import LabelEncoder
encodex=LabelEncoder()
x.iloc[:,0]=encodex.fit_transform(x.iloc[:,0])

#onehotencoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
encode_x=ColumnTransformer([('Country', OneHotEncoder(), [0])],remainder='passthrough'    )   
x=encode_x.fit_transform(x)

encode_y=LabelEncoder()
y=encode_y.fit_transform(y)


##training and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
train=StandardScaler()
x_train=train.fit_transform(x_train)
x_test=train.transform(x_test)

"""

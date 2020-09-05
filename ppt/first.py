# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:29:28 2020

@author: TOSHIBA
"""


import pandas as pd
import numpy as np
#d=pd.Series([1,2,3,4,5,6])

#print(d)
#df=pd.date_range("20200510", periods=10)
#print(df)
#tab=pd.DataFrame(np.random.randn(10,5),index=df,columns=['A','B','C','D','e'])
#print(tab)
#df=pd.DataFrame({'A':d,'B':d,'c':d})
#print(df)
read=pd.read_csv('Data.csv')
x=read.iloc[:,:-1]
#print(x)
y=read.iloc[:,3]
#print(y)
#check about missing data
from sklearn.impute import SimpleImputer
mis1=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0) 
mis1=mis1.fit(x.iloc[:,1:3])
x.iloc[:,1:3]=mis1.transform(x.iloc[:,1:3])
#print(x.iloc[:,1:3])
#print(x)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

encode = LabelEncoder()
x.iloc[:,0]=encode.fit_transform(x.iloc[:,0]) 
ct = ColumnTransformer([("Country", OneHotEncoder(), [1])])  
# remainder = 'passthrough')
#ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
#encode=OneHotEncoder(categorical_features=[0])
x= ct.fit_transform(x).toarray()
#x=x[:,:1]
encode_y=LabelEncoder()
y=encode_y.fit_transform(y)
#x=encode.fit_transform(x).toarray()
from sklearn.model_selection import train_test_split
a,b,c,d=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
gbt=StandardScaler()
a=gbt.fit_transform(a)
b=gbt.transform(b)



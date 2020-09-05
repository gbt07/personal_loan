# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 22:23:45 2020

@author: TOSHIBA
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




from sklearn import model_selection


#here, we are using pandas library to read excel.

dataset = pd.read_excel('Bank_Personal_Loan_Modelling.xlsx','Data')
dataset.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]
columns = dataset.head()


data_info=dataset.info()
#here we will use lambda function to show that there is any null set or not
null_info=dataset.apply(lambda x : sum(x.isnull()))

#now we will correlate every column with each other for that we will import library seaborn
import seaborn as sns
#correlation_info=sns.pairplot(dataset.iloc[:,1:])


#here we have seen some negative values in experience column, thus we will clean some data
count_neg=dataset[dataset['Experience'] < 0]['Experience'].count() 




Exp = dataset.loc[dataset['Experience'] >0]
neg_experience = dataset.Experience < 0
column_name = 'Experience'
mylist = dataset.loc[neg_experience]['ID'].tolist()


#here we will count total number of negative experience
count_negativeExperince=neg_experience.value_counts()


#here we will try to replace negative value
for id in mylist:
    age = dataset.loc[np.where(dataset['ID']==id)]["Age"].tolist()[0]
    education = dataset.loc[np.where(dataset['ID']==id)]["Education"].tolist()[0]
    filtered = Exp[(Exp.Age == age) & (Exp.Education == education)]
    exp = filtered['Experience'].mean()
    dataset.loc[dataset.loc[np.where(dataset['ID']==id)].index, 'Experience'] = exp
    
#here we will do all the statistical calculations
all_calculations=dataset.describe().transpose()

#here we will do some ploting between different variables
sns.boxplot(x='Education',y='Income',hue='PersonalLoan',data=dataset)

sns.boxplot(x="Education", y='Mortgage', hue="PersonalLoan", data=dataset,color='yellow')
sns.countplot(x="SecuritiesAccount", data=dataset,hue="PersonalLoan")
sns.countplot(x='Family',data=dataset,hue='PersonalLoan',palette='Set1')
sns.countplot(x='CDAccount',data=dataset,hue='PersonalLoan')

# Customers who does not have CD account, they don't have loan as well. But all customers who has CD account has loan as well

sns.distplot( dataset[dataset.PersonalLoan == 0]['CCAvg'], color = 'r')
sns.distplot( dataset[dataset.PersonalLoan == 1]['CCAvg'], color = 'g')

#higher credit card speding shows a higher probaility of taking personal loan and lower as lower probaility.
print('Credit card spending of Non-Loan customers: ',dataset[dataset.PersonalLoan == 0]['CCAvg'].median()*1000)


print('Credit card spending of Loan customers    : ', dataset[dataset.PersonalLoan == 1]['CCAvg'].median()*1000)



fig, figure = plt.subplots()
colors = {1:'red',2:'yellow',3:'green'}
figure.scatter(dataset['Experience'],dataset['Age'],c=dataset['Education'].apply(lambda x:colors[x]))
plt.xlabel('Experience')
plt.ylabel('Age')


#The above plot by this code show with experience and age have a positive correlation. 
#As experience increase age also increases. and the colors show the education level. There is gap in the mid forties of age and also more people in the UG level



#By the observation, we have seen Income and CCAvg is moderately correlated and Age and Experience is highly correlated
sns.boxplot(x=dataset.Family,y=dataset.Income,hue=dataset.PersonalLoan)






#now splitting the dataset,train test split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dataset.drop(['ID','Experience'], axis=1), test_size=0.3 , random_state=100)
train_labels = train_set.pop('PersonalLoan')
test_labels = test_set.pop('PersonalLoan')

#applying Decision tree model algorithm
#importing library


from sklearn.tree import DecisionTreeClassifier
data_model=DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
data_model.fit(train_set, train_labels)


#now we will check score
data_model.score(test_set , test_labels)



#importing naivw bayes library ad using this algorithm
from sklearn.naive_bayes import GaussianNB
naive_model = GaussianNB()
naive_model.fit(train_set, train_labels)

predict = naive_model.predict(test_set)

#again checking score
naive_model.score(test_set,test_labels)


#importing random forest algorithm and aplying on dataset
from sklearn.ensemble import RandomForestClassifier
randomforest_model = RandomForestClassifier(max_depth=2, random_state=0)
randomforest_model.fit(train_set, train_labels)

predicted_random=randomforest_model.predict(test_set)
randomforest_model.score(test_set,test_labels)


#importing KNN ad doing same

#here in KNN non continous values are not allowed, thus here we need some sonversion
train_set_indep = dataset.drop(['Experience' ,'ID'] , axis = 1).drop(labels= "PersonalLoan" , axis = 1)
train_set_dep = dataset["PersonalLoan"]
X = np.array(train_set_indep)
Y = np.array(train_set_dep)
X_Train = X[ :3500, :]
X_Test = X[3501: , :]
Y_Train = Y[:3500, ]
Y_Test = Y[3501:, ]
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 21 , weights = 'uniform', metric='euclidean')
knn.fit(X_Train, Y_Train)    
predicted = knn.predict(X_Test)
from sklearn.metrics import accuracy_score
acc = accuracy_score(Y_Test, predicted)


#import logistic regression and aplly on dataset
from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression()
regressor.fit(train_set,train_labels)



X=dataset.drop(['PersonalLoan','Experience','ID'],axis=1)
y=dataset.pop('PersonalLoan')
models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('LR', LogisticRegression()))
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=12345)
	cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


#at last we have done a comparision between the algorithm and after that we found that random forest has maximum accuracy.


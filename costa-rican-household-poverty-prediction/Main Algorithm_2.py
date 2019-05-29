# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 16:25:14 2019

@author: MMOHTASHIM
"""

import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import svm,preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
import pickle
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
df=pd.read_csv("train.csv")
df_2=pd.read_csv("test.csv")
df.drop(["v2a1","Id","idhogar",'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned'],1,inplace=True)###These values are being dropped due to annomalies or nan or inf present
df_2.drop(["v2a1","Id","idhogar",'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned'],1,inplace=True)###These values are being dropped due to annomalies or nan or inf present
list_types=df.dtypes.tolist()
list_types_2=df_2.dtypes.tolist()
print("Running")
def data_type(list_types):#This function returns a list containing the index for column which are not numeric
    index=-1
    indexes=[]
    for a in list_types:
            index+=1
            if a!=np.float64 and a!=np.int64:
                indexes.append(index)
    return indexes
index=data_type(list_types)


########Convert non-numeric data to binary 
def Binary_converter(df,index): 
    for i in index:
        for ii in range(len(df)):
            point=df.iloc[ii,i]
            if point == "no":
                df[df.columns[i]].replace(point,0,inplace=True)
                print("replaced")
            if point == "yes":
                df[df.columns[i]].replace(point,1,inplace=True)
                print("replaced")
    return df
df_2=Binary_converter(df,index)
df_2=df.apply(pd.to_numeric)
############################################
def Check_NAN(): ##Function to know which column contain NAN to check annomalies and prevent data loss,no need to call
    columns=df.columns.tolist()
    pop=[]
    for i in columns:
            if df[i].isnull().values.any():
                pop.append(i)
def get_X_sample(X,N):##after multiple tries ,reached the conclusion that data is too large so going to take random subsets and test the data:
    random_indices = np.arange(0, X.shape[0])    # array of all indices
    np.random.shuffle(random_indices)                         # shuffle the array
    X_sample=X[random_indices[:N]]  # get N samples without replacement
    return X_sample
##NOT RELATED
#def Pickle_Learner(X):
#    X_sample=X[:11000]
#    clf=MeanShift(n_jobs=-1)
#    clf.fit(X_sample)   
#    #######saving the classifier
#    file_Name = "MeansShift"           
#    fileObject = open(file_Name,'wb') 
#    pickle.dump(clf,fileObject)   
#    fileObject.close()
#    print(set(clf.labels_))
##Pickle_Learner(X)
#def Load_Pickle_MeanShift():
#    fileObject = open("MeansShift",'rb') 
#    clf=pickle.load(fileObject)
#    return clf
##Not related
##clf=Load_Pickle_MeanShift()
##df_2["Poverty Level"]=[np.NaN for i in range(len(df_2))]
##labels=clf.labels_
##print(len(labels))
def remodify_levels(labels,df):###Updated Dataframe
    a=-1
    for i in range(len(labels)):
        a+=1
        df.iloc[i,-1]=labels[a]
    df.to_csv("ReModified Test_3.csv")


X=np.array(df.drop("Target",1)) #repeated variable
#
#
X=preprocessing.scale(X)
y=np.array(df["Target"])

#Best estimator found by grid search:
#SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
#  max_iter=-1, probability=False, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)
def SVM(X,y):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2)
    clf=svm.SVC(C=1, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)##Found through grid search
    clf.fit(X_train,y_train)
    accuracy=clf.score(X_test,y_test)
    file_Name = "SVM_2"           
    fileObject = open(file_Name,'wb') 
    pickle.dump(clf,fileObject)   
    fileObject.close()
    return accuracy
def Load_Pickle_SVM():
    fileObject = open("SVM_2",'rb') 
    clf=pickle.load(fileObject)
    return clf
## Classified called and pickled
print(SVM(X,y))
clf=Load_Pickle_SVM()
X_test=np.array(df_2.drop("Target",1))
predictions=clf.predict(X_test)
print(set(predictions))
remodify_levels(predictions,df_2)

def remodify_levels_2(prediction,df):
    a=-1
    for i in range(len(prediction)):
        a+=1
        df.iloc[11000+i,-1]=prediction[a]
    df.to_csv("ReModified Test_2.csv")
#just to check statistics  
df_2=pd.read_csv("ReModified Test_3.csv")   
sample_statistics_0=df[(df["Target"]==1)].describe()
sample_statistics_1=df[(df["Target"]==2)].describe()
sample_statistics_2=df[(df["Target"]==3)].describe()
sample_statistics_3=df[(df["Target"]==4)].describe()
sample_statistics_4=df_2[(df_2["Target"]==4)].describe()

##caleld randomizedSearchCV to improve accuracy
print("Fitting the classifier to the training set")
param_grid = {'C': [0.0001,0.001,0.01, 0.1, 1, 10, 100], 'kernel': ['rbf', 'sigmoid'],"gamma": [0.1,0.2,0.3,0.4, 0.5,0,6,0.7,0.8,0.9,1.0]}
clf = RandomizedSearchCV(svm.SVC(class_weight='balanced'), param_grid,n_jobs=-1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2)
clf = clf.fit(X_train, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)


########This is to visualize the data at random and check for accuracy
column_names=df.columns.values.tolist()
list_names=random.sample(column_names,1)
ax=df_2[list_names].rolling(5).mean().plot(c="g")
df.loc[(df["Target"]==1),list_names].rolling(5).mean().plot(ax=ax,c="r")
plt.show()



    
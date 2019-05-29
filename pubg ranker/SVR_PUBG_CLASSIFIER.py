# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 20:09:03 2019

@author: MMOHTASHIM
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm,preprocessing
import pickle
from sklearn.model_selection import GridSearchCV

def clean_training_data():
    df=pd.read_csv("train_V2.csv")
    df=df.iloc[:10000,:]
    #df.to_csv("taken_data.csv")
    matchType_numeric=[]
#    df=pd.read_csv("taken_data.csv")
    print(df.dtypes.tolist())
    df=df.drop(["matchId","Id","groupId"],1)
    
    matchtype_list=df["matchType"].tolist()
    match_types=list(set(matchtype_list))
    for data in df["matchType"]:
        for match_type in match_types:
            if data==match_type:
                index=match_types.index(match_type)
                matchType_numeric.append(index)
    df.drop(["matchType"],1)
    df["matchType"]=np.array(matchType_numeric)
    df.fillna(value=9999999,inplace=True)
    df.to_csv("cleaned_data.csv")

def machine_learner_data():
    print("Getting Data")
    df=pd.read_csv("cleaned_data.csv")
    df=df.iloc[:10000,:]
    X=np.array(df.drop(["winPlacePerc"],1))
    X=preprocessing.scale(X)
    y=np.array(df["winPlacePerc"])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    print("Training Phase")
    clf=svm.SVR()
    parameters = {
    "kernel": ["rbf"],
    "C": [1,10,10,100,1000],
    "gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }
    clf = GridSearchCV(clf, parameters, cv=5, verbose=2,n_jobs=-1)
    clf.fit(X_train, y_train)
    print("The Accuracy is",clf.score(X_test,y_test))
    with open("pubg_svr.pickle","wb") as f:
        pickle.dump(clf,f)
def clean_testing_data():
    df=pd.read_csv("test_V2.csv")
    matchType_numeric=[]
    df=df.drop(["matchId","Id","groupId"],1)
    matchtype_list=df["matchType"].tolist()
    match_types=list(set(matchtype_list))
    for data in df["matchType"]:
        for match_type in match_types:
            if data==match_type:
                index=match_types.index(match_type)
                matchType_numeric.append(index)
    df.drop(["matchType"],1)
    df["matchType"]=np.array(matchType_numeric)
    df.fillna(value=9999999,inplace=True)
    df.to_csv("cleaned_data_2.csv")
def testing_data():
    pickle_in=open("pubg_svr.pickle","rb")
    clf=pickle.load(pickle_in)
    df=pd.read_csv("cleaned_data_2.csv")
    X=np.array(df)
    
    y = clf.predict(X)
    y=np.array(y)
    df["winPlacePerc"]=y
    df.to_csv("cleaned_data_2.csv")
clean_training_data()
machine_learner_data()
clean_testing_data()
testing_data()
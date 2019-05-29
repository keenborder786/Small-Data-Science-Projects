# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 21:26:29 2019

@author: MMOHTASHIM
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use("ggplot")
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import svm

def read_data():
    df=pd.read_csv("sales_train_v2.csv")
    ####lets get rid of useless things
    df.drop(["date","date_block_num" ,"item_price"],1,inplace=True)
    df=df.dropna()
    #########combing item_cnt_day
    df_2=df.groupby(['shop_id', 'item_id'])["item_cnt_day"].apply(lambda x : x.astype(int).sum())
    print(df_2)
    df_2.to_csv("sales-regression.csv")

####cleaning the data and converting it into dataframe
df=pd.read_csv("sales-regression.csv")
shop_id=np.array(df.iloc[:,0])
item_id=np.array(df.iloc[:,1])
item_cnt_day=np.array(df.iloc[:,2])
dic={'shop_id':shop_id,'item_id':item_id,"item_cnt_day":item_cnt_day}
df=pd.DataFrame(dic,columns=["shop_id","item_id","item_cnt_day"])
######################################################


df=df.dropna()
k=df['item_cnt_day'].unique().tolist()

clf=LinearRegression()

X=np.array(df.drop(["item_cnt_day"],1))
y=np.array(df["item_cnt_day"])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)
print(accuracy)









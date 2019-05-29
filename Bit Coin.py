# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 00:52:59 2019

@author: MMOHTASHIM
"""


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.dates as mdates
import pandas as pd
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
import numpy as np



df = pd.read_csv(r"C:\Users\MMOHTASHIM\Downloads\Bitcoin Historical Data - Investing.com (2).csv")
df.set_index("Date",inplace=True)
df.index = pd.to_datetime(df.index)
for i in df.columns.tolist():
    df['{}'.format(i)] = df['{}'.format(i)].str.replace(',', '')
    df['{}'.format(i)] = df['{}'.format(i)].str.replace('$', '')
    df['{}'.format(i)] = df['{}'.format(i)].str.replace('%', '')
    df['{}'.format(i)] = df['{}'.format(i)].str.replace('M', '')
    df['{}'.format(i)] = df['{}'.format(i)].str.replace('K', '')
    df['{}'.format(i)] = df['{}'.format(i)].astype(float)
df.fillna(0,inplace=True)
df["Change %"]=df["Change %"]/100


df_ohlc=df["Price"].resample("M").ohlc()
df_ohlc.reset_index(inplace=True)
print(df_ohlc.head())
df_ohlc["Date"]=df_ohlc["Date"].map(mdates.date2num)
ax1=plt.subplot(311)
ax2=plt.subplot(312)
ax3=plt.subplot(313)
ax1.set_ylabel("ohlc")
ax2.set_ylabel("Volume Traded")
ax3.set_xlabel("Date")
ax3.set_ylabel("Change %")
#ax2=plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1,sharex=ax1)
ax1.xaxis_date()
candlestick_ohlc(ax1,df_ohlc.values,width=2,colorup='g')
ax2.plot(df["Vol."])
ax3.plot(df["Change %"])
plt.show()
plt.pause(15)
plt.close()



X=np.array(df_ohlc.shift(-7).dropna())
accuracy=[]
accuracy_2=[]
t=[]
t_2=[]
y=np.array(df_ohlc["close"][:len(X)])
for i in range(1000):
    clf=LinearRegression()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
    clf.fit(X_train,y_train)
#    print(clf.score(X_test,y_test))
    a=clf.score(X_test,y_test)
    if a>0:
        accuracy.append(a) 
        t.append(i)
        
    else:
        accuracy_2.append(a) 
        t_2.append(i)
print("The mean accuracy over 10000 trails is :" ,np.mean(np.array(accuracy)))
ax1=plt.subplot(211)
ax2=plt.subplot(212)
ax1.scatter(x=t,y=accuracy)
ax2.set_xlabel("Trials")
ax2.set_ylabel("Negative Accuracy")
ax1.set_ylabel("Positive Accuracy")
ax2.scatter(x=t_2,y=accuracy_2)





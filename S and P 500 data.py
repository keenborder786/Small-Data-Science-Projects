import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
style.use("ggplot")

df=pd.read_csv("S&P 500 Historical Data.csv",index_col=0)
df=df.iloc[::-1]
print(df)
print(df.index)
dates=pd.to_datetime(df.index)
df=df.reset_index(drop=True)
df=df.set_index(dates)
print(df)
forecast_out=math.ceil(len(df)*0.1)
df["label"]=df["Price"].shift(-forecast_out)
df=df.drop(["Vol."],1)
X=np.array(df.drop(["Price"],1))
X=X[:-forecast_out]
X_lately=X[-forecast_out:]
print(len(X))

df=df.dropna(axis=0)


y=np.array(df["Price"])

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
clf=LinearRegression()
clf.fit(X_train,y_train)
accuracy=clf.score(X_test,y_test)


forecast_set=clf.predict(X_lately)

df["Forecast"]=np.nan

last_date=df.iloc[-1].name
unix_time=last_date.timestamp()
one_day=86400
unix_time+=one_day


for i in forecast_set:
    time=datetime.datetime.fromtimestamp(unix_time)
    unix_time+=one_day
    df.loc[time]=[np.nan for _ in range(len(df.columns)-1)] + [i]
print(df)
df["Forecast_moving_mean"]=df["Forecast"].rolling(10).mean()
df["Forecast_moving_corr"]=df["Forecast"].rolling(10).corr()
df["Forecast_moving_std"]=df["Forecast"].rolling(10).std()



plt.plot(df["Forecast_moving_mean"],label="mean")
plt.plot(df["Forecast_moving_corr"],label="correlation")
plt.plot(df["Forecast_moving_std"],label="std")
plt.plot(df["Price"],label="Orginal Price")
plt.plot(df["Forecast"],label="Forecasted Price")
plt.xlabel("Date")
plt.ylabel("Price/Relation")
plt.legend()
plt.show()
    
    





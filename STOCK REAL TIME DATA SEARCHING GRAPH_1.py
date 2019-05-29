import matplotlib.pyplot as plt
from matplotlib import style
import urllib
import json
import urllib.request
import pandas as pd
import time
import scipy.stats as stats
import random
import numpy as np
from statistics import mean,stdev
requestTypeURL = "function=TIME_SERIES_DAILY"
requestTypeURL_2 = "function=TIME_SERIES_INTRADAY"
myKey=""
mainURL = "https://www.alphavantage.co/query?"
style.use('fivethirtyeight')
def dailyData(symbol, requestType=requestTypeURL, apiKey=myKey,datatype="csv"):
    symbolURL = "symbol=" + str(symbol)
    apiURL = "apikey=" + myKey
    datatype="datatype" + datatype
    completeURL = mainURL + requestType + '&' + symbolURL + '&' + apiURL + "&" + datatype
    with urllib.request.urlopen(completeURL) as req:
        data = json.load(req)
        df=pd.DataFrame(data["Time Series (Daily)"])
        df=df.transpose()
        df=df.iloc[::-1]
        return df
#apple = dailyData('AAPL')
#print(apple)
def MinuteData(symbol, requestType=requestTypeURL_2,interval="1min",outputsize="full", apiKey=myKey,datatype="csv"):
    symbolURL = "symbol=" + str(symbol)
    interval = "interval=" + interval
    outputsize= "outputsize=" + outputsize
    apiURL = "apikey=" + myKey
    datatype="datatype" + datatype
    completeURL = mainURL + requestType + '&' + symbolURL + '&' + interval + "&" + outputsize  +  '&' + apiURL + "&" + datatype
    with urllib.request.urlopen(completeURL) as req:
       data = json.load(req)
       df=pd.DataFrame(data['Time Series (1min)']) 
       df=df.transpose()
       df=df.iloc[::-1]
       return df
def normalise_data():
        df=pd.read_csv("FB STOCK DATA.csv")
        mean_close=df["4. close"].mean()
        std_close=df["4. close"].std()
        data=[]
        for i in range(len(df["4. close"])):
            data.append(random.gauss(mean_close,std_close))
        data.sort()
        pdf = stats.norm.pdf(data, mean_close, std_close)
        plt.plot(data, pdf) # including h here is crucial
        plt.show()    
def update_graph():
    df=MinuteData("FB")
    df.to_csv("FB STOCK DATA.csv")
    print("The Data is being updated")
    plt.ion()
    fig, ax = plt.subplots()
    plt.title("FB STOCK DATA")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.plot(df["1. open"][-6:],label="open")
    plt.plot(df["2. high"][-6:],label="high")
    plt.plot(df["3. low"][-6:],label="low")
    plt.plot(df["4. close"][-6:],label="close")
    plt.plot(df["5. volume"][-6:],label="volume")
    plt.legend()
    plt.show()
    plt.pause(60/2)
    print("Fetching the new data")
    plt.close()
    print("Visualising the Data")
    normalise_data()
    plt.pause(60/3)
    plt.close()
start=time.time()
while True:
        print("Please wait-mining the data")
        if time.time()-start>=60:
            update_graph()
            start=time.time()
#def calculate_beta():
#    #CLEANING DATA    
#    df=pd.read_csv("FB Historical Data.csv")
#    df.set_index("Date",inplace=True)
#    df.drop(["Vol."],1,inplace=True)
#    df["Change %"]= df["Change %"].str.replace('%', '')
#    df = df.astype(float)
#    
#    
#    #CLEANING DATA  
#    df_2=pd.read_csv("S&P 500 Historical Data (1).csv")
#    df_2.set_index("Date",inplace=True) 
#    df_2.drop(["Vol."],1,inplace=True)
#    df_2["Price"] = df_2['Price'].str.replace(',', '')
#    df_2["Open"] = df_2["Open"].str.replace(',', '')
#    df_2["High"] = df_2["High"].str.replace(',', '')
#    df_2["Low"]= df_2["Low"].str.replace(',', '')
#    df_2["Change %"]= df_2["Change %"].str.replace('%', '')
#    df_2 = df_2.astype(float)
#        
#    
#    
#    ##CONVERTING TO ARRAY AND MAKING OF SOME LENGHT
#    xs=np.array((df_2["Change %"]))
#    ys=np.array((df["Change %"][:-1]))
#    ##CALCULATING m,b through predefined linear equation formulas
#    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
#         ((mean(xs)*mean(xs)) - mean(xs*xs)))
#    
#    b = mean(ys) - m*mean(xs)
#    print("Your Beta is"+ str(m))
#    regression_line = [(m*x)+b for x in xs]
#    ###drawing the graph
#    plt.scatter(xs,ys,color='#003F72')
#    plt.plot(xs, regression_line)
#    plt.xlabel("Market Return")
#    plt.ylabel("Company Return")
#    plt.title("CAPM MODEL")
#    plt.show()
#    plt.pause(30)
#    plt.close()
#    return m, b
#def expected_return():
#    beta=calculate_beta()[1]
#    expected_returns=[]
#     #CLEANING DATA    
#    df=pd.read_csv("FB Historical Data.csv")
#    df.set_index("Date",inplace=True)
#    df.drop(["Vol."],1,inplace=True)
#    df["Change %"]= df["Change %"].str.replace('%', '')
#    df = df.astype(float)
#    
#    
#    #CLEANING DATA  
#    df_2=pd.read_csv("S&P 500 Historical Data (1).csv")
#    df_2.set_index("Date",inplace=True) 
#    df_2.drop(["Vol."],1,inplace=True)
#    df_2["Price"] = df_2['Price'].str.replace(',', '')
#    df_2["Open"] = df_2["Open"].str.replace(',', '')
#    df_2["High"] = df_2["High"].str.replace(',', '')
#    df_2["Low"]= df_2["Low"].str.replace(',', '')
#    df_2["Change %"]= df_2["Change %"].str.replace('%', '')
#    df_2 = df_2.astype(float)
#    
#    ###Calculating returns
#    for a in range(len(df[:-1])):
#        i=df["Change %"][a]
#        j=df_2["Change %"][a]
#        expected_return=i+float(beta)*(j-i)
#        expected_returns.append(expected_return)
#    mu=mean(expected_returns)
#    sigma=stdev(expected_returns)
#    data=[]
#    for b in range(len(expected_returns)):
#        data.append(random.normalvariate(mu,sigma))
#    data.sort()
#    pdf = stats.norm.pdf(data, mu, sigma)
#    plt.plot(data, pdf) # including pdf here is crucial
#    plt.xlabel("EXPECTED RETURN")
#    plt.show()  
#    return expected_returns
#print(expected_return())
    
    
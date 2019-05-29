# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 21:08:49 2018

@author: MMOHTASHIM
"""
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean,stdev
import numpy as np
import random
import scipy.stats as stats

def calculate_beta():
    #CLEANING DATA    
    df=pd.read_csv("FB Historical Data.csv")
    df.set_index("Date",inplace=True)
    df.drop(["Vol."],1,inplace=True)
    df["Change %"]= df["Change %"].str.replace('%', '')
    df = df.astype(float)
    
    
    #CLEANING DATA  
    df_2=pd.read_csv("S&P 500 Historical Data (1).csv")
    df_2.set_index("Date",inplace=True) 
    df_2.drop(["Vol."],1,inplace=True)
    df_2["Price"] = df_2['Price'].str.replace(',', '')
    df_2["Open"] = df_2["Open"].str.replace(',', '')
    df_2["High"] = df_2["High"].str.replace(',', '')
    df_2["Low"]= df_2["Low"].str.replace(',', '')
    df_2["Change %"]= df_2["Change %"].str.replace('%', '')
    df_2 = df_2.astype(float)
        
    
    
    ##CONVERTING TO ARRAY AND MAKING OF SOME LENGHT
    xs=np.array((df_2["Change %"]))
    ys=np.array((df["Change %"][:-1]))
    ##CALCULATING m,b through predefined linear equation formulas
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    b = mean(ys) - m*mean(xs)
    print("Your Beta is"+ str(m))
    regression_line = [(m*x)+b for x in xs]
    ###drawing the graph
    plt.scatter(xs,ys,color='#003F72')
    plt.plot(xs, regression_line)
    plt.xlabel("Market Return")
    plt.ylabel("Company Return")
    plt.title("CAPM MODEL")
    plt.show()
    plt.pause(30)
    plt.close()
    return m, b
def expected_return():
    beta=calculate_beta()[1]
    expected_returns=[]
     #CLEANING DATA    
    df=pd.read_csv("FB Historical Data.csv")
    df.set_index("Date",inplace=True)
    df.drop(["Vol."],1,inplace=True)
    df["Change %"]= df["Change %"].str.replace('%', '')
    df = df.astype(float)
    
    
    #CLEANING DATA  
    df_2=pd.read_csv("S&P 500 Historical Data (1).csv")
    df_2.set_index("Date",inplace=True) 
    df_2.drop(["Vol."],1,inplace=True)
    df_2["Price"] = df_2['Price'].str.replace(',', '')
    df_2["Open"] = df_2["Open"].str.replace(',', '')
    df_2["High"] = df_2["High"].str.replace(',', '')
    df_2["Low"]= df_2["Low"].str.replace(',', '')
    df_2["Change %"]= df_2["Change %"].str.replace('%', '')
    df_2 = df_2.astype(float)
    
    ###Calculating returns
    for a in range(len(df[:-1])):
        i=df["Change %"][a]
        j=df_2["Change %"][a]
        expected_return=i+float(beta)*(j-i)
        expected_returns.append(expected_return)
    mu=mean(expected_returns)
    sigma=stdev(expected_returns)
    data=[]
    for b in range(len(expected_returns)):
        data.append(random.normalvariate(mu,sigma))
    data.sort()
    pdf = stats.norm.pdf(data, mu, sigma)
    plt.plot(data, pdf) # including h here is crucial
    plt.xlabel("EXPECTED RETURN")
    plt.show()  
    return expected_returns
print(expected_return())
    
    
    
    
    



    
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:28:05 2019

@author: MMOHTASHIM
"""
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
style.use("ggplot")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


df=pd.read_csv("irsis data.txt")
print(df)

list_species=df["species"].values.tolist()
uniques=set(list_species)
x=0
for unique in uniques:
    for i in df["species"]:
       if i==unique:
           df=df.replace(i,x)
    x+=1

df=df.drop("petal_width",1)
X=np.array(df.drop("species",1))
y=np.array(df["species"])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)
clf=svm.SVR()
clf.fit(X_train,y_train)



accuracy=clf.score(X_test,y_test)
print(accuracy)
support_vectors=clf.support_vectors_
print(support_vectors)
colors=["r","b","k"]
X=np.array(df)

a=0
for unique in uniques:
    for i in X:
        if i[-1]==a:
            ax.scatter(i[0],i[1],i[2],c=colors[a])
    a+=1
for s in support_vectors:
    ax.scatter(s[0],s[1],s[2],c="g")
plt.show()


# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset and print head,info of the dataset
2.Check for null values
3.Import kmeans and fit it to the dataset
4.Plot the graph using elbow method
5.Print the predicted array
6.Plot the customer segments

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by:  KUSHMA S 
RegisterNumber:  212224040168


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df= pd.read_csv("Mall_Customers.csv")

df.head()
df.info()
df.isnull().sum()

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init="k-means++",n_init=10)
  kmeans.fit(df.iloc[:,3:])
  wcss.append(kmeans.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1,11),wcss)
plt.xlabel("No of clusters")
plt.ylabel("wcss")
plt.title("Elbow method")
km=KMeans(n_clusters=5,n_init=10)

km.fit(df.iloc[:,3:])
y_pred=km.predict(df.iloc[:,3:])
y_pred

df["cluster"]=y_pred
dt0=df[df["cluster"]==0]
dt1=df[df["cluster"]==1]
dt2=df[df["cluster"]==2]
dt3=df[df["cluster"]==3]
dt4=df[df["cluster"]==4]
plt.scatter(dt0["Annual Income (k$)"],dt0["Spending Score (1-100)"],c="red",label="cluster1")
plt.scatter(dt1["Annual Income (k$)"],dt1["Spending Score (1-100)"],c="black",label="cluster2")
plt.scatter(dt2["Annual Income (k$)"],dt2["Spending Score (1-100)"],c="blue",label="cluster3")
plt.scatter(dt3["Annual Income (k$)"],dt3["Spending Score (1-100)"],c="green",label="cluster4")
plt.scatter(dt4["Annual Income (k$)"],dt4["Spending Score (1-100)"],c="magenta",label="cluster5")
plt.legend()
plt.title("Customer Segments")

*/
```

## Output:

<img width="1156" height="257" alt="497513224-2b122106-e9e6-4171-8b76-dc8be9601816" src="https://github.com/user-attachments/assets/508d53ec-6f6f-4651-9067-b58cefe7b144" />

<img width="1122" height="552" alt="497513244-e387e80e-1a2a-413b-ad48-2b9b883e219b" src="https://github.com/user-attachments/assets/2ad08c90-7912-42a0-8e77-af661e2e2dc8" />

<img width="1256" height="581" alt="497513266-d3d5c946-27aa-4cde-bccb-08217a70c142" src="https://github.com/user-attachments/assets/ec328282-9196-4ac3-b0ba-7bb1e96269a9" />

<img width="1022" height="242" alt="497513301-8df540ff-b5dd-45af-8008-718ddeb97e0c" src="https://github.com/user-attachments/assets/93e20b37-3a79-4404-8faf-fa2e648789e5" />

<img width="1283" height="582" alt="497513426-e9918252-1bf8-4ab4-a6ea-eedf2cf44fdf" src="https://github.com/user-attachments/assets/85c7534a-6b90-41c0-a1a3-aee092c74ccf" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset=pd.read_csv("C:\\Users\\harsh\\Downloads\\Customers.csv")
#print(dataset.head())

X=dataset.iloc[:,[3,4]].values
#print(X)

#Using the elbow method to find the optimal number of clusters

wcss= []
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
'''
plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("No. of clusters")
plt.ylabel("Wcss")
#plt.show()
'''

#Fitting K-MEans to the dataset
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_means=kmeans.fit_predict(X)

#Visualise the clusters

plt.scatter(X[y_means==0,0],X[y_means==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],s=100,c='cyan',label='Cluster 4')
plt.scatter(X[y_means==4,0],X[y_means==4,1],s=100,c='black',label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')

plt.title('Cluster of Customers')
plt.xlabel('Annual Income($)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()

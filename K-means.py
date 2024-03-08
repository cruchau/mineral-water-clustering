#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:39:32 2024

@author: arnaudcruchaudet
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



waters = pd.read_excel("EauxMinÃ©rales.xls", header = 0)


X = waters.iloc[:,1:10]
y = waters['Nom']


#function to standardize data
X1= StandardScaler().fit_transform(X)
kmeans = KMeans(init="random", n_clusters=4, n_init=10, max_iter=300)
kmeans.fit(X1)
y_kmeans = kmeans.predict(X1)
waters.loc[:,"y_kmeans"]=y_kmeans


#Visualization  
plt.scatter(X1[:, 0], X1[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)



labels=y
figure, ax = plt.subplots()
ax.scatter(X1[:, 0], X1[:, 1])
for i, txt in enumerate(labels):
    ax.annotate(txt, (X1[:, 0][i], X1[:, 1][i]))
    
    

# The lowest SSE value
print(kmeans.inertia_)

# Final locations of the centroid
print(kmeans.cluster_centers_)

# The number of iterations required to converge
print(kmeans.n_iter_)

# labels
print(kmeans.labels_)


#Elbow method
kmeans_kwargs = { "init": "random",  "n_init": 100, "max_iter": 300}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X1)
    sse.append(kmeans.inertia_)
    
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()   

# identify the elbow point automatically
kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
print(kl.elbow)
kl.plot_knee()

#We choose K=3   

kmeans = KMeans(init="random", n_clusters=3, n_init=100, max_iter=300)
kmeans.fit(X1)
y_kmeans = kmeans.predict(X1)
waters.loc[:,"y_kmeans_F"]=y_kmeans


# The lowest SSE value
print(kmeans.inertia_)





##########Combining PCA and Kmeans########

#Applying PCA
pca = PCA().fit(X1)
X_pca=pca.transform(X1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
X_pca2=X_pca[:, 0:7]


#Applying Elbow method
kmeans_kwargs = { "init": "random",  "n_init": 100, "max_iter": 300}

sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(X_pca2)
    sse.append(kmeans.inertia_)
    
kl = KneeLocator(range(1,11), sse, curve="convex", direction="decreasing")
print(kl.elbow)
kl.plot_knee()

kmeans = KMeans(init="random", n_clusters=4, n_init=10, max_iter=300)
kmeans.fit(X_pca2)
y_kmeans = kmeans.predict(X_pca2)


# The lowest SSE value
print(kmeans.inertia_)

#Visualization (quite better when combining two algo)

plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


plt.scatter(X_pca2[:, 0], X_pca2[:, 2], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 2], c='black', s=200, alpha=0.5)


#PCA+Kmeans
pca = PCA().fit(X1)
X_pca=pca.transform(X1)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
X_pca2=X_pca[:, 0:7]
kmeans = KMeans(init="random", n_clusters=3, n_init=100, max_iter=300)
kmeans.fit(X_pca2)
y_kmeans = kmeans.predict(X_pca2)


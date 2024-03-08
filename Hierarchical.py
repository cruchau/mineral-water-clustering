#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 10:45:30 2024

@author: arnaudcruchaudet
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


waters = pd.read_excel("EauxMinÃ©rales.xls", header = 0)


X = waters.iloc[:,1:11]
X2= waters.iloc[:,1:10]
y = waters['Nom']


plt.figure(figsize=(10, 7))
plt.title("Mineral waters Dendrogram")

# Dendogram for different method and metrics
selected_data = waters.iloc[:,1:11]
clusters = shc.linkage(selected_data, 
            method='single', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()

selected_data = waters.iloc[:,1:11]
clusters = shc.linkage(selected_data, 
            method='complete', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()

selected_data = waters.iloc[:,1:11]
clusters = shc.linkage(selected_data, 
            method='average', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()


selected_data = waters.iloc[:,1:11]
clusters = shc.linkage(selected_data, 
            method='centroid', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()


selected_data = waters.iloc[:,1:11]
clusters = shc.linkage(selected_data, 
            method='complete', 
            metric="mahalanobis")  # Put weight in distance
shc.dendrogram(Z=clusters)
plt.show()


selected_data = waters.iloc[:,1:11]
clusters = shc.linkage(selected_data, 
            method='ward', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()


cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit(X)
cluster.labels_
cluster_num = cluster.labels_
sns.scatterplot(x="Calcium", 
                y="Prix/litre", 
                data=X, 
                hue=cluster_num, palette="rainbow").set_title('Mineral Waters')
waters.loc[:,"Cluster"]=cluster_num

cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='complete')
cluster.fit(X)
cluster.labels_
cluster_num = cluster.labels_
sns.scatterplot(x="Sodium", 
                y="Prix/litre", 
                data=X, 
                hue=cluster_num, palette="rainbow").set_title('Mineral Waters')
waters.loc[:,"Cluster2"]=cluster_num

#using standarded data

XT= StandardScaler().fit_transform(X2)
selected_data = XT
clusters = shc.linkage(selected_data, 
            method='ward', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()

XT= StandardScaler().fit_transform(X2)
selected_data = XT
clusters = shc.linkage(selected_data, 
            method='single', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()

XT= StandardScaler().fit_transform(X2)
selected_data = XT
clusters = shc.linkage(selected_data, 
            method='complete', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()

XT= StandardScaler().fit_transform(X2)
selected_data = XT
clusters = shc.linkage(selected_data, 
            method='average', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()

XT= StandardScaler().fit_transform(X2)
selected_data = XT
clusters = shc.linkage(selected_data, 
            method='centroid', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show() #Inversion problem



cluster = AgglomerativeClustering(n_clusters=5, affinity='manhattan', linkage='single')
cluster.fit(XT)
cluster.labels_
cluster_num = cluster.labels_
sns.scatterplot(x="Sodium", 
                y="Prix/litre", 
                data=X, 
                hue=cluster_num, palette="rainbow").set_title('Mineral Waters')
waters.loc[:,"Cluster3"]=cluster_num


##############  Adding PCA  ################################"

#Variance analysis ==> choosing number of PC
pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


X1= StandardScaler().fit_transform(X)
pca = PCA(n_components=8)
pca.fit(X1)
print(pca.components_)
print(pca.explained_variance_ratio_)


#Feature vector 
X_pca=pca.fit_transform(X1)
selected_data = X_pca
clusters = shc.linkage(selected_data, 
            method='ward', 
            metric="euclidean")
shc.dendrogram(Z=clusters)
plt.show()


#with manhattan distance & PCA 
cluster = AgglomerativeClustering(n_clusters=4, affinity="manhattan", linkage='average')
cluster.fit(X_pca)
cluster.labels_
cluster_num = cluster.labels_
sns.scatterplot(x="Calcium", 
                y="Prix/litre", 
                data=X, 
                hue=cluster_num, palette="rainbow").set_title('Mineral Waters')
waters.loc[:,"Cluster"]=cluster_num

#with Mahalanobis distance & PCA 

cluster = AgglomerativeClustering(n_clusters=3, affinity="mahalanobis", linkage='average')
cluster.fit(X_pca)
cluster.labels_
cluster_num = cluster.labels_
sns.scatterplot(x="Sodium", 
                y="Calcium", 
                data=X, 
                hue=cluster_num, palette="rainbow").set_title('Mineral Waters')
waters.loc[:,"Cluster"]=cluster_num



cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage='complete')
cluster.fit(X_pca)
cluster.labels_
cluster_num = cluster.labels_
sns.scatterplot(x="Sodium", 
                y="Calcium", 
                data=X, 
                hue=cluster_num, palette="rainbow").set_title('Mineral Waters')
waters.loc[:,"Cluster"]=cluster_num
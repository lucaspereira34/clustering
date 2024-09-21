# -*- coding: utf-8 -*-

#%% Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.stats as stats
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'

#%% Import dataset

dataset = pd.read_csv('credit_card_data.csv')

#%% Dataset information

print(dataset.info())

#%% Descriptive statistics

# drop not relevant variables
df = dataset.drop(columns={'Sl_No', 'Customer Key'})

# descriptive statistics
desc_table = df.describe().T

#%% Standardize variables

df_z = df.apply(zscore, ddof=1)

#%% 3D graph

fig = px.scatter_3d(df_z,
                    x='Avg_Credit_Limit',
                    y='Total_Credit_Cards',
                    z='Total_visits_bank')

fig.show()

#%% Identify number of clusters (Elbow method)

elbow= []

K = range(1,11) # we can set the stop point manually

for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(df_z)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Number of clusters', fontsize=16)
plt.xticks(range(1, 11))
plt.ylabel('WCSS', fontsize=16)
plt.title('Elbow method', fontsize=16)
plt.show()

#%% Identify number os cluster (Silhouette score)

stt = []
J = range(2, 11) # we can set the stop point manually

for j in J:
    kmeansSil = KMeans(n_clusters=j, init='random', random_state=100).fit(df_z)
    stt.append(silhouette_score(df_z, kmeansSil.labels_))
    
plt.figure(figsize=(16,8))
plt.plot(range(2, 11), stt, color='purple', marker='o')
plt.xlabel('Number of clusters', fontsize=16)
plt.ylabel('Silhouette', fontsize=16)
plt.title('Silhouette score', fontsize=16)
plt.axvline(x=stt.index(max(stt))+2, linestyle='dotted', color='red')
plt.show()

#%% K-Means Non-Hierarchical Clustering

# 3 clusters as evidentiated by the previous methods
kmeans = KMeans(n_clusters=3, init='random', random_state=100).fit(df_z)

# create variable for clusters
kmeans_clusters = kmeans.labels_
df['kmeans_cluster'] = kmeans_clusters
df['kmeans_cluster'] = df['kmeans_cluster'].astype('category')
df_z['kmeans_cluster'] = kmeans_clusters
df_z['kmeans_cluster']  = df_z['kmeans_cluster'].astype('category')

#%% ANOVA

pg.anova(dv='Avg_Credit_Limit', 
         between='kmeans_cluster',
         data=df_z,
         detailed=True).T
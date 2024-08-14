# -*- coding: utf-8 -*-

#%% import packages

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
import pingouin as pg
import plotly.express as px 
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

#%% Import dataset

dataset = pd.read_csv('country_data.csv')

#%% Visualize information about data and variables

# descriptive statistics
desc_table = dataset.describe()

# drop column with country labels
data =  dataset.drop(columns=['country'])

# Pearson correlation matrix
corr_mtx = pg.rcorr(data, method='pearson', upper='pval', decimals=4,
                      pval_stars={0.01: '***', 0.05: '**', 0.1: '*'})

#%% Correlation heatmap

# correlation matrix
corr = data.corr()

# heatmap
fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x = corr.columns,
        y = corr.index,
        z = np.array(corr),
        text = corr.values,
        texttemplate='%{text:.2f}',
        colorscale='viridis'
        )
    )

fig.update_layout(height=600, width=600)

fig.show()

#%% Standardize variables

# Apply ZScore to all variables, setting average=0 and std=1 to each variable
z_data = data.apply(zscore, ddof=1)

#%% Agglomerative hierarchical clustering: euclidean distance + single linkage

# Distances
euclidian_dist = pdist(z_data, metric='euclidean')

# Metrics (distances)
    ## euclidean
    ## sqeuclidean
    ## chebyshev
    ## canberra
    ## correlation
    
# Linkage methods
    ## single
    ## complete
    ## average
    
# Create dendrogram
plt.figure(figsize=(16,8))
single_link = sch.linkage(z_data, method='single', metric='euclidean')
dendrogram_s = sch.dendrogram(single_link)
plt.title('Single Linkage Dendrogram', fontsize=16)
plt.xlabel('Countries', fontsize=16)
plt.ylabel('Euclidian Distance', fontsize=16)
plt.show()

#%% Agglomerative hierarchical clustering: euclidean distance + average linkage

plt.figure(figsize=(16,8))
avg_link = sch.linkage(z_data, method='average', metric='euclidean')
dendrogram_a = sch.dendrogram(avg_link)
plt.title('Average Linkage Dendrogram', fontsize=16)
plt.xlabel('Countries', fontsize=16)
plt.ylabel('Euclidean Distance', fontsize=16)
plt.show()

#%% Agglomerative hierachical clustering: euclidean distance + complete linkage

plt.figure(figsize=(16,8))
complete_link = sch.linkage(z_data, method='complete', metric='euclidean')
dendrogram_c = sch.dendrogram(complete_link)
plt.title('Complete Linkage Dendrogram', fontsize=16)
plt.xlabel('Countries', fontsize=16)
plt.ylabel('Euclidean Distance', fontsize=16)
plt.show()

# Create a new variable that stores a cluster label for each index
cluster = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'complete')
set_cluster = cluster.fit_predict(z_data)

data['cluster'] = set_cluster
data.cluster = data.cluster.astype('category')

z_data['cluster'] = set_cluster
z_data.cluster = z_data.cluster.astype('category')

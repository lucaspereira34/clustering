# Clustering

**Clustering Analysis** is an **unsupervised learning** technique used to group observations of a sample into clusters.

These clusters are formed such that **objects within the same group are more similar to each other than to those in other clusters**. Therefore, it is possible to uncover patterns, relationships, and structures that may not be immediately apparent.

As an **unsupervised learning** method:
- It is characterized as an **exploratory technique** and thus does not have a predictive nature for out-of-sample observations.
- If new observations are added to the sample, new clusters must be generated, as the inclusion of new data may alter the group structure.
- If one or more variables are changed, new clusters must also be generated, as modifications in the variables can affect the composition of the clusters.

This project covers two types of clustering:

- **Agglomerative Hierarchical Clustering**: Builds a hierarchy os clusters by starting with individual points and merging them.
- **K-Means**: Divides the data into a predefined number of clusters. K-means iteratively assigns data points to clusters based on their distance to centroids.

## Table of Contents

- [Libraries](#libraries)
- [Agglomerative Hierarchical Clustering](#agglomerative-hierarchical-clustering)

## Libraries

Libraries and functions used in this project.
~~~python
# Import libraries
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
~~~

## Agglomerative Hierarchical Clustering

A dataset containing socio-economic data from 167 countries is used to illustrate this topic.
By employing Agglomerative Hierarchical Clustering, the countries are grouped based on the similarity of their data.

Source: https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data.

The python script for this example is *country.py*.

This analysis relies on specific choices:

- Choice of **dissimilarity measure (distance)**:
  - Distance between observations based on the selected variables;
  - Indicates the degree of difference among observations.
 
- Choice of **linkage method** for observations:
  - Specification of the distance measure when clusters have been formed. 

### Dataset Overview

Read dataset file to a dataframe
~~~python
dataset = pd.read_csv('country_data.csv')
~~~

#### Descriptive Statistics

~~~python
desc_table = dataset.describe()
~~~

<img src="https://github.com/user-attachments/assets/fc066b43-a626-43e7-b3b2-1526eaed277f" alt="Descriptive Statistics" width="550" height="250"> 

#### Pearson Correlation Matrix

~~~python
# drop column with country labels
data =  dataset.drop(columns=['country'])

# Pearson correlation matrix
corr_mtx = pg.rcorr(data, method='pearson', upper='pval', decimals=4,
                      pval_stars={0.01: '***', 0.05: '**', 0.1: '*'})
~~~

<img src="https://github.com/user-attachments/assets/84062bb4-fb42-4291-8ed7-9ed79e58ac4a" alt="Pearson Correlation" width="550" height="250">

#### Standardize variables applying ZScore

~~~python
# Apply ZScore to all variables, setting average=0 and std=1 to each variable
z_data = data.apply(zscore, ddof=1)
~~~

#### Single Linkage

~~~python
plt.figure(figsize=(16,8))
single_link = sch.linkage(z_data, method='single', metric='euclidean')
dendrogram_s = sch.dendrogram(single_link)
plt.title('Single Linkage Dendrogram', fontsize=16)
plt.xlabel('Countries', fontsize=16)
plt.ylabel('Euclidian Distance', fontsize=16)
plt.show()
~~~

<img src="https://github.com/user-attachments/assets/a7f8a26b-3902-4061-ae9e-2692f715cc46" alt="Single Linkage" width="550" height="250">

#### Average Linkage

~~~python
plt.figure(figsize=(16,8))
avg_link = sch.linkage(z_data, method='average', metric='euclidean')
dendrogram_a = sch.dendrogram(avg_link)
plt.title('Average Linkage Dendrogram', fontsize=16)
plt.xlabel('Countries', fontsize=16)
plt.ylabel('Euclidean Distance', fontsize=16)
plt.show()
~~~

<img src="https://github.com/user-attachments/assets/dc379933-3ec5-476f-960d-be25aa8b0bc7" alt="Average Linkage" width="550" height="250">

#### Complete Linkage

~~~python
plt.figure(figsize=(16,8))
complete_link = sch.linkage(z_data, method='complete', metric='euclidean')
dendrogram_c = sch.dendrogram(complete_link)
plt.title('Complete Linkage Dendrogram', fontsize=16)
plt.xlabel('Countries', fontsize=16)
plt.ylabel('Euclidean Distance', fontsize=16)
plt.show()
~~~

<img src="https://github.com/user-attachments/assets/820dd7f5-b79f-4b06-a429-7ca87bd65bd0" alt="Complete Linkage" width="550" height="250">


#### Cluster Labels

Since the complete linkage method returned the best result, we create a new variable that stores labels from the clusters defined by the AgglomerativeClustering function using linkage = 'complete'.

~~~python
cluster = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'complete')
set_cluster = cluster.fit_predict(z_data)

data['cluster'] = set_cluster
data.cluster = data.cluster.astype('category')

z_data['cluster'] = set_cluster
z_data.cluster = z_data.cluster.astype('category')
~~~

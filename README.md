# Introduction

In this project we apply unsupervised learning to organize countries in clusters based on their stats.

The dataset can be found in: https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data

## Packages

~~~python
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

## Visualizing the dataset

~~~python
dataset = pd.read_csv('country_data.csv')
~~~

~~~python
desc_table = dataset.describe()
~~~

~~~python
# drop column with country labels
data =  dataset.drop(columns=['country'])

# Pearson correlation matrix
corr_mtx = pg.rcorr(data, method='pearson', upper='pval', decimals=4,
                      pval_stars={0.01: '***', 0.05: '**', 0.1: '*'})
~~~

#### Descriptive Statistics
<img src="https://github.com/user-attachments/assets/fc066b43-a626-43e7-b3b2-1526eaed277f" alt="Descriptive Statistics" width="550" height="250"> 

#### Pearson Correlation Matrix
<img src="https://github.com/user-attachments/assets/84062bb4-fb42-4291-8ed7-9ed79e58ac4a" alt="Pearson Correlation" width="550" height="250">


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

pat = data.apply(zscore, ddof=1)

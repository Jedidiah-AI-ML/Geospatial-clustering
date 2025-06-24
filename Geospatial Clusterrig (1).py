#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install geopy


# In[2]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from geopy.distance import geodesic


# In[3]:


data = pd.read_csv("../geospatial analysis/train.csv")


# In[4]:


data.head()


# Now let's calculate the real-world distace between the pickup point and the delivery location using the geodesic formula

# In[5]:


def calculate_distance(row):
    return geodesic(
        (row['Restaurant_latitude'], row['Restaurant_longitude']),
        (row['Delivery_location_latitude'], row['Delivery_location_longitude'])
    ).km
data['Distace_km']= data.apply(calculate_distance, axis=1)


# above we wrote a function to calcualte the real-world distance usig geodesic formula, we already imported it so we just passed the values that it will e needing. Then we created a new column and added the values for each row there 
# 
# Visualising all delivery locationns across india on an interactive map using Plotly

# In[6]:


pip install plotly-geo


# In[7]:


pip install plotly


# In[8]:


import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lon=data['Delivery_location_longitude'],
    lat=data['Delivery_location_latitude'],
    mode='markers',
    marker=dict(color='blue', size=6, opacity=0.7),
    name = 'Delivery Locations',
    hovertemplate = 'Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra>Delivery</estra>'
))

fig.update_layout(
    title='Mappig Our Reach - Delivery Locatios Across India',
    geo = dict(
        scope='asia',
        showland=True,
        landcolor='rgb(229,229,229)',
        showcountries=True,
        countrycolor='rgb(220,220,220)',
        showlakes = False,
        lonaxis=dict(range=[68,98]), #focus o india
        lataxis=dict(range=[6,38])
    ),
    margin = dict(l=0,r=0,t=60,b=0),
    showlegend=False
)
fig.show()
    


# the graph shows delivery activity is cocetrated more in the southern ad central regions of india. there's also a moderate spread into cenntral and eastern parts, and relatively fewer delivery points in the northern and northeat zones.

# ## Performin K-Means Clustering
# 
# let's perform K-Means Clustering on delivery locations and visualize the clusters along with their geograghic centroids

# In[12]:


from sklearn.cluster import KMeans

X= data[['Delivery_location_latitude', 'Delivery_location_longitude']]
k=3
kmeans=KMeans(n_clusters=k, random_state=42)
data['Cluster']= kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

fig = go.Figure()

for cluster_label in sorted(data['Cluster'].unique()):
    cluster_data = data[data['Cluster']== cluster_label]
    fig.add_trace(go.Scattergeo(
        lon=cluster_data['Delivery_location_longitude'],
        lat=cluster_data['Delivery_location_latitude'],
        mode='markers',
        marker=dict(size=6, opacity=0.7),
        hovertemplate='<b>Cluster:</b> %{text}<br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>',
        text=[f"{cluster_label}"] * len(cluster_data)
        
    ))

fig.add_trace(go.Scattergeo(
    lon=centroids[:, 1],
    lat=centroids[:, 0],
    mode= 'markers',
    name='Centroids',
    marker=dict(size=15, symbol='x', color='red', line=dict(width=2, color='black')),
    hovertemplate='<b>Centroid</b><br>Lat: %{lat:.4f}<br>Lon: %{lon:.4f}<extra></extra>'
))

fig.update_layout(
    title=f' Geo-spatial Clustering of Delivery Locations (k={k})',
    geo=dict(
        scope='asia',
        showland=True,
        landcolor="rgb(229,229,229)",
        showcountries=True,
        countrycolor="rgb(204, 204, 204)",
        lonaxis=dict(range=[68, 98]),
        lataxis=dict(range=[6,38]),
    ),
    legend_title='Clusters',
    margin=dict(l=0, r=0, t=60, b=0)
)
fig.show()


# In[ ]:





# In[14]:


filtered_data = data[data['Cluster'] != 1]
filtered_centroids= centroids[[0,2]] #keep only clusters 0 and 2
# step 3: map clustter nmes
cluster_label={
    0: "Central Delivery Zone",
    2: "Southern Delivery Zone"
}
filtered_data['Optimized_Zone'] = filtered_data['Cluster'].map(cluster_label)


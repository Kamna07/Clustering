#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


# In[2]:


data = pd.read_csv(r'C:\Users\ony\Downloads\Machine Learning A-Z Template Folder\Part 4 - Clustering\Section 25 - Hierarchical Clustering\Hierarchical_Clustering\Mall_Customers.csv')
print(data.head(n=10))
x = data.iloc[:, [3,4]].values


# In[8]:


import scipy.cluster.hierarchy as hs
dendrogram = hs.dendrogram(hs.linkage(x,method = 'ward'))
plt.title("DENDORAM")
plt.xlabel('customer')
plt.ylabel('distance')
plt.show()


# In[15]:


ac = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean')
y = ac.fit_predict(x)


# In[16]:


plt.scatter(x[y == 0,0], x[y == 0,1], s = 100, c='red',label = '1')
plt.scatter(x[y == 1,0], x[y == 1,1], s = 100, c='blue',label = '2')
plt.scatter(x[y == 2,0], x[y == 2,1], s = 100, c='green',label = '3')
plt.scatter(x[y == 3,0], x[y == 3,1], s = 100, c='cyan',label = '4')
plt.scatter(x[y == 4,0], x[y == 4,1], s = 100, c='magenta',label = '5')

plt.title("Clusters of clients")
plt.xlabel("Annual income")
plt.ylabel("spending score")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





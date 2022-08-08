#!/usr/bin/env python
# coding: utf-8

# # reading the data 
# 

# In[1]:


import pandas as pd
import numpy as np
db_milesDriven = pd.read_csv(r"C:\Users\Amir\technolab\miles-driven.csv", sep = '|')
db_road_accidents=pd.read_csv(r"C:\Users\Amir\technolab\road-accidents.csv", sep = '|', skiprows = 9)



# # overview about data

# In[2]:


db_milesDriven.head()


# In[3]:


db_road_accidents.head()


# # textual and a graphical summary of the data

# In[4]:


db_milesDriven.describe()


# In[5]:


db_road_accidents.describe()


# In[6]:


import seaborn as sns
sns.histplot(data=db_road_accidents)


# In[7]:



sns.displot(data=db_milesDriven)


# In[8]:


sns.histplot(data=db_road_accidents , x=db_road_accidents['perc_fatl_speed'])


# In[9]:


sns.histplot(data=db_road_accidents , x=db_road_accidents['perc_fatl_alcohol'])


# In[10]:


sns.histplot(data=db_road_accidents , x=db_road_accidents['perc_fatl_1st_time'])


# In[11]:


g = sns.PairGrid(db_road_accidents)
g.map_offdiag(sns.regplot)
g.add_legend()


# # quantify the association

# In[12]:


from numpy.random import seed
from scipy.stats import pearsonr
seed(1)


# In[13]:


corr_1 = np.corrcoef(db_road_accidents['perc_fatl_1st_time'], db_road_accidents['drvr_fatl_col_bmiles'])
corr_1


# In[14]:


corr_2 =  np.corrcoef(db_road_accidents['perc_fatl_speed'], db_road_accidents['drvr_fatl_col_bmiles'])
corr_2


# In[15]:


corr_3 =  np.corrcoef(db_road_accidents['perc_fatl_alcohol'], db_road_accidents['drvr_fatl_col_bmiles'])
corr_3


# # linear regression

# In[111]:




# Import the linear model function from sklearn
from sklearn import linear_model

# Create the features and target DataFrames
features = db_road_accidents[['perc_fatl_speed', 'perc_fatl_alcohol', 'perc_fatl_1st_time']]
target = db_road_accidents['drvr_fatl_col_bmiles']

# Create a linear regression object
reg = linear_model.LinearRegression()

# Fit a multivariate linear regression model
reg.fit(features, target)

# Retrieve the regression coefficients
fit_coef = reg.coef_
fit_coef


# A=perc_fatl_1st_timeB=perc_fatl_alcohol 
# Y=drvr_fatl_col_bmiles
# A->B -VE
# B-> Y +VE
# A-> Y  -VE
# in corr
# A->Y +VE in regression
# then 'perc_fatal_1st_time' is a masking relationship

# # standraization 

# In[112]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# # perform PCA and vizualize the first two component

# In[113]:


from sklearn.decomposition import PCA
pca = PCA()

# Fit the standardized data to the pca
pca.fit(features_scaled)

# Plot the proportion of variance explained on the y-axis of the bar plot
import matplotlib.pyplot as plt
plt.bar(range(1, pca.n_components_ + 1),  pca.explained_variance_ratio_)
plt.xlabel('Principal component #')
plt.ylabel('Proportion of variance explained')
plt.xticks([1, 2, 3])

# Compute the cumulative proportion of variance explained by the first two principal components
two_first_comp_var_exp = sum(pca.explained_variance_ratio_[:2])
print("The cumulative variance of the first two principal components is {}".format(
    round(two_first_comp_var_exp, 5)))


# In[114]:


# Transform the scaled features using two principal components
pca = PCA(n_components=2)
p_comps = pca.fit_transform(features_scaled)

# Extract the first and second component to use for the scatter plot
p_comp1 = p_comps[:,0]
p_comp2 = p_comps[:,1]

# Plot the first two principal components in a scatter plot
plt.scatter(p_comp1, p_comp2)


# In[115]:


# Import KMeans from sklearn
from sklearn.cluster import KMeans

# A loop will be used to plot the explanatory power for up to 10 KMeans clusters
ks = range(1, 10)
inertias = []
for k in ks:
    # Initialize the KMeans object using the current number of clusters (k)
    km = KMeans(n_clusters=k, random_state=8)
    # Fit the scaled features to the KMeans object
    km.fit(features_scaled)
    # Append the inertia for `km` to the list of inertias
    inertias.append(km.inertia_)
    
# Plot the results in a line plot
plt.plot(ks, inertias, marker='o')


# In[116]:


# Create a KMeans object with 3 clusters, use random_state=8 
km = KMeans(n_clusters=3, random_state=8)

# Fit the data to the `km` object
km.fit(features_scaled)

# Create a scatter plot of the first two principal components
# and color it according to the KMeans cluster assignment 
plt.scatter(p_comps[:,0], p_comps[:,1], c=km.labels_)


# In[117]:


Kmean.cluster_centers_


# # Visualize the feature differences between the clusters

# In[122]:


# Create a new column with the labels from the KMeans clustering
db_road_accidents['cluster'] = km.labels_

# Reshape the DataFrame to the long format
melt_car = pd.melt(db_road_accidents, id_vars=['cluster'], 
                   value_vars=['perc_fatl_speed', 'perc_fatl_alcohol', 'perc_fatl_1st_time'],
                   var_name='measurement', value_name='percent')

melt_car.head()


# In[123]:


# Create a violin plot splitting and coloring the results according to the km-clusters
sns.violinplot(x='percent', y='measurement', data=melt_car, hue='cluster' )


# # merging miles driven with each state in car accidents dataset

# In[128]:


miles_driven = db_milesDriven['million_miles_annually']
total = pd.concat([db_road_accidents,miles_driven],axis=1)
total.head()


# In[134]:


num_fatal=db_road_accidents['drvr_fatl_col_bmiles']*db_milesDriven['million_miles_annually']/1000
finall = pd.concat([total,num_fatal],axis=1)
finall.rename(columns = {0 : 'num_fatal'}, inplace = True)
finall.head()


# In[135]:


# # Create a barplot of the total number of accidents per cluster
sns.barplot(x='cluster', y='num_fatal', data=finall, estimator=sum, ci=None)

# # Calculate the number of states in each cluster and their 'num_drvr_fatl_col' mean and sum.
count_mean_sum = finall.groupby('cluster').agg(['count', 'mean', 'sum'])['num_fatal']
count_mean_sum


# In[133]:


# Which cluster would you choose?
cluster_num = 2


# In[ ]:





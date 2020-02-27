#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib as pl
import seaborn as sns
import json
import numpy as np


# In[2]:


Movie_df=pd.read_csv("/Volumes/Ankur/DataSet for ML Project/movies.csv")


# In[3]:


Ratings_df=pd.read_csv("/Volumes/Ankur/DataSet for ML Project/ratings.csv")


# In[4]:


print (Ratings_df.head())


# In[5]:


# Fill NaN values in user_id and movie_id column with 0
Ratings_df['userId'] = Ratings_df['userId'].fillna(0)
Ratings_df['movieId'] = Ratings_df['movieId'].fillna(0)

# Replace NaN values in rating column with average of all values
Ratings_df['rating'] = Ratings_df['rating'].fillna(Ratings_df['rating'].mean())


# In[25]:


# Randomly sample 1% of the ratings dataset
small_data = Ratings_df.sample(frac=0.0015)
# Check the sample info
print(small_data.info())


# In[26]:


from sklearn.model_selection import train_test_split
train_data, test_data =train_test_split(small_data, test_size=0.2)


# In[27]:


# Create two user-item matrices, one for training and another for testing
train_data_matrix = train_data.values
test_data_matrix = test_data.values

# Check their shape
print(train_data_matrix.shape)
print(test_data_matrix.shape)


# In[28]:


from sklearn.metrics.pairwise import pairwise_distances

# User Similarity Matrix
user_correlation = 1 - pairwise_distances(train_data, metric='correlation')
user_correlation[np.isnan(user_correlation)] = 0
print(user_correlation[:4, :4])


# In[29]:


# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(train_data_matrix.T, metric='correlation')
item_correlation[np.isnan(item_correlation)] = 0
print(item_correlation[:4, :4])


# In[30]:


# Function to predict ratings
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # Use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


# In[31]:


from sklearn.metrics import mean_squared_error
from math import sqrt

# Function to calculate RMSE
def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))


# In[32]:


# Predict ratings on the training data with both similarity score
user_prediction = predict(train_data_matrix, user_correlation, type='user')
item_prediction = predict(train_data_matrix, item_correlation, type='item')

# RMSE on the test data
print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))


# In[33]:


# RMSE on the train data
print('User-based CF RMSE: ' + str(rmse(user_prediction, train_data_matrix)))
print('Item-based CF RMSE: ' + str(rmse(item_prediction, train_data_matrix)))


# In[ ]:





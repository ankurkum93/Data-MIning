#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing the libraries

import pandas as pd
import numpy as np


# In[2]:


#Reading the dataset
ratings=pd.read_csv("/Volumes/Ankur/DataSet for ML Project/ratings.csv")


# In[3]:


ratings.head(5)


# In[4]:


movie=pd.read_csv("/Volumes/Ankur/DataSet for ML Project/movies.csv")


# In[5]:


movie.head(5)


# In[6]:


movie=pd.merge(movie,ratings, on="movieId")
movie.head()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sea
sea.set_style('white')


# In[8]:


#Grouping by title and rating by taking the mean ratings and in descending order
movie.groupby('title')["rating"].mean().sort_values(ascending=False).head()


# In[9]:


movie.groupby('title')["rating"].count().sort_values(ascending=False).head()


# In[10]:


ratings=pd.DataFrame(movie.groupby('title')['rating'].mean())
ratings.head()


# In[11]:


ratings['num of ratings']=pd.DataFrame(movie.groupby('title')['rating'].count())
ratings.head()


# In[12]:


plt.figure(figsize=(12,6))
ratings['num of ratings'].hist(bins=70)


# In[13]:


plt.figure(figsize=(12,6))
ratings['rating'].hist(bins=70)


# In[14]:


sea.jointplot(x='rating', y='num of ratings', data=ratings, alpha=0.5)


# In[15]:


ratings.sort_values("num of ratings", ascending=False).head(10)


# In[16]:


movie.head(10)


# In[17]:


movie=movie.drop(['genres'], axis=1)


# In[18]:


movie.head(10)


# In[19]:


movie.count()


# In[20]:


movie=movie.drop(['timestamp'],axis=1)


# In[21]:


movie.head()


# movie(['title']).unique().tolist()

# In[22]:


combine_movie_rating=movie.dropna(axis=0,subset=['title'])


# In[23]:


movie_ratingCount = (combine_movie_rating.
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    )
movie_ratingCount.head()


# In[24]:



rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
rating_with_totalRatingCount.head()


# In[25]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)
print(movie_ratingCount['totalRatingCount'].describe())


# In[26]:


popularity_threshold = 1000
rating_popular_movie= rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
rating_popular_movie.head()


# In[27]:



rating_popular_movie.shape


# In[ ]:


## create a Pivot matrix

movie_features_df=rating_popular_movie.pivot_table( index='title',columns='userId',values='rating').fillna(0)


# In[ ]:


from scipy.sparse import csr_matrix

movie_features_df_matrix = csr_matrix(movie_features_df.values)

from sklearn.neighbors import NearestNeighbors


model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_features_df_matrix)


# In[ ]:


query_index=np.random.choice(movie_features_df.shape[0])
print(query_index)
distances,indices=model_knn.kneighbors(movie_features_df.iloc[query_index,:].values.reshape(1,-1),n_neighbors=4)


# In[ ]:


movie_features_df.head()


# In[ ]:


for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_features_df.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]], distances.flatten()[i]))


# In[ ]:





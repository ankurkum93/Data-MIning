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


print (Ratings_df.info())


# In[6]:


print(Movie_df.info())


# In[7]:


print(Movie_df.head())


# In[8]:


Ratings_df=Ratings_df.drop("timestamp", axis=1)


# In[9]:


from matplotlib import pyplot as plt


# In[10]:


# Import new libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import wordcloud
from wordcloud import WordCloud, STOPWORDS

# Create a wordcloud of the movie titles
Movie_df['title'] = Movie_df['title'].fillna("").astype('str')
title_corpus = ' '.join(Movie_df['title'])
title_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', height=2000, width=4000).generate(title_corpus)

# Plot the wordcloud
plt.figure(figsize=(16,8))
plt.imshow(title_wordcloud)
plt.axis('off')
plt.show()


# In[11]:


import seaborn as sns
sns.set_style('whitegrid')
sns.set(font_scale=1.5)
get_ipython().run_line_magic('matplotlib', 'inline')

# Display distribution of rating
sns.distplot(Ratings_df['rating'].fillna(Ratings_df['rating'].median()))


# In[12]:


# Join both the files into one dataframe
dataset = pd.merge(Movie_df, Ratings_df)
# Display 20 movies with highest ratings
dataset[['title','genres','rating']].sort_values('rating', ascending=False).head(20)


# In[13]:


# Make a census of the genre keywords
genre_labels = set()
for s in Movie_df['genres'].str.split('|').values:
    genre_labels = genre_labels.union(set(s))

# Function that counts the number of times each of the genre keywords appear
def count_word(dataset, ref_col, census):
    keyword_count = dict()
    for s in census: 
        keyword_count[s] = 0
    for census_keywords in dataset[ref_col].str.split('|'):        
        if type(census_keywords) == float and pd.isnull(census_keywords): 
            continue        
        for s in [s for s in census_keywords if s in census]: 
            if pd.notnull(s): 
                keyword_count[s] += 1
    #______________________________________________________________________
    # convert the dictionary in a list to sort the keywords by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count

# Calling this function gives access to a list of genre keywords which are sorted by decreasing frequency
keyword_occurences, dum = count_word(Movie_df, 'genres', genre_labels)
keyword_occurences[:5]


# In[14]:


# Define the dictionary used to produce the genre wordcloud
genres = dict()
trunc_occurences = keyword_occurences[0:18]
for s in trunc_occurences:
    genres[s[0]] = s[1]

# Create the wordcloud
genre_wordcloud = WordCloud(width=1000,height=400, background_color='white')
genre_wordcloud.generate_from_frequencies(genres)

# Plot the wordcloud
f, ax = plt.subplots(figsize=(16, 8))
plt.imshow(genre_wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[15]:


#Content Based Recommender System
# Break up the big genre string into a string array
Movie_df['genres'] = Movie_df['genres'].str.split('|')
# Convert genres to string value
Movie_df['genres'] = Movie_df['genres'].fillna("").astype('str')


# In[16]:


Movie_df.head()


# In[17]:


#Using TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(Movie_df['genres'])
tfidf_matrix.shape


# I will be using the Cosine Similarity to calculate a numeric quantity that denotes the similarity between two movies. Since we have used the TF-IDF Vectorizer, calculating the Dot Product will directly give us the Cosine Similarity Score. Therefore, we will use sklearn's linear_kernel instead of cosine_similarities since it is much faster.

# In[18]:


from sklearn.metrics.pairwise import linear_kernel
Lin_ker = linear_kernel(tfidf_matrix, tfidf_matrix)
Lin_ker[:4, :4]


# In[28]:


# Build a 1-dimensional array with movie titles
#titles = Movie_df['title']
indices = pd.Series(Movie_df.index, index=Movie_df['title'])

query_index=np.random.choice(Movie_df.title)

# Function that get movie recommendations score of movies based on the  Linear Kernel  
def genre_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(Lin_ker[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

print('Recommendations for {0}:\n'.format(genre_recommendations(query_index)))


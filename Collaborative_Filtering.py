# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:58:08 2020

@author: Manuel Antonio Noriega Ramos
"""

#Setting up the environment

import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, split

#Retrieving Data

m_df = pd.read_csv('movies_metadata.csv')

small_df_id = pd.read_csv(r'datasets_3405_6663_links_small.csv')

keywords = pd.read_csv('keywords.csv')

credits_df = pd.read_csv('IMDB-Dataset\credits.csv')

ratings_small = pd.read_csv('ratings_small.csv')

######################### EXPLORING THE DATA #################################

#Cheking for nulls and see data types
m_df.info()

small_df_id.info()

keywords.info()

credits_df.info()

ratings_small.info()

#Define function to convert string into int
def convert_int(row):
    try:
        return int(row)
    except:
        return np.nan

#Define function to convert string into float
def convert_float(row):
    try:
        return float(row)
    except:
        return np.nan

    
#In Case of values that are not compatible with literal_eval function
def literal_evaluation(row):
    try:
        return literal_eval(row)
    except:
        return np.nan
    
#Formating the columns

m_df['id'] = m_df['id'].apply(convert_int) #converting id column into int

m_df['genres'] = m_df['genres'].apply(literal_eval) #transforming string values

#Extract information from genre in this case we are interested in 

def extract_values(row):
    
    values_list = [i['name'] for i in row]
    row = ''
    for i in range(len(values_list)):
        row = row +' '+ values_list[i]
    row = row[1:]
    return row

#Extracting values from genres column
m_df['genres'] = m_df['genres'].apply(extract_values)

'''Calculating Weighted Rating according to IMDB (WR) = $(\frac{v}{v + m} . R) + (\frac{m}{v + m} . C)$
where,

v is the number of votes for the movie
m is the minimum votes required to be listed in the chart
R is the average rating of the movie
C is the mean vote across the whole report
'''
#Calculating C
C = np.mean(m_df['vote_average'])
                 
#Calculating m, in this case we select movies with votes higher than the 80% of the movies in the report
m = m_df['vote_count'].quantile(0.8)

#Define weighted rating function
def weighted_rating(df, C, m):
    
    def RW(row):
        v = row['vote_count']
        R = row['vote_average']
        row['Weighted_Rating'] = (v/(v+m))*R + (m/(v+m))*C
        return row
    
    return df.apply(RW, axis=1)

#Calculating Weighted Rating
m_df = weighted_rating(m_df, C, m)

#Top 25 movies according to their weighted rating
m_df.sort_values(by=['Weighted_Rating'], ascending = False)[['title','Weighted_Rating']].iloc[0:25]

#Extracting values from keywords
keywords['keywords'] = keywords['keywords'].apply(literal_evaluation)#converting string values into list
keywords['keywords'] = keywords['keywords'].apply(extract_values)#extract keywords into text

#DATA VISUALIZATION 

#Top 10 Movie Genre 
plt.rc('axes', titlesize=14) # fontsize of the axes title
plt.rc('xtick', labelsize=10) # fontsize of the tick labels
plt.rc('ytick', labelsize=10) # fontsize of the tick labels
plt.title('Top 10 Movie Genres') # Plot title
m_df.genres.str.split(expand=True).stack().value_counts()[0:10].plot.bar(color='#05C081')


#Average Weighted Rating by Genre

genre_list = list(m_df.genres.str.split(expand=True).stack().value_counts()[0:10].index)

def plot_weighted_rating_genre(df, genre_list):
    
    genre_dict = {}
    
    for genre in genre_list:  
      genre_dict['{}'.format(genre)]=np.mean(df[df['genres'].str.contains(genre)]['Weighted_Rating'])
      
    plt.rc('axes', titlesize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.title('Weighted Average by Genre')
    return pd.Series(genre_dict, index= genre_dict.keys()).sort_values(ascending=False).plot.bar(color='#05AFC0')
        
plot_weighted_rating_genre(m_df, genre_list)

#Movie Distribution by decade

#Data for the pie chart
m_df['release_date'] = pd.to_datetime(m_df['release_date'], errors='coerce')
dates = pd.DatetimeIndex(m_df['release_date'].dropna()).year
dates = dates.groupby((dates//10)*10)
dates = pd.Series({k:len(v) for (k,v) in dates.items()})

#Pie Chart parameter values
x = dates.index.astype('str')
y = dates.values
percent = 100.*y/y.sum()
explode = dates.apply(lambda x: 0.2 if 100.*x/y.sum() > 20 else 0)

#Pie Chart feature config

plt.rcParams["figure.figsize"] = (20,10) # figure size
plt.rc('axes', titlesize=25) # fontsize of the axes title
plt.title('Movie Distribution by Decade')


patches, texts, percentages = plt.pie(y, startangle=90, radius=1.2, autopct='%1.2f%%', explode=explode)
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]
sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                          key=lambda x: x[2],
                                          reverse=True))
    
plt.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.4, 0.5),
           fontsize=14)
plt.show()

#Top 10 most popular movies

m_df['popularity']= m_df['popularity'].apply(convert_float)
popular_movies = m_df.loc[:,['original_title','popularity']].sort_values('popularity', ascending=False)[0:10]
plt.rcParams["figure.figsize"] = (12,10)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=16)
plt.barh(popular_movies.original_title, popular_movies.popularity, color='#05C081')
plt.gca().invert_yaxis()
plt.title('Top 10 most popular movies')


#Weighted Rating Distribution
#Data for the pie chart

labels= ['1-2', '2-3', '3-4', '4-5', '5-6', '6-7','7-8','8-9','9-10']
w_ratings = pd.cut(m_df.Weighted_Rating, bins=np.linspace(1,10, 10), labels= labels).value_counts(normalize=True)*100

#Pie Chart parameter values
x = w_ratings.index.astype('str')
y = w_ratings.values
explode = w_ratings.apply(lambda x: 0.2 if x > 20 else 0)

#Pie Chart feature config

plt.rcParams["figure.figsize"] = (20,10) # figure size
plt.rc('axes', titlesize=25) # fontsize of the axes title
plt.title('Weighted Rating Distribution')


patches, texts, percentages = plt.pie(y, startangle=90, radius=1.2, autopct='%1.2f%%', explode=explode)
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, y)]
sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),
                                          key=lambda x: x[2],
                                          reverse=True))
    
plt.legend(patches, labels, loc='center left', bbox_to_anchor=(-0.4, 0.5),
           fontsize=14)
plt.show()

################## BUILDING CONTENT BASED RECOMMENDER SYSTEM #################

'''To build this recommender system content based it will use the credits information, 
the sinopis, the genres and keywords'''
#Extrating movies from the data for better performance
sm_df = m_df[m_df['id'].isin(small_df_id['tmdbId'])]

#Extracting values from credits
credits_df['cast'] = credits_df['cast'].apply(literal_evaluation)
credits_df['crew'] = credits_df['crew'].apply(literal_evaluation)

#Define function to extract credits removing spaces in the names

def extract_cast(row):
    
    values_list = [i['name'].replace(' ','') for i in row]
    row = ''
    
    if len(values_list) >= 3:
    
        for i in range(3):
            row = row +' '+ values_list[i]
    else:
        for i in range(len(values_list)):
            row = row +' '+ values_list[i]
    row = row[1:]
        
    return row

#Extract 3 or less main actors from cast 
credits_df['cast'] = credits_df['cast'].apply(extract_cast)

#Extracting Director's name from crew

def extract_director(row):

    values_list = [i['name'].replace(' ','') for i in row if i['job'] == 'Director']
    row = ''
    for i in range(len(values_list)):
        row = row +' '+ values_list[i]
    row = row[1:]
    return row

#Extracting Director from crew column
credits_df['director'] = credits_df['crew'].apply(extract_director)

#Merging dataframes

sm_df = sm_df.merge(keywords, on='id')
sm_df = sm_df.merge(credits_df, on='id')

#id column into int
sm_df['id'] = sm_df['id'].apply(convert_int) 

#Retrieving relevant data
df = sm_df.loc[:,['id','genres','original_title', 'overview',
      'tagline', 'title','vote_average', 'vote_count', 
      'Weighted_Rating', 'keywords', 'cast','director']]

#Exploring Data
df.info()

#Define function to Tokanize columns overview and keywords
def tokenize_lemmatization(row):
    try:
        WNlemma = nltk.WordNetLemmatizer()
        tokens = nltk.word_tokenize(row)
        l_tokens = [WNlemma.lemmatize(str.lower(w)) for w in tokens]
        row = ''
        for i in range(len(l_tokens)):
            row = row +' '+ l_tokens[i]
        row = row[1:]
        return row
    except:
        row = ' '
        return row

#Tokanize columns overview and keywords
df['keywords'] = df['keywords'].apply(tokenize_lemmatization) 
df['overview'] = df['overview'].apply(tokenize_lemmatization)

#Combine all columns into column Combined_Features
def combine_features(row):
    
    columns = ['title', 'genres', 'overview',
              'keywords','cast', 'director']
    
    row['Combined_Features'] = ''
    
    for i in columns:
        
        if row[i] != ' ':
            
            row['Combined_Features'] = row['Combined_Features']+' '+row[i]+' '
    row[1:]

    return row

#Creating Combined Features
df = df.apply(combine_features, axis=1)

#Adding index Column 
df.reset_index(inplace = True)

#IMPLEMENTING MODEL

vect = vect = CountVectorizer(max_df=0.70, ngram_range=(1,2), analyzer='word')

vect.fit(df['Combined_Features']) #training the model 

count_matrix = vect.transform(df['Combined_Features']) #transform series into frequency matrix

cosine_sim = cosine_similarity(count_matrix) #create similarity matrix

#Creating function to retrieve index from title
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

#Creating function to retrieve title from index
def get_title_from_index(index):
    return df[df.index == index][["title", "Weighted_Rating"]].values[0]

#Function that retrieves the top 5 similar movies to the movie that the user likes

def movie_top_list(title):
    
    index = get_index_from_title(title)
    
    similar_movies = list(enumerate(cosine_sim[index]))
    
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
    
    list_movies = [get_title_from_index(i) for i in [j[0] for j in sorted_similar_movies][0:20]]
    
    sorted_by_wr = sorted(list_movies,key=lambda x:x[1],reverse=True)[0:10]
            
    for movie in sorted_by_wr:
        
        print(movie)
        
    return

#################### EXAMPLE 1: CONTENT BASE REOMMENDATIONS ##################
title = 'Star Wars'
movie_top_list(title)

################# INTRODUCING COLLABORATIVE FILTERING TO THE MODEL ###########

#Merging ratings df with df on id and movieid column so the movieId and index can match
df_cf = df.loc[:, ['index', 'id']].merge(ratings_small, left_on='id', right_on='movieId')

#Creating an instace to read the data set
reader = Reader()

#Preparing Data for model
data = Dataset.load_from_df(ratings_small[['userId', 'movieId', 'rating']], reader)

#Creating instance SVD predicting model
svd = SVD()

#Cross Validation with 5 folds to see performance of the SVD model
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose = True)

#Setting up trainig data
trainset = data.build_full_trainset()

#Training Model
svd.fit(trainset)

#Making prediction with uder id=1, 
svd.predict(1, 5).est

###################### FUSING BOTH MODELS #####################################

#Defining function to retrieve top 10 recommendations according to the user id and tittle of the movie

def CF_RS(UserId, title):
    
    index = get_index_from_title(title)
    
    similar_movies = list(enumerate(cosine_sim[index]))
    
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
    
    list_movies = [np.reshape(get_title_from_index(i), (1,2) )for i in [j[0] for j in sorted_similar_movies][0:20]]
    
    est_scores = pd.Series([svd.predict(UserId, i).est for i in [j[0] for j in sorted_similar_movies][0:20]])
    
    df_cfrs = pd.DataFrame(np.concatenate(list_movies), columns = ['Title', 'Weighted_Rating'])
    
    df_cfrs['User_Est'] = est_scores
    
    df_cfrs = df_cfrs.sort_values(by='Weighted_Rating', ascending = False).head(10)
    
    df_cfrs.sort_values(by='User_Est', ascending = False, inplace = True)
    
    return df_cfrs

###### EXAMPLE 2: USING COLLABORATIVE AND CONTENT BASED APPROACH #############

UserId = 10 #User Id one
title = 'The Terminator' #Setting the movie avatar Avatar as example
CF_RS(UserId, title)

####################### SPELLING RECOMMENDER ####################################

nltk.download('punkt')
nltk.download('wordnet')

#Creating list of wordsfrom the data set
correct_spellings = list(df.title)

'''Definig fuction that uses the Jaccard Distance as a similarity score. In this 
function we define the number of ngrams that will be used to perform the similarity
score. The entries will be a list of one or more words. The uoutput of this function will
be the 5 movie titles with the highest Jaccard Score, therefore the most similar 
movie titles according to the input that the user writes'''

def jaccard_similarity(entries, n_grams):
    
    entries = [i.lower() for i in entries]
    jaccard_dict = {}
    n_grams = n_grams

    for i in range(len(entries)):
        
        jaccard_score = [(nltk.jaccard_distance(set(nltk.ngrams(entries[i], n= n_grams)), set(nltk.ngrams(w, n = n_grams))), w) for w in correct_spellings]
            
        jaccard_dict['{}'.format(entries[i])] =  jaccard_score    
        
        jaccard_score = []    
        
    for k in jaccard_dict.keys():
        
        jaccard_dict[k] = sorted(jaccard_dict[k])
    
    recommendation_list = [w[i][1] for i in range(5) for w in jaccard_dict.values()]
    
    return recommendation_list

###################### EXAMPLE 3: SPELL CHECKING ##############################

#Spelling check using Jaccard Distance Coefficient

entries = ['starr waars'] #Movie title input from the user
n_grams = 3 #set ngram for similarity calculation
jaccard_similarity(entries, n_grams) #retrieving 5 highest js movie titles






























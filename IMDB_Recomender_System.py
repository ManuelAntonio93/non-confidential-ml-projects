# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 10:08:55 2020

@author: Manuel Antonio Noriega Ramos
"""
import pandas as pd
import numpy as np

df = pd.read_csv('imdb.csv', error_bad_lines=False)

########################### PREPARING THE DATA ################################

#Checking missing data and dimentionality 
df.info()
#Adding Column Index
df.reset_index(inplace=True)

#Retrieving Relevant Data for the movie engine
df = df.loc[:, ['index', 'title','imdbRating','type', 'Action', 
                'Adult', 'Adventure', 'Animation', 'Biography',
                'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
                'FilmNoir', 'GameShow', 'History', 'Horror', 'Music', 'Musical',
                'Mystery', 'News', 'RealityTV', 'Romance', 'SciFi', 'Short', 'Sport',
                'TalkShow', 'Thriller', 'War', 'Western']]

#Gettin rid of the (year) expression in title column
df['title'] = df['title'].replace(r' [(]\d+[)]', '', regex=True)

#list column names
features = list(df.columns) 

#Filling Nans with blank space
for column in features:
    
    df[column].fillna(' ', inplace = True)

#Function bynary variables into text

def binary_to_text(row):
    
    genre_list = ['Action', 'Adult', 'Adventure', 'Animation', 'Biography',
              'Comedy', 'Crime', 'Documentary', 'Drama', 'Family', 'Fantasy',
              'FilmNoir', 'GameShow', 'History', 'Horror', 'Music', 'Musical',
              'Mystery', 'News', 'RealityTV', 'Romance', 'SciFi', 'Short', 'Sport',
              'TalkShow', 'Thriller', 'War', 'Western']
    
    for i in genre_list:
        if row[i] == 1:
            row[i] = i
        else:
            row[i] = ' '
    return row
    
#Applying function
df = df.apply(binary_to_text, axis=1)

#Converting all columns except index column into string
df.iloc[:, range(1,len(df.columns))] = df.iloc[:, range(1,len(df.columns))].astype(str)
    
#Create column with all the features 

def combine_features(row):
    
    row['Combined_Features'] = ''
    
    for i in df.columns[1:]:
        
        if row[i] != ' ':
            row['Combined_Features'] = row['Combined_Features']+' '+row[i]+' '
    
    row = row[1:]
    return row

#Appliying combine features 
df = df.apply(combine_features, axis = 1) #Aplying function to all rows

###################### BUILDING RECOMMENDER SYSTEM ############################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vect = CountVectorizer() 

vect.fit(df['Combined_Features']) #training the model 

count_matrix = vect.transform(df['Combined_Features']) #transform series into frequency matrix

cosine_sim = cosine_similarity(count_matrix) #create similarity matrix

#Creating function to retrieve index from title
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]

#Creating function to retrieve title from index
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

#Function that retrieves the top 5 similar movies to the movie that the user likes

def movie_top_5_list(title):
    
    index = get_index_from_title(title)
    
    similar_movies = list(enumerate(cosine_sim[index]))
    
    sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
    
    list_movies = [get_title_from_index(i) for i in [j[0] for j in sorted_similar_movies][0:5]]
        
    for movie in list_movies:
        
        print(movie)
        
    return

####################### SPELLING RECOMMENDER ####################################
import nltk
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

###################### EXAMPLE 1: SPELL CHECKING ##############################

#Spelling check using Jaccard Distance Coefficient

entries = ['batman'] #Movie title input from the user
n_grams = 3 #set ngram for similarity calculation
jaccard_similarity(entries, n_grams) #retrieving 5 highest js movie titles

########## EXAMPLE 2: LOOKING FOR SIMILAR MOVIE THAT THE USER LIKES############
#Movie that user likes
movie_title = 'Batman Begins'

#Printing top 5 most similar movies that the user likes
movie_top_5_list(movie_title)

###################### EXAMPLE 3: USING BOTH FUNCTIONS #######################

movie_title = 'Batmaan'

entries = [movie_title] #movie titles list

n_grams = 3 #set ngram for similarity calculation

suggested_movie = jaccard_similarity(entries, n_grams)[0] #retrieving 1 highest js movie title

#Printing top 5 most similar movies that the user likes
movie_top_5_list(suggested_movie)














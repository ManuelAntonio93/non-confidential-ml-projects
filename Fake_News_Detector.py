# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 16:52:22 2020

@author: Manuel Antonio Noriega Ramos
"""
import pandas as pd
import numpy as np

df_fake = pd.read_csv('Fake.csv')

df_true = pd.read_csv('True.csv')

############################ EXPLORING DATA ##################################

#Checking for Nulls
df_fake.info()
df_true.info()

#Checking Average document length by Dataframe

def doc_len(row): #Define document length function
    
    return len(row)
    
np.mean(df_fake['text'].apply(doc_len)) #Avg document length fake news
np.mean(df_true['text'].apply(doc_len)) #Avg document length true news

#Checkin Average of digits per documentes in both dataframes
def count_digits(row):
        
    return  sum(c.isdigit() for c in row)

np.mean(df_fake['text'].apply(count_digits)) #Avg document digits fake news
np.mean(df_true['text'].apply(count_digits)) #Avg document digits true news

#Checking average non character per document in both dataframes
import re
    
def count_nonchar(row):
    
    return len(re.sub("[\w]", "", row))

np.mean(df_fake['text'].apply(count_nonchar)) #Avg document non char fake news
np.mean(df_true['text'].apply(count_nonchar)) #Avg document non char true news

############################ PREPARING DATA ##################################

#Create target column, fake = 1 and true = 0
df_fake['target'] = 1
df_true['target'] = 0

#Concatenate both dataframes by text and target columns
df = pd.concat([df_fake.loc[:, ['text', 'target']], df_true.loc[:, ['text', 'target']]])

#Shuffle Dataframe rows to avoid biased sampling
df = df.sample(frac=1, random_state = 0).reset_index(drop=True)

########################## MODEL SELECTION ##############################

from sklearn.model_selection import train_test_split

#Splitting data with 0.25 test size data
X_train, X_test, y_train, y_test = train_test_split(df['text'], 
                                                    df['target'], 
                                                    random_state=0)

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(min_df=3, max_df=0.70, ngram_range=(2,5)).fit(X_train)

X_train_vectorized = vect.transform(X_train)
    
X_test_vectorized = vect.transform(X_test)
  

#MULTINOMIAL NAIVE BAYES

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

MNB = MultinomialNB(alpha= 0.1)
    
MNB.fit(X_train_vectorized, y_train) #Training the model 
    
predictions_MNB = MNB.predict(X_test_vectorized) #Predicting values 
    
auc_score_MNB = roc_auc_score(y_test, predictions_MNB) #Calculating auc score

MNB_score = MNB.score(X_test_vectorized, y_test) #Calculating accuracy score

MNB_mean = np.mean(cross_val_score(MNB, X_train_vectorized, y_train, cv=5, scoring = 'accuracy'))#Cross-Validation EMC Score (5 Folds)

#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100, max_iter=4000).fit(X_train_vectorized, y_train) #Training the model
    
predictions_lr = lr.predict(X_test_vectorized) #Predicting values 
    
auc_score_lr = roc_auc_score(y_test, predictions_lr) #Calculating auc score

lr_score = lr.score(X_test_vectorized, y_test) #Calculating accuracy score

lr_mean = np.mean(cross_val_score(lr, X_train_vectorized, y_train, cv=5, scoring = 'accuracy'))#Cross-Validation EMC Score (5 Folds)


########################## MODEL UNDERSTANDING ################################

#get all the tokens from the vect model
feature_names = np.array(vect.get_feature_names())

# Sort the coefficients from the model LINEAR REGRESSION
sorted_coef_index = lr.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

# Sort the coefficients from the model MULTINOMIAL NAIVES BAYES
sorted_coef_index = MNB.coef_[0].argsort()

# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))





























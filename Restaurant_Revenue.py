# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 10:42:38 2020

@author: Manuel Antonio Noriega Ramos
"""
#Preparing Enviroment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
#Procesing Data Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
#Importing modules for Machine Learning Regression Modeling
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('train.csv')

#################### CLEANING AND EXPLORING THE DATA ##########################

df.info() # Data types, checking null values

df.describe() # Stat Summary of the Data

#Checking nans counts 
missing_count = df.isnull().sum().sort_values(ascending=False).apply(lambda x: x/len(df))

#Dropping all columns with more than 20% of missing values
for i in df.columns:
    if (df[i].isnull().sum()/len(df)) > 0.2:
        
        df.drop(labels=i, axis=1, inplace=True)

#Dropping irrelevant columns
df.drop('Id', axis=1, inplace=True)

#Getting categorical column names
cat_columns = df.select_dtypes(include=['object']).columns
        
#Define function to preprocess categorical data

def encode(data):
    '''function to encode non-null data and replace it in the original data'''
    encoder = OrdinalEncoder()
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    return data

#Applying function each categorical column 
df_enc = df.copy().loc[:, cat_columns].apply(encode, axis=0)

#Merging non encoded columns with the encoded columns
df_enc = df_enc.merge(df.drop(cat_columns, axis=1), left_index= True, right_index= True).loc[:, df.columns]

#Defining a KNN imputer to treat missing Data 
imputer = KNNImputer()
# fit on the dataset
imputer.fit(df_enc)
# transform the dataset
df_trans = pd.DataFrame(imputer.transform(df_enc), columns=df_enc.columns)

########################## FEATURE SELCCTION ##################################

'''Applying feature selection to see if reducing the dimensionality 
of the data improves the model performance'''
# Chi feature selection for categorical input and categorical output
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from scipy.stats import pearsonr

'''Reducing dimentionality by just taking into account the most relevant features 
according to their correlation to the target value, an analysis that will be done
for numerical variables and categorical variables respectively'''

#Creating copy df for feature selecction
df_fs = df_trans

#MOST RELEVANT CATEGORICAL FEATURES 
'''Analysing most relevant categorical features that contributes the most 
to the target variable using ANOVA test'''

#Selecting K categorical features 

cat_features = cat_columns #categorical variable names
df_fs_cat = df_fs.loc[:,cat_features] #creating df with only categorical features
fs = SelectKBest(score_func=f_classif, k=10) #creating feature selector
fs.fit(df_fs_cat, df_fs['SalePrice']) #training feature selector
cols = fs.get_support(indices=True) #Get columns to keep and create new dataframe with those only
cat_df = df_fs_cat.iloc[:,cols] #retrieving most relevant features


#MOST RELEVANT NUMERICAL FEATURES

'''Analysing most relevant numerical features that contributes the most 
to the target variable using f_regression feature analysis
for numeric input and numerical output'''

#Selecting numerical columns
num_col = df_fs.drop(cat_columns, axis=1).columns #numerical variable names
df_fs_num = df_fs.loc[:, num_col] #creating df with only numerical features
fs = SelectKBest(score_func=f_regression, k=11) #creating feature selector
fs.fit(df_fs_num, df_fs['SalePrice']) #training feature selector
cols = fs.get_support(indices=True) #Get columns to keep and create new dataframe with those only
num_df = df_fs_num.iloc[:,cols] #retrieving most relevant features


#Merging the two dataframes 

df_fs = cat_df.merge(num_df, left_index= True, right_index= True)

#CATEGORICAL CORRELATION ANALYSIS BETWEEN FEATURES
'''After selecting the 20 most relevant categorical and numerical features let's see how
correlated are these features between each other depending of they category'''
#Define function for categorical correlation analysis
def cat_analysis(df, cat_col):
    
    catdf = df.loc[:, cat_col]
    catdf = catdf.astype('int')
    cat_dict={}
    for i in range(0,len(catdf.columns)):
        X=catdf.drop(catdf.columns[i], axis=1)
        y=catdf.iloc[:,i]
        chi_scores = chi2(X,y)
        p_values = pd.Series(chi_scores[1],index = X.columns)
        print("For",catdf.columns[i])
        print(p_values)
        cat_dict['{}'.format(catdf.columns[i])] = p_values
        for j in range (0, len(p_values)):
            if (p_values[j]<0.05):
                print('Dependent Variable {} Value:'.format(X.columns[j]))
                print(p_values[j],'\n')

    df = pd.DataFrame.from_dict(cat_dict)
    
    return df

#Displaying P-Values from Chi square test 
df_cat_analysis = cat_analysis(df_fs, cat_df.columns)

'''After analysing the diferent results it can be concluded 
the variables have a high correlation between and that is something 
that it has to be taken into consideration when choosing a model'''

#NUMERICAL CORRELATION ANALYSIS BETWEEN FEATURES

#Define function for numerical correlation analysis
def num_analysis(df, num_col):
    
    numdf = df.loc[:, num_col]
    num_dict = {}
    for i in range(0,len(numdf.columns)-1):
        for j in range(i+1, len(numdf.columns)):
        
            X = numdf.loc[:,numdf.columns[i]]
            y = numdf.loc[:,numdf.columns[j]]
            pearson_score = pearsonr(X,y)
            num_dict['{}_vs_{}'.format(numdf.columns[i], numdf.columns[j])] = pearson_score
            print('Pearson Coefficient between {} and {} is: {}'.format(numdf.columns[i], 
                                                                numdf.columns[j], pearson_score))
    df = pd.DataFrame.from_dict(num_dict).T
    df.columns = ['R_score', 'p-value']
    df.sort_values(by='R_score', ascending=False, inplace=True)
    return df

#Displaying Pearson Coefficient and P-Values
df_num_analysis = num_analysis(df_fs, num_df.columns)

'''After analysing the diferent results it can be concluded the 
variables have a high correlation between and that is something that 
it has to be taken into consideration when choosing a model'''

#CATEGORICAL VS NUMERICAL ANALYSIS BETWEEN FEATURES

def cat_vs_num(df, cat_col, num_col):
    
    numdf = df.loc[:, num_col]
    catdf = df.loc[:, cat_col]
    cat_num_dict = {}
    
    for i in catdf.columns:
        
        f_scores = f_classif(numdf, catdf[i])
        p_values = pd.Series(f_scores[1],index = numdf.columns)
        cat_num_dict['{}'.format(i)] = p_values
    df = pd.DataFrame.from_dict(cat_num_dict)
    
    return df

cat_num_dict = cat_vs_num(df_fs, cat_df.columns, num_df.columns)

'''After analysing the diferent results it can be concluded the 
variables have a high correlation between and that is something that 
it has to be taken into consideration when choosing a model'''

#Defining Features and Target variables
X = df_fs.drop('SalePrice', axis = 1)
y = df_fs.loc[:, 'SalePrice']

#Normalizing Data with MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#Splitting Data into trainig and test Datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size= 0.2)

########################### MODEL SELECTION ###################################

#LINEAR REGRESSION

from sklearn.linear_model import LinearRegression

LR = LinearRegression().fit(X_train, y_train)#trainig the model

LR_score = LR.score(X_test, y_test)#R2 score

LR_mean_score = np.mean(cross_val_score(LR, X, y, cv=5, scoring = 'r2'))#Cross-Validation LR R2 Score (5 Folds)

#POLYNOMIAL REGRESSION

from sklearn.preprocessing import PolynomialFeatures 

degrees = [2,3,4] #list of degrees for evaluation
poly_scores =[] #Derfine list to store score values

#Evaluating each degree to find the one that produce the best score
for i in degrees:
    
    poly_features =  PolynomialFeatures(degree=i)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.fit_transform(X_test)
    PR = LinearRegression().fit(X_train_poly, y_train)
    poly_scores.append(PR.score(X_test_poly, y_test))

#Creating Series to compare scores by degree
pd.Series(poly_scores, index=degrees).sort_values(ascending=False)

#Creting model with the best degree for polynomial regression
poly_features =  PolynomialFeatures(degree=5)#Setting model parameters
X_train_poly = poly_features.fit_transform(X_train)#transforming train features into polynomial features
X_test_poly = poly_features.fit_transform(X_test)#transforming test features into polynomial features

PR = LinearRegression().fit(X_train_poly, y_train)#Training model

PR_score = PR.score(X_test_poly, y_test)#R2 score

X_poly = poly_features.fit_transform(X)#transforming features into polynomial features

PR_mean_score = np.mean(cross_val_score(PR, X_poly, y, cv=5, scoring = 'r2'))#Cross-Validation PR R2 Score (5 Folds)

#RIDGE REGRESSION

from sklearn.linear_model import Ridge

'''Assesing different values for the main hyperparameters for the Ridge Regressor,
the following approach is to avoid over demand of machine power
testing just one hyperparameter at a time with different values'''
'''
#Creating list of hyperparameter dictionaries
grid_list_ridge = [{'alpha': [1, 10, 15, 20, 50, 100, 150]},
                    {'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']}]

#Creating list for the best parameters
best_params_list_ridge = []

#Loop to get the best hyperparameters
for i in grid_list_ridge:

    ridge = Ridge()
    grid_ridge = GridSearchCV(ridge, param_grid= i, scoring='r2', cv= 10)
    grid_ridge.fit(X_train, y_train)
    best_params_list_ridge.append(grid_ridge.best_params_)
'''
'''After knowing the best hyperparameters, I tested all the combinations and 
the best results for the cross-validation r2 score were obtained using
alpha=15, solver='lsqr' '''

#Define model after hyperparameter evaluation
ridge = Ridge(alpha=15, solver='lsqr')#define model
ridge.fit(X_train, y_train)#trainig model

ridge_score = ridge.score(X_test, y_test)#R2 Score

ridge_mean_score = np.mean(cross_val_score(ridge, X, y, cv=5, scoring = 'r2'))#Cross-Validation Ridge R2 Score (5 Folds)

#LASSO REGRESSION

from sklearn.linear_model import Lasso

'''Assesing different values for the main hyperparameters for the Lasso Regressor,
the following approach is to avoid over demand of machine power
testing just one hyperparameter at a time with different values'''

#Define possible hyperparameter values for the model
grid_values_alpha = {'alpha': [1, 10, 15, 20, 50, 100, 150]} # alpha values
lass = Lasso()#Define model
lasso = GridSearchCV(lass, param_grid= grid_values_alpha, scoring='r2', cv= 10)#selecting best alpha for model
lasso.fit(X_train, y_train)#training model
lasso.best_params_#best alpha

lasso_score = lasso.score(X_test, y_test)#R2 Score
lasso_mean_score = np.mean(cross_val_score(lasso, X, y, cv=5, scoring = 'r2'))#Cross-Validation Lasso R2 Score (5 Folds)

#ELASTICNET REGRESSION

from sklearn.linear_model import ElasticNet

'''Assesing different values for the main hyperparameters for the ElasticNet Regressor,
the following approach is to avoid over demand of machine power
testing just one hyperparameter at a time with different values'''
'''
#Creating list of hyperparameter dictionaries
grid_list_elasticnet = [{'alpha': [1, 10, 15, 20, 25, 30, 50, 100, 150]},
                        {'l1_ratio': [0,0.25,0.5,0.75,0.90,1]}]

#Creating list for the best parameters
best_params_list_elasticnet = []

#Loop to get the best hyperparameters
for i in grid_list_elasticnet:

    elasticnet = ElasticNet(max_iter=100000)
    grid_elasticnet = GridSearchCV(elasticnet, param_grid= i, scoring='r2', cv= 10)
    grid_elasticnet.fit(X_train, y_train)
    best_params_list_elasticnet.append(grid_elasticnet.best_params_)
'''
'''After knowing the best hyperparameters, I tested all the combinations and 
the best results for the cross-validation r2 score were obtained using 
kalpha=50,l1_ratio=1'''

elasticnet = ElasticNet(alpha=50,l1_ratio=1)#Setting model parameters

#Tuning best combinations of hyperparameters
'''
grid_elasticnet = GridSearchCV(elasticnet, param_grid= grid_list_elasticnet[], scoring='r2', cv= 10)
grid_elasticnet.best_params_

'''
elasticnet.fit(X_train, y_train)#Training the model

elasticnet_score = elasticnet.score(X_test,y_test)#ElasticNet R2 Score

elasticnet_mean_score = np.mean(cross_val_score(elasticnet, X, y, cv=5, scoring = 'r2'))#Cross-Validation ElasticNet R2 Score (5 Folds)

#SUPPORT VECTOR MACHINE REGRESSION SVR

from sklearn.svm import SVR

'''Assesing different values for the main hyperparameters for the SVR Regressor,
the following approach is to avoid over demand of machine power
testing just one hyperparameter at a time with different values'''
'''
#Create list of hyperparameter dictionaries
grid_list_SVR = [{'C': [1, 10, 50, 100, 500, 1000, 1500, 2000, 3000, 4000]},
                 {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                 {'gamma': ['scale', 'auto']},
                 {'coef0': [0,0.1,0.5,1,5,10]},
                 {'epsilon': [0.1,0.25,0.5,1,5,10]},
                 {'degree': [2,3,4,5]}]

#Creating list for the best parameters
best_params_list_svr = []

#Loop to get the best hyperparameters
for i in grid_list_SVR:

    svr = SVR()
    grid_svr = GridSearchCV(svr, param_grid= i, scoring='r2', cv= 10)
    grid_svr.fit(X_train, y_train)
    best_params_list_svr.append(grid_svr.best_params_)
'''
'''After knowing the best hyperparameters, I tested all the combinations and 
the best results for the cross-validation r2 score were obtained using 
kernel='poly', C=2000, degree=2, gamma='scale', epsilon=0.1, coef0=0.0'''

svr = SVR(kernel='poly', C=2000, degree=2, gamma='scale', epsilon=0.1, coef0=0.0)

#Tuning best combinations of hyperparameters
'''
grid_svr = GridSearchCV(svr, param_grid= grid_list_svr[], scoring='r2', cv= 10)
grid_svr.best_params_

'''
svr.fit(X_train, y_train)
svr_score = svr.score(X_test, y_test) #R2 score
svr_mean_score = np.mean(cross_val_score(svr, X, y, cv=5, scoring = 'r2'))#Cross-Validation svr Score (5 Folds)

#RANDOM FOREST REGRESSOR

from sklearn.ensemble import RandomForestRegressor

'''Assesing different values for the main hyperparameters for the Random Forest Regressor,
the following approach is to avoid over demand of machine power
testing just one hyperparameter at a time with different values'''
'''
#Define possible hyperparameter values for the model
grid_list_RFR = [{'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]},
                 {'max_features': [None, 'auto', 'sqrt']},
                 {'min_samples_leaf': [1, 2, 4]},
                 {'min_samples_split': [2, 5, 10]},
                 {'n_estimators': [100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}]

#Creating list for the best parameters
best_params_list_RFR = []

#Loop to get the best hyperparameters
for i in grid_list_RFR:

    RFR = RandomForestRegressor(random_state=1)
    grid_RFR = GridSearchCV(RFR, param_grid= i, scoring='r2', cv= 10)
    grid_RFR.fit(X_train, y_train)
    best_params_list_RFR.append(grid_RFR.best_params_)
'''

'''After knowing the best hyperparameters, I tested all the combinations and 
the best results for the cross-validation r2 score were obtained using 
random_state=1, max_depth=20, max_features='sqrt', min_samples_leaf=1, 
min_samples_split=2, n_estimators=300'''

#Setting model hyperparameters
RFR = RandomForestRegressor(random_state=1, max_depth=20, max_features='sqrt',
                            min_samples_leaf=1, min_samples_split=2, n_estimators=300)

#Tuning best combinations of hyperparameters
'''
grid_RFR = GridSearchCV(RFR, param_grid= grid_list_RFR[], scoring='r2', cv= 10)
grid_RFR.fit(X_train, y_train)
grid_RFR.best_params_

'''
RFR.fit(X_train, y_train)#trainig model
RFR_score = RFR.score(X_test, y_test)#RFR R2 score
RFR_mean_score = np.mean(cross_val_score(RFR, X, y, cv=5, scoring = 'r2'))#Cross-Validation RFR Score (5 Folds)

#GRADIENT BOOSTING REGRESSOR

from sklearn.ensemble import GradientBoostingRegressor

'''Assesing different values for the main hyperparameters for the Gradient Boosting Regressor,
the following approach is to avoid over demand of machine power
testing just one hyperparameter at a time with different values'''
'''
#Creating list of hyperparameter dictionaries
grid_list_GBR = [{'loss': ['ls', 'lad', 'huber', 'quantile']},
                 {'learning_rate': [0.001,0.01,0.1,0.25,0.5,0.1]},
                 {'n_estimators':[100,150,200,250,500,1000,1500,2000]},
                 {'min_samples_split': [2,3,4,5]},
                 {'min_samples_leaf': [1,2,3,4,5]},
                 {'max_features': [None,'auto', 'sqrt', 'log2']},
                 {'max_depth': [3,4,5]}]

#Creating list for the best parameters
best_params_list_GBR = []

#Loop to get the best hyperparameters
for i in grid_list_GBR:

    GBR = GradientBoostingRegressor(random_state=1)
    grid_GBR = GridSearchCV(GBR, param_grid= i, scoring='r2', cv= 10)
    grid_GBR.fit(X_train, y_train)
    best_params_list_GBR.append(grid_GBR.best_params_)
'''
'''After knowing the best hyperparameters, I tested all the combinations and 
the best results for the cross-validation r2 score were obtained using 
kernel='poly', C=2000, degree=2, gamma='scale', epsilon=0.1, coef0=0.0'''

#setting model hyperparameters
GBR = GradientBoostingRegressor(random_state=1, loss='ls', n_estimators=100, learning_rate=0.15,
                                min_samples_split=3, min_samples_leaf=1, max_features=None, max_depth=3)

#Tuning best combinations of hyperparameters
'''
grid_GBR = GridSearchCV(GBR, param_grid= grid_list_GBR[], scoring='r2', cv= 10)
grid_GBR.fit(X_train, y_train)
grid_GBR.best_params_

'''

GBR.fit(X_train, y_train)#trainig model
GBR_score = GBR.score(X_test, y_test)#GBR R2 score
GBR_mean_score = np.mean(cross_val_score(GBR, X, y, cv=5, scoring = 'r2'))#Cross-Validation GBR Score (5 Folds)

#XGB REGRESSOR

import xgboost as xgb

'''Assesing different values for the main hyperparameters for the XGB Regressor,
the following approach is to avoid over demand of machine power
testing just one hyperparameter at a time with different values'''
'''
grid_list_XGBR = [{'learning_rate':[None, 0.001, 0.01, 0.05, 0.1, 0.2]},
                  {'max_depth':[1,2,3,4,5]},
                  {'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]},
                  {'gamma': [None, 0, 0.1, 0.2, 0.3, 0.4, 0.5]},
                  {'min_child_weight': [None, 1,2,3,4,5,6]},
                  {'subsample':[None, 0.6,0.7,0.8,0.9]},
                  {'colsample_bytree':[None, 0.6,0.7,0.8,0.9]},
                  {'reg_alpha': [0, 1, 5, 10, 15, 20, 50, 100]}]

#Creating list for the best parameters

best_params_list_XGBR = []

#Loop to get the best hyperparameters
for i in grid_list_XGBR:

    XGBR = xgb.XGBRegressor(objective="reg:squarederror", random_state=1)
    grid_XGBR = GridSearchCV(XGBR, param_grid= i, scoring='r2', cv= 10)
    grid_XGBR.fit(X_train, y_train)
    best_params_list_XGBR.append(grid_XGBR.best_params_)
'''
'''After knowing the best hyperparameters, I tested all the combinations and 
the best results for the cross-validation r2 score were obtained using 
random_state=1, max_depth=20, max_features='sqrt', min_samples_leaf=1, 
min_samples_split=2, n_estimators=300'''

#Setting model hyperparameters
XGBR = xgb.XGBRegressor(objective="reg:squarederror", learning_rate= 0.08, max_depth=3, n_estimators=145,
                        min_child_weight=1, reg_alpha=40, random_state=1)

#Tuning best combinations of hyperparameters
'''
grid_XGBR = GridSearchCV(XGBR, param_grid= grid_list_XGBR[] , scoring='r2', cv= 10)
grid_XGBR.fit(X_train, y_train)
grid_XGBR.best_params_
'''
XGBR.fit(X_train, y_train)#trainig model
XGBR_score = XGBR.score(X_test, y_test)#XGBR R2 score
XGBR_mean_score = np.mean(cross_val_score(XGBR, X, y, cv=5, scoring = 'r2'))#Cross-Validation XGBR Score (5 Folds)

#LIGHTGBM REGRESSOR

import lightgbm as lgb

'''Assesing different values for the main hyperparameters for the LightGBM Regressor,
the following approach is to avoid over demand of machine power
testing just one hyperparameter at a time with different values'''
'''
grid_list_LGBMR = [{'learning_rate':[None, 0.001, 0.01, 0.05, 0.1, 0.2]},
                   {'num_leaves':[31, 40, 45, 50]},
                   {'max_depth':[1,2,3,4]},
                   {'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]},
                   {'min_child_weight': [None, 1,2,3,4,5,6]},
                   {'subsample':[None, 0.6,0.7,0.8,0.9]},
                   {'colsample_bytree':[None, 0.6,0.7,0.8,0.9]},
                   {'reg_alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

#Creating list for the best parameters

best_params_list_LGBMR = []

#Loop to get the best hyperparameters
for i in grid_list_LGBMR:

    LGBMR = lgb.LGBMRegressor(objective='regression', random_state=1)
    grid_LGBMR = GridSearchCV(LGBMR, param_grid= i, scoring='r2', cv= 10)
    grid_LGBMR.fit(X_train, y_train)
    best_params_list_LGBMR.append(grid_LGBMR.best_params_)
'''
'''After knowing the best hyperparameters, I tested all the combinations and 
the best results for the cross-validation r2 score were obtained using 
random_state=1, max_depth=20, max_features='sqrt', min_samples_leaf=1, 
min_samples_split=2, n_estimators=300'''

#Setting model hyperparameters
LGBMR = lgb.LGBMRegressor(objective='regression', max_depth=4, learning_rate = 0.05, 
                          n_estimators= 100, reg_alpha=0.7, random_state=1)

#Tuning best combinations of hyperparameters
'''
grid_LGBMR = GridSearchCV(LGBMR, param_grid= grid_list_LGBMR[1], scoring='r2', cv= 10)
grid_LGBMR.fit(X_train, y_train)
grid_LGBMR.best_params_
'''
LGBMR.fit(X_train, y_train)#trainig model
LGBMR_score = LGBMR.score(X_test, y_test)#LGBMR R2 score
LGBMR_mean_score = np.mean(cross_val_score(LGBMR, X, y, cv=5, scoring = 'r2'))#Cross-Validation LGBMR Score (5 Folds)

#NEURAL NETWORK REGRESSOR

from sklearn.neural_network import MLPRegressor

'''Assessing most significant hyperparameters of the neural network model'''
'''
grid_list_NNR = [{'learning_rate':['constant', 'invscaling', 'adaptive']},
                 {'solver':['lbfgs', 'sgd', 'adam']},
                 {'hidden_layer_sizes':[(10,), (50,), (100,), (150,)]},
                 {'alpha':[0.001, 0.01, 0.1, 0.005, 0.05, 0.5, 1.0, 5.0]}]

#Creating list for the best parameters

best_params_list_NNR = []

for i in grid_list_NNR:

    NNR = MLPRegressor(max_iter=2000)
    grid_NNR = GridSearchCV(NNR, param_grid= i, scoring='r2', cv= 10)
    grid_NNR.fit(X_train, y_train)
    best_params_list_NNR.append(grid_NNR.best_params_)
'''
'''After assesing different values the best hyperparameters were
learning_rate='adaptive', max_iter=2000, hidden_layer_sizes=(50,50), and the rest
of hyperparameters were left as the default values'''

NNR = MLPRegressor(max_iter=2000, learning_rate='invscaling', hidden_layer_sizes=(150,150,150), alpha=0.1)

#Tuning best combinations of hyperparameters
'''
grid_NNC = GridSearchCV(NNC, param_grid= grid_list_NNC[], scoring='roc_auc', cv= 10)
grid_NNC,fit(X_Train, y_train)
grid_NNC.best_params_

'''
NNR.fit(X_train, y_train) #Trainig Model
NNR_score = NNR.score(X_test, y_test) #Accuracy Score
NNR_mean_score = np.mean(cross_val_score(NNR, X, y, cv=5, scoring = 'r2'))#Cross-Validation NNC Score (5 Folds)

########################## COMPARING MODELS ###################################

#ENSEMBLE APPROACH
'''Considering the four models with the best R2 score I will calculate the average of the 
predictions produced by these models, in this case I'm not going to use weighted average because 
the performance of each model is not that different
'''
#Define function to calculate average or weighted average from regression models

def model_average(regression_models, test_data, weights=None, weighted=False):
    
    predictions_dict = {}
    
    if weighted==True and weights==None:
        
        raise Exception('For weighted average you must input a list of weights')
        
        return
    
    elif weighted==False and weights!=None:
        print('weighted paramater is false, all weigths from list \
              will be ignored, the result will be a normal averge')
    
    for i in range(len(regression_models)):
        
        predictions_dict['{}'.format(i)] = pd.Series(regression_models[i].predict(test_data))
    
    df_pred = pd.DataFrame.from_dict(predictions_dict, orient = 'index').T
    
    #Calculating weighted or not weighted average
    
    if weighted == True:
        
        if sum(weights) != 1:
            raise Exception('Sum of weights must be equal to 1')
            return
        
        else:
            
            for i in range(len(df_pred.columns)):
                
                df_pred.iloc[:,i]= df_pred.iloc[:,i]*weights[i]
            
            df_pred['Avg']= df_pred.apply(sum, axis=1)
            
            return df_pred['Avg']
    else:
        
        df_pred['Avg']= df_pred.apply(np.mean, axis=1)
        return df_pred['Avg']


#List of models to take into consideration
list_models = [RFR, GBR, XGBR, LGBMR]

#List of weights 
weights = [0.1,0.3,0.3,0.3]

#Calculate normal average of each model predictions
EMR_pred = model_average(list_models, X_test)#normal avergage 
EMR_pred_weighted = model_average(list_models, X_test, weights=weights, weighted=True)#weighted average

#EMR R2 and mse Score
EMR_score = r2_score(y_test, EMR_pred)
EMR_mse = mean_squared_error(y_test, EMR_pred)

#EMR R2 and mse Score
EMR_weighted_score = r2_score(y_test, EMR_pred_weighted)
EMR_weighted_mse = mean_squared_error(y_test, EMR_pred_weighted)

#Cross validation with EMR non-weighted model
EMR_r2_list = []

for i in range(5):
    
    X_train_EMR, X_test_EMR, y_train_EMR, y_test_EMR = train_test_split(X, y, random_state=i, test_size= 0.2)
    EMR_pred_cv = model_average(list_models, X_test_EMR)
    EMR_r2_list.append(r2_score(y_test_EMR, EMR_pred_cv))
    
#Cross-Validation R2 score 
EMR_mean_score = np.mean(EMR_r2_list)

#Cross validation with EMR non-weighted model
EMR_weighted_r2_list = []

for i in range(5):
    
    X_train_EMR, X_test_EMR, y_train_EMR, y_test_EMR = train_test_split(X, y, random_state=i, test_size= 0.2)
    EMR_weighted_pred_cv = model_average(list_models, X_test_EMR, weights=weights, weighted=True)
    EMR_weighted_r2_list.append(r2_score(y_test_EMR, EMR_weighted_pred_cv))

#Cross-Validation R2 score weighted
EMR_weighted_mean_score = np.mean(EMR_weighted_r2_list) 

########################## COMPARING MODELS ###################################

#Creating a Series from the different results of each model (Accuracy Scores and CrossValidation accuracy scores)

s1 = pd.Series([LR_score, PR_score, ridge_score, lasso_score, elasticnet_score, 
                svr_score, RFR_score, GBR_score, XGBR_score, LGBMR_score, NNR_score,
                EMR_score, EMR_weighted_score], name = 'R2_Score')

s2 = pd.Series([LR_mean_score, PR_mean_score, ridge_mean_score, lasso_mean_score, 
                elasticnet_mean_score, svr_mean_score, RFR_mean_score, GBR_mean_score, 
                XGBR_mean_score, LGBMR_mean_score, NNR_mean_score,
                EMR_mean_score, EMR_weighted_mean_score], name = 'R2_Mean')

models_df = pd.DataFrame([s1,s2])
models_df.columns = ['LR', 'PR', 'ridge', 'lasso', 'elasticnet', 'svr', 'RFR', 
                     'GBR', 'XGBR', 'LGBMR', 'NNR', 'EMR', 'EMR']

#Highest Accuracy Score Model(s)
models_df.T[models_df.T['R2_Score'] == max(models_df.T['R2_Score'])]

#Highest Cross-Validation Accuracy Score Model(s)
models_df.T[models_df.T['R2_Mean'] == max(models_df.T['R2_Mean'])]











































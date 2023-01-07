# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:09:56 2020

@author: Manuel Antonio Noriega Ramos
"""
#PREPARING THE ENVIROMENT

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

#Importing modules for Machine Learning Classification Modeling
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score, roc_auc_score
#Retrieving Heart Desease Data

df = pd.read_csv('datasets_228_482_diabetes.csv')

####################### EXPLORING THE DATA ####################################

df.info() # Data types, checking null values

df.describe() # Stat Summary of the Data

#Cheking average Values of the features by Outcome
df.groupby('Outcome').mean() 

'''Cheking if the data is balanced. Data is balanced if the 
prpoportion of the target values is at least (60%-40%)'''

df['Outcome'].value_counts()
df['Outcome'].value_counts()/len(df)

'''The Data is unbalanced, so the approach with this Data is to achieve the
best roc_auc_score in contraposition to just assessing accuracy'''

#DATA VISUALIZATION 

#Bining Age Column in 12 bins
Ages = pd.cut(df['Age'], bins= 10)

#Heart Disease Frequency for Ages
pd.crosstab(Ages,df.Outcome).plot(kind="bar",figsize=(20,6))
plt.title('Diabetes Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

#Defining Features and Target variables
X = df.drop('Outcome', axis = 1)
y = df.loc[:, 'Outcome']

#Normalizing Data with MinMaxScaler
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#Splitting Data into trainig and test Datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size= 0.2)

########################### MODEL SELECCTION ##################################

#SVC CLASSIFIER

from sklearn.svm import SVC

'''Assesing different values for the main hyperparameters for the SVC Classifier,
the following approach is to avoid over demand of machine power
testing just one hyperparameter at a time with different values'''

#Comenting code for Hyperparameter Tunning

grid_values_C = {'C': [1, 10, 50, 100, 500, 1000, 1500, 2000]} # C values
grid_values_k = {'kernel': ['linear', 'poly', 'rbf']} # Kernel values
grid_values_gamma = {'gamma': ['scale', 'auto']} #Gamma values

grid_list_SVC = [grid_values_C, grid_values_k, grid_values_gamma]

#Creating list for the best parameters
best_params_list_svc = []

for i in grid_list_SVC:

    svc = SVC(random_state=1)
    grid_svc = GridSearchCV(svc, param_grid= i, scoring='roc_auc', cv= 10)
    grid_svc.fit(X_train, y_train)
    best_params_list_svc.append(grid_svc.best_params_)


'''After knowing the best hyperparameters, I tested all the combinations and 
the best results for the accuracy score were obtained using only kernel= rbf'''

svc = SVC(kernel='rbf')

#Tuning best combinations of hyperparameters
'''
grid_svc = GridSearchCV(svc, param_grid= grid_list_svc[], scoring='roc_auc', cv= 10)
grid_svc.best_params_

'''
svc.fit(X_train, y_train) #Trainig Model

svc_score = svc.score(X_test, y_test) #Accuracy Score

svc_mean = np.mean(cross_val_score(svc, X, y, cv=5, scoring = 'accuracy'))#Cross-Validation svc Score (5 Folds)

svc_recall = np.mean(cross_val_score(svc, X, y, cv=5, scoring = 'recall'))#Cross-Validation svc Recall (5 Folds)

svc_roc_auc = np.mean(cross_val_score(svc, X, y, cv=5, scoring = 'roc_auc'))#Cross-Validation svc Roc_Auc (5 Folds)

#LOGISTIC REGRESSION 

from sklearn.linear_model import LogisticRegression

'''Assesing different values for the main hyperparameters for Logistic Regression,
testing and assesing the hyperparameters C and solver'''

#Comenting code for Hyperparameter Tunning

grid_values_lr = [{'C': [1,10,100,1000,2000]}, {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]

best_params_list_lr = [] #Creating empty list to 

for i in grid_values_lr:

    lr = LogisticRegression(max_iter=4000)

    grid_lr = GridSearchCV(lr, param_grid= i, scoring= 'roc_auc', cv= 10)

    grid_lr.fit(X_train, y_train)

    best_params_list_lr.append(grid_lr.best_params_)


'''After knowing the best hyperparameters, I tested all the combinations and 
the best results for the accuracy score were obtained using 
only kernel= newton-cg and default C = 1.0'''

lr = LogisticRegression(solver='newton-cg', C=10, max_iter=4000)

#Tuning best combinations of hyperparameters
'''
grid_lr = GridSearchCV(lr, param_grid= grid_list_lr[], scoring='roc_auc', cv= 10)
grid_lr.best_params_

'''

lr.fit(X_train, y_train) #Trainig Model

lr_score = lr.score(X_test, y_test) #Accuracy Score

lr_mean = np.mean(cross_val_score(lr, X, y, cv=5, scoring = 'accuracy'))#Cross-Validation lr Score (5 Folds)

lr_recall = np.mean(cross_val_score(lr, X, y, cv=5, scoring = 'recall'))#Cross-Validation lr Recall(5 Folds)

lr_roc_auc =  np.mean(cross_val_score(lr, X, y, cv=5, scoring = 'roc_auc'))#Cross-Validation lr Roc_Auc (5 Folds)

#NAIVE BAYES

'''Because the simplicity of this model I will use the dafault hyperparameters values'''

from sklearn.naive_bayes import GaussianNB

GNB = GaussianNB()

GNB.fit(X_train, y_train) #Trainig Model

GNB_score = GNB.score(X_test, y_test) #Accuracy Score

GNB_mean = np.mean(cross_val_score(GNB, X_train, y_train, cv=5, scoring = 'accuracy'))#Cross-Validation GNB Score (5 Folds)

GNB_recall = np.mean(cross_val_score(GNB, X_train, y_train, cv=5, scoring = 'recall'))#Cross-Validation GNB Mean (5 Folds)

GNB_roc_auc = np.mean(cross_val_score(GNB, X_train, y_train, cv=5, scoring = 'roc_auc'))#Cross-Validation GNB Roc_Auc (5 Folds)

#GRADIENT BOOSTING CLASSIFIER 

from sklearn.ensemble import GradientBoostingClassifier

'''Assesing GBC hyperperparameters one at a time to reduce machine power'''

grid_values_GBC = [{'max_depth': [3,4,5,6,7,8,9,10]},
                   {'max_features': ['auto', 'sqrt']},
                   {'min_samples_leaf': [1, 2, 4]},
                   {'min_samples_split': [2, 5, 10]},
                   {'n_estimators': [50, 100, 500, 1000, 1500, 2000]},
                   {'learning_rate':[0.001,0.005,0.01, 0.1]}]

#Creating list for the best parameters
best_params_list_GBC = []

for i in grid_values_GBC:

    GBC = GradientBoostingClassifier(random_state=1)
    grid_GBC = GridSearchCV(GBC, param_grid= i, scoring='roc_auc', cv= 10)
    grid_GBC.fit(X_train, y_train)
    best_params_list_GBC.append(grid_GBC.best_params_)


'''After assesing the diferent combinations I conclude that the best way to tune this model for the best
roc_auc score is setting the hyperparameters as n_estimators= 100, learning_rate=0.01, max_depth=3, 
min_samples_split=2, max_features='sqrt', min_samples_leaf=1, with the rest of hyperparameters set 
at their default values'''

GBC = GradientBoostingClassifier(n_estimators= 100, learning_rate=0.01, max_depth=3, 
                                 min_samples_split=2, max_features='sqrt', min_samples_leaf=1)

#Tuning best combinations of hyperparameters
'''
grid_GBC = GridSearchCV(GBC, param_grid= grid_list_GBC[4], scoring='roc_auc', cv= 10)
grid_GBC.best_params_

'''

GBC.fit(X_train, y_train) #Trainig Model

GBC_score = GBC.score(X_test, y_test) #Accuracy Score

GBC_mean = np.mean(cross_val_score(GBC, X, y, cv=5, scoring = 'accuracy'))#Cross-Validation GBC Score (5 Folds)

GBC_recall = np.mean(cross_val_score(GBC, X, y, cv=5, scoring = 'recall'))#Cross-Validation GBC Recall (5 Folds)

GBC_roc_auc = np.mean(cross_val_score(GBC, X, y, cv=5, scoring = 'roc_auc'))#Cross-Validation GBC Roc_Auc (5 Folds)

#KNN CLASIFFIER

from sklearn.neighbors import KNeighborsClassifier

'''Because of the simplicity of the model I will just evaluate the n_neighbors hyperparameter'''


grid_values_knn = {'n_neighbors': [5,6,7,8,9,10,11,12,13,14,15,16]}

KNN = KNeighborsClassifier()

grid_KNN = GridSearchCV(KNN, param_grid=grid_values_knn, scoring = 'roc_auc')

grid_KNN.fit(X_train, y_train)

grid_KNN.best_params_


''' After the evaluation I came to the conclussion that the best results are obtained after setting
the hyperparameter n_neighbor = 8'''

KNN = KNeighborsClassifier(n_neighbors= 12)

KNN.fit(X_train, y_train) #Trainig Model

KNN_score = KNN.score(X_test, y_test) #Accuracy Score

KNN_mean = np.mean(cross_val_score(KNN, X, y, cv=5, scoring = 'accuracy'))#Cross-Validation KNN Score (5 Folds)

KNN_recall = np.mean(cross_val_score(KNN, X, y, cv=5, scoring = 'recall'))#Cross-Validation KNN Mean (5 Folds)

KNN_roc_auc = np.mean(cross_val_score(KNN, X, y, cv=5, scoring = 'roc_auc'))#Cross-Validation KNN Roc_Auc (5 Folds)

#RANDOM FOREST CLASSIFIER

from sklearn.ensemble import RandomForestClassifier

'''Assessing most significant hyperparameters of the RFC'''

grid_list_RFC = [{'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]},
                 {'max_features': ['auto', 'sqrt']},
                 {'min_samples_leaf': [1, 2, 4]},
                 {'min_samples_split': [2, 5, 10]},
                 {'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}]


#Creating list for the best parameters

best_params_list_RFC = []

for i in grid_list_RFC:

    RFC = RandomForestClassifier(random_state = 1)
    grid_RFC = GridSearchCV(RFC, param_grid= i, scoring='roc_auc', cv= 10)
    grid_RFC.fit(X_train, y_train)
    best_params_list_RFC.append(grid_RFC.best_params_)

'''After the evaluation the best roc_auc score is obtained with the hyperparameters
max_depth=20 and min_samples_leaf=2'''

RFC = RandomForestClassifier(max_depth=20, random_state=1, min_samples_leaf=2)

#Tuning best combinations of hyperparameters
'''
grid_RFC = GridSearchCV(RFC, param_grid= grid_list_RFC[4], scoring='roc_auc', cv= 10)
grid_RFC.best_params_

'''
RFC.fit(X_train, y_train) #Trainig Model

RFC_score = RFC.score(X_test, y_test) #Accuracy Score

RFC_mean = np.mean(cross_val_score(RFC, X, y, cv=5, scoring = 'accuracy'))#Cross-Validation RFC Score (5 Folds)

RFC_recall = np.mean(cross_val_score(RFC, X, y, cv=5, scoring = 'recall'))#Cross-Validation RFC Recall (5 Folds)

RFC_roc_auc = np.mean(cross_val_score(RFC, X, y, cv=5, scoring = 'roc_auc'))#Cross-Validation RFC Roc_Auc (5 Folds)

#XGBOOST CLASSIFIER

import xgboost as xgb

'''Assessing most significant hyperparameters of the xgb'''

grid_list_XGB = [{'learning_rate':[0.001,0.005,0.01,0.1]},
                 {'max_depth':[1,2,3,4,5]},
                 {'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}]

#Creating list for the best parameters

best_params_list_XGB = []

for i in grid_list_XGB:

    XGB = xgb.XGBClassifier(objective="binary:logistic", random_state=1)
    grid_XGB = GridSearchCV(XGB, param_grid= i, scoring='roc_auc', cv= 10)
    grid_XGB.fit(X_train, y_train)
    best_params_list_XGB.append(grid_XGB.best_params_)

'''After the evaluation the best roc_auc score is obtained with the hyperparameters
n_estimator = 400'''

XGB = xgb.XGBClassifier(objective="binary:logistic", random_state=1, n_estimators = 400)

#Tuning best combinations of hyperparameters
'''
grid_XGB = GridSearchCV(XGB, param_grid= grid_list_XGB[4], scoring='roc_auc', cv= 10)
grid_XGB.best_params_

'''
XGB.fit(X_train, y_train) #Trainig Model

XGB_score = XGB.score(X_test, y_test) #Accuracy Score

XGB_mean = np.mean(cross_val_score(XGB, X, y, cv=5, scoring = 'accuracy'))#Cross-Validation XGB Score (5 Folds)

XGB_recall = np.mean(cross_val_score(XGB, X, y, cv=5, scoring = 'recall'))#Cross-Validation XGB Recall (5 Folds)

XGB_roc_auc = np.mean(cross_val_score(XGB, X, y, cv=5, scoring = 'roc_auc'))#Cross-Validation XGB Roc_Auc (5 Folds)

#NEURAL NETWORK CLASSIFIER
from sklearn.neural_network import MLPClassifier

'''Assessing most significant hyperparameters of the neural network model'''

grid_list_NNC = [{'learning_rate':['constant', 'invscaling', 'adaptive']},
                 {'solver':['lbfgs', 'sgd', 'adam']},
                 {'hidden_layer_sizes':[(10,), (50,), (100,), (150,)]},
                 {'alpha':[0.001, 0.01, 0.1, 0.005, 0.05, 0.5, 1.0, 5.0]}]

#Creating list for the best parameters

best_params_list_NNC = []

for i in grid_list_NNC:

    NNC = MLPClassifier(max_iter=2000)
    grid_NNC = GridSearchCV(NNC, param_grid= i, scoring='roc_auc', cv= 10)
    grid_NNC.fit(X_train, y_train)
    best_params_list_NNC.append(grid_NNC.best_params_)

'''After assesing different values the best hyperparameters were
learning_rate='adaptive', max_iter=2000, hidden_layer_sizes=(50,50), and the rest
of hyperparameters were left as the default values'''

NNC = MLPClassifier(learning_rate='adaptive', max_iter=2000, hidden_layer_sizes=(50,50))

NNC.fit(X_train, y_train) #Trainig Model

NNC_score = NNC.score(X_test, y_test) #Accuracy Score

NNC_mean = np.mean(cross_val_score(NNC, X, y, cv=5, scoring = 'accuracy'))#Cross-Validation NNC Score (5 Folds)

NNC_recall = np.mean(cross_val_score(NNC, X, y, cv=5, scoring = 'recall'))#Cross-Validation NNC Recall (5 Folds)

NNC_roc_auc = np.mean(cross_val_score(NNC, X, y, cv=5, scoring = 'roc_auc'))#Cross-Validation NNC Roc_Auc (5 Folds)

######################## CREATING ENSEMBLE MODEL #############################

from random import randrange #To produce random int numbers

#Using OOP to create an Ensemble Model

class Ensemble_Model:
    
    def __init__(self, ml_models): #Argument list of the models previously trained
          
        self.ml_models = ml_models 
        
    def fit(self, Train_Features, Train_target): #Method fit for each model in ml_models
        
        self.fit_models = [i.fit(Train_Features, Train_target) for i in
                           self.ml_models]
    
    def predict(self, Test_Data): #Method predict by taking into account all the predictions from each model
        
        predict_list = [i.predict(Test_Data) for i in self.fit_models]
        
        df_predictions = pd.DataFrame(np.array(predict_list)).T
        
        def pred_output(row):
            
            if np.mean(row) > 0.5:
                
                row['predictions'] = 1
            
            elif np.mean(row) < 0.5:
                
                row['predictions'] = 0
            
            else:
                
                row['predictions'] = randrange(2)
                
            return row
         
        self.predictions = df_predictions.apply(pred_output, axis = 1)
        
        return self.predictions['predictions']
    
        
    def score(self, Test_Data, True_Data): #Method Score for test data and true data
        
        df_score = pd.DataFrame({'pred':self.predict(Test_Data), 'true': True_Data.reset_index(drop = True)})
        
        def col_score(row):
            
            if row.loc['pred'] == row.loc['true']:
                
                row['score'] = 1
            else:
                
                row['score'] = 0
            
            return row
        
        score_result = np.mean(df_score.apply(col_score, axis = 1)['score'])
            
        return score_result

#Creating model list from each trained model previously
models_list = [svc, lr, GNB, GBC, KNN, RFC, XGB, NNC] 

#Creating instance EMC 
EMC = Ensemble_Model(models_list)

EMC.fit(X_train, y_train) #Training the model      

EMC.predict(X_test)  #Making predictions

t = EMC.predictions #Prediction DataFrame

EMC_score = EMC.score(X_test, y_test) #Accuracy Score 

EMC_recall = recall_score(y_test, EMC.predict(X_test)) #Recall Score

EMC_roc_auc = roc_auc_score(y_test, EMC.predict(X_test)) #Roc_Auc Score

#Cross validation with EMC model

EMC_acc_list = []

for i in range(5):
    
    X_train_EMC, X_test_EMC, y_train_EMC, y_test_EMC = train_test_split(X, y, random_state=i, test_size= 0.2)
    EMC.fit(X_train_EMC, y_train_EMC) 
    EMC_acc_list.append(EMC.score(X_test_EMC, y_test_EMC))

#Cross-Validation EMC Score (5 Folds)
EMC_mean = np.mean(EMC_acc_list)

#Cross validation with EMC model Recall

EMC_recall_list = []

for i in range(5):
    
    X_train_EMC, X_test_EMC, y_train_EMC, y_test_EMC = train_test_split(X, y, random_state=i, test_size= 0.2)
    EMC.fit(X_train_EMC, y_train_EMC) 
    EMC_recall_list.append(recall_score(y_test_EMC, EMC.predict(X_test_EMC)))

#Cross-Validation EMC Score (5 Folds)
EMC_recall = np.mean(EMC_recall_list)

#Cross validation with EMC model Roc_Auc

EMC_roc_auc_list = []

for i in range(5):
    
    X_train_EMC, X_test_EMC, y_train_EMC, y_test_EMC = train_test_split(X, y, random_state=i, test_size= 0.2)
    EMC.fit(X_train_EMC, y_train_EMC) 
    EMC_roc_auc_list.append(roc_auc_score(y_test_EMC, EMC.predict(X_test_EMC)))

#Cross-Validation EMC Score (5 Folds)
EMC_roc_auc = np.mean(EMC_roc_auc_list)

########################## COMPARING MODELS ###################################

#Creating a Series from the different results of each model (Accuracy Scores and CrossValidation accuracy scores)

s1 = pd.Series([svc_score, lr_score, GNB_score, GBC_score, KNN_score, 
                RFC_score, XGB_score, NNC_score, EMC_score], name = 'Score')

s2 = pd.Series([svc_mean, lr_mean, GNB_mean, GBC_mean, KNN_mean, 
                RFC_mean, XGB_mean, NNC_mean, EMC_mean], name = 'Mean')

s3 = pd.Series([svc_recall, lr_recall, GNB_recall, GBC_recall, KNN_recall, 
                RFC_recall, XGB_recall, NNC_recall, EMC_recall], name = 'Recall')

s4 = pd.Series([svc_roc_auc, lr_roc_auc, GNB_roc_auc, GBC_roc_auc, KNN_roc_auc, 
                RFC_roc_auc, XGB_roc_auc, NNC_roc_auc, EMC_roc_auc], name = 'Roc_Auc')

models_df = pd.DataFrame([s1,s2,s3,s4])
models_df.columns = ['svc', 'lr', 'GNB', 'GBC', 'KNN', 'RFC', 'XGB', 'NNC', 'EMC']

#Highest Accuracy Score Model(s)
models_df.T[models_df.T['Score'] == max(models_df.T['Score'])]

#Highest Cross-Validation Accuracy Score Model(s)
models_df.T[models_df.T['Mean'] == max(models_df.T['Mean'])]

#Highest Cross-Validation Recall Score Model(s)
models_df.T[models_df.T['Recall'] == max(models_df.T['Recall'])]

#Highest Cross-Validation Roc_Auc Score Model(s)
models_df.T[models_df.T['Roc_Auc'] == max(models_df.T['Roc_Auc'])]

########################## FEATURE SELECTION #################################

'''Applying feature selection to see if reducing the dimensionality 
of the data improves the model performance'''
# Chi feature selection for categorical input and categorical output
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr


#NUMERICAL CORRELATION ANALYSIS

#Selecting numerical columns
num_col = df.drop('Outcome', axis=1).columns

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
num_analysis(df, num_col)

'''Analysing most relevant numerical features that contributes the most 
to the target variable using ANOVA feature analysis
for numeric input and categorical output'''

fs = SelectKBest(score_func=f_classif, k='all')
fs.fit(df.loc[:,num_col], df['Outcome'])

# what are scores for the features
dict_scores = {}
for i in range(len(fs.scores_)):
    dict_scores['{}'.format(num_col[i])] = [fs.scores_[i]]

df_scores = pd.DataFrame.from_dict(dict_scores).T
df_scores.columns = ['Scores']
df_scores.sort_values(by='Scores', ascending=False, inplace=True)
df_scores.plot(kind='bar')

'''After Analysing the correlation between the feature variables and
their impact on the target variable the best candidate variable
to be removed is SkinThickness because is not highly significative to the target variable and it has
a high correlation with some of the other features'''

#Selecting relevant features from df_enc
X_fs = df.drop(['SkinThickness', 'Outcome'], axis=1)
y = df['Outcome']

#Normalizing Data with MinMaxScaler
scaler = MinMaxScaler()
X_fs = scaler.fit_transform(X_fs)

#Splitting training (80%) and test data (20%)
X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_fs, y, random_state=0, test_size= 0.2)

########### IMPLEMENTING MODELS AFTER FEATURE SELECTION #######################

#EMC model

EMC.fit(X_train_fs, y_train_fs) 
EMC_score_fs = EMC.score(X_test_fs, y_test_fs) 
EMC_recall_fs = recall_score(y_test_fs, EMC.predict(X_test_fs))#Recall Score
#Cross validation with EMC model

EMC_acc_list = []


for i in range(5):
    
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_fs, y, random_state=i, test_size= 0.2)
    EMC.fit(X_train_fs, y_train_fs) 
    EMC_acc_list.append(EMC.score(X_test_fs, y_test_fs))

#Cross-Validation EMC Score (5 Folds)
EMC_mean_fs = np.mean(EMC_acc_list)

#Cross validation with EMC model Recall

EMC_recall_list = []


for i in range(5):
    
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_fs, y, random_state=i, test_size= 0.2)
    EMC.fit(X_train_fs, y_train_fs) 
    EMC_recall_list.append(recall_score(y_test_fs, EMC.predict(X_test_fs)))
    
#Cross validation with EMC model Roc_Auc

EMC_roc_auc_list = []

for i in range(5):
    
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_fs, y, random_state=i, test_size= 0.2)
    EMC.fit(X_train_fs, y_train_fs) 
    EMC_roc_auc_list.append(roc_auc_score(y_test_fs, EMC.predict(X_test_fs)))    

#Cross-Validation EMC Score (5 Folds)
EMC_roc_auc = np.mean(EMC_roc_auc_list)

#Cross-Validation EMC Score (5 Folds)
EMC_recall_fs = np.mean(EMC_recall_list)

#Cross-Validation EMC Score (5 Folds)
EMC_roc_auc_fs = np.mean(EMC_roc_auc_list)

#SVC model
svc.fit(X_train_fs, y_train_fs)
svc_score_fs = svc.score(X_test_fs, y_test_fs)
svc_mean_fs = np.mean(cross_val_score(svc, X_fs, y, cv=5, scoring = 'accuracy'))
svc_recall_fs = np.mean(cross_val_score(svc, X_fs, y, cv=5, scoring = 'recall'))
svc_roc_auc_fs = np.mean(cross_val_score(svc, X_fs, y, cv=5, scoring = 'roc_auc'))

#Logistic Regression model
lr.fit(X_train_fs, y_train_fs)
lr_score_fs = lr.score(X_test_fs, y_test_fs)
lr_mean_fs = np.mean(cross_val_score(lr, X_fs, y, cv=5, scoring = 'accuracy'))
lr_recall_fs = np.mean(cross_val_score(lr, X_fs, y, cv=5, scoring = 'recall'))
lr_roc_auc_fs = np.mean(cross_val_score(lr, X_fs, y, cv=5, scoring = 'roc_auc'))

#Gaussian Naive Bayes model
GNB.fit(X_train_fs, y_train_fs)
GNB_score_fs = GNB.score(X_test_fs, y_test_fs)
GNB_mean_fs = np.mean(cross_val_score(GNB, X_fs, y, cv=5, scoring = 'accuracy'))
GNB_recall_fs = np.mean(cross_val_score(GNB, X_fs, y, cv=5, scoring = 'recall'))
GNB_roc_auc_fs = np.mean(cross_val_score(GNB, X_fs, y, cv=5, scoring = 'roc_auc'))

#Gradient Boosting model
GBC.fit(X_train_fs, y_train_fs)
GBC_score_fs = GBC.score(X_test_fs, y_test_fs)
GBC_mean_fs = np.mean(cross_val_score(GBC, X_fs, y, cv=5, scoring = 'accuracy'))
GBC_recall_fs = np.mean(cross_val_score(GBC, X_fs, y, cv=5, scoring = 'recall'))
GBC_roc_auc_fs = np.mean(cross_val_score(GBC, X_fs, y, cv=5, scoring = 'roc_auc'))

#K Nearest Neighbor model
KNN.fit(X_train_fs, y_train_fs)
KNN_score_fs = KNN.score(X_test_fs, y_test_fs)
KNN_mean_fs = np.mean(cross_val_score(KNN, X_fs, y, cv=5, scoring = 'accuracy'))
KNN_recall_fs = np.mean(cross_val_score(KNN, X_fs, y, cv=5, scoring = 'recall'))
KNN_roc_auc_fs = np.mean(cross_val_score(KNN, X_fs, y, cv=5, scoring = 'roc_auc'))

#Random Forest model
RFC.fit(X_train_fs, y_train_fs)
RFC_score_fs = RFC.score(X_test_fs, y_test_fs)
RFC_mean_fs = np.mean(cross_val_score(RFC, X_fs, y, cv=5, scoring = 'accuracy'))
RFC_recall_fs = np.mean(cross_val_score(RFC, X_fs, y, cv=5, scoring = 'recall'))
RFC_roc_auc_fs = np.mean(cross_val_score(RFC, X_fs, y, cv=5, scoring = 'roc_auc'))

#Xgb model
XGB.fit(X_train_fs, y_train_fs) #Trainig Model
XGB_score_fs = XGB.score(X_test_fs, y_test_fs) #Accuracy Score
XGB_mean_fs = np.mean(cross_val_score(XGB, X_fs, y, cv=5, scoring = 'accuracy'))
XGB_recall_fs = np.mean(cross_val_score(XGB, X_fs, y, cv=5, scoring = 'recall'))
XGB_roc_auc_fs = np.mean(cross_val_score(XGB, X_fs, y, cv=5, scoring = 'roc_auc'))

#Neuronal Network model

NNC.fit(X_train_fs, y_train_fs) #Trainig Model
NNC_score_fs = NNC.score(X_test_fs, y_test_fs) #Accuracy Score
NNC_mean_fs = np.mean(cross_val_score(NNC, X_fs, y, cv=5, scoring = 'accuracy'))#Cross-Validation NNC Score (5 Folds)
NNC_recall_fs = np.mean(cross_val_score(NNC, X_fs, y, cv=5, scoring = 'recall'))#Cross-Validation NNC Recall (5 Folds)
NNC_roc_auc_fs = np.mean(cross_val_score(NNC, X_fs, y, cv=5, scoring = 'roc_auc'))#Cross-Validation NNC Roc_Auc (5 Folds)

################ COMPARING MODELS AFTER FEATURE SELECTION ####################

#Creating a Series from the different results of each model (Accuracy Scores and CrossValidation accuracy scores)

s5 = pd.Series([svc_score_fs, lr_score_fs, GNB_score_fs, GBC_score_fs, KNN_score_fs, 
                RFC_score_fs, XGB_score_fs, NNC_score_fs, EMC_score_fs], name = 'Score')

s6 = pd.Series([svc_mean_fs, lr_mean_fs, GNB_mean_fs, GBC_mean_fs, KNN_mean_fs, 
                RFC_mean_fs, XGB_mean_fs, NNC_mean_fs, EMC_mean_fs], name = 'Mean')

s7 = pd.Series([svc_recall_fs, lr_recall_fs, GNB_recall_fs, GBC_recall_fs, KNN_recall_fs, 
                RFC_recall_fs, XGB_recall_fs, NNC_recall_fs, EMC_recall_fs], name = 'Recall')

s8 = pd.Series([svc_roc_auc_fs, lr_roc_auc_fs, GNB_roc_auc_fs, GBC_roc_auc_fs, KNN_roc_auc_fs, 
                RFC_roc_auc_fs, XGB_roc_auc_fs, NNC_roc_auc_fs, EMC_roc_auc_fs], name = 'Roc_Auc')

models_df_fs = pd.DataFrame([s5,s6,s7, s8])
models_df_fs.columns = ['svc_fs', 'lr_fs', 'GNB_fs', 'GBC_fs', 'KNN_fs', 'RFC_fs', 'XGB_fs', 'NNC_fs', 'EMC_fs']

#Highest Accuracy Score Model(s)
models_df_fs.T[models_df_fs.T['Score'] == max(models_df_fs.T['Score'])]

#Highest Cross-Validation Accuracy Score Model(s)
models_df_fs.T[models_df_fs.T['Mean'] == max(models_df_fs.T['Mean'])]

#Highest Cross-Validation Recall Score Model(s)
models_df_fs.T[models_df_fs.T['Recall'] == max(models_df_fs.T['Recall'])]

#Highest Cross-Validation Roc_Auc Score Model(s)
models_df_fs.T[models_df_fs.T['Roc_Auc'] == max(models_df_fs.T['Roc_Auc'])]

#COMPARING ALL MODELS FROM THE ANALYSIS

df_model_comp = models_df.merge(models_df_fs, left_index=True, right_index=True )

#Highest Accuracy Score Model(s)
df_model_comp.T[df_model_comp.T['Score'] == max(df_model_comp.T['Score'])]

#Highest Cross-Validation Accuracy Score Model(s)
df_model_comp.T[df_model_comp.T['Mean'] == max(df_model_comp.T['Mean'])]

#Highest Cross-Validation Recall Score Model(s)
df_model_comp.T[df_model_comp.T['Recall'] == max(df_model_comp.T['Recall'])]

#Highest Cross-Validation Roc_Auc Score Model(s)
df_model_comp.T[df_model_comp.T['Roc_Auc'] == max(df_model_comp.T['Roc_Auc'])]














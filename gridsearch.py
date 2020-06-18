#!/usr/bin/env python
# coding: utf-8

# In[16]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn_extra.cluster import KMedoids
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np


# In[17]:


def gridsearch_svm(X_val, y_val):
    
    param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
  
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 

    # fitting the model for grid search 
    grid_fit = grid.fit(X_val, y_val)
    best_params=grid_fit.best_params_
    
    return best_params


# In[18]:


def gridsearch_logistic(X_val, y_val):
    param_grid = {'penalty': ['l1', 'l2'],'C':[0.001,.009,0.01,.09,1,5,10,25]}
    grid = GridSearchCV(LogisticRegression(), param_grid, scoring = 'recall')
    
    # fitting the model for grid search 
    grid_fit = grid.fit(X_val, y_val)
    best_params=grid_fit.best_params_
    
    return best_params


# In[19]:


def gridsearch_knn(X_val, y_val):
    param_grid = {"n_neighbors": np.arange(1, 31, 2),"metric": ["euclidean", "cityblock"]}
    grid = GridSearchCV( KNeighborsClassifier(), param_grid)
    
    # fitting the model for grid search 
    grid_fit = grid.fit(X_val, y_val)
    best_params=grid_fit.best_params_
    
    return best_params


# In[20]:


def gridsearch_dt(X_val, y_val):
    param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],
          'random_state':[123],
          'criterion':["gini","entropy"],
          'random_state':[123]}
    
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, n_jobs=-1)
    # fitting the model for grid search 
    grid_fit = grid.fit(X_val, y_val)
    best_params=grid_fit.best_params_
    
    return best_params


# In[21]:


def gridsearch_rf(X_val, y_val):
    param_grid = {'criterion':['gini','entropy'],
              'n_estimators':[10,15,20,25,30],
              'min_samples_leaf':[1,2,3],
              'min_samples_split':[3,4,5,6,7], 
              'random_state':[123],
              'n_jobs':[-1]}
    #Making models with hyper parameters sets
    grid = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=-1)
    # fitting the model for grid search 
    grid_fit = grid.fit(X_val, y_val)
    best_params=grid_fit.best_params_
    
    return best_params


# In[ ]:





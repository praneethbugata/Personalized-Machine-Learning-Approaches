import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from plotting import plotting_accuracy_p2,plotting_mae_p2,plotting_mae,plotting_accuracy
from gridsearch import gridsearch_svm,gridsearch_logistic,gridsearch_knn,gridsearch_dt,gridsearch_rf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

def models_train_batch(X_train, X_val, y_train, y_val):
    models=[]
    classifiers_name=["Logistic Regression","SVM","Naive Bayes","Decision Tree","Random Forest","AdaBoost"]
    #logistic regression
    grid_params_lr=gridsearch_logistic(X_val, y_val)
    model_lr = LogisticRegression(C=grid_params_lr['C'],penalty=grid_params_lr['penalty'])
    model_lr.fit(X_train,y_train)
    models.append(model_lr)
    
    #svm
    grid_params_svm=gridsearch_svm(X_val, y_val)
    model_svm = SVC(C=grid_params_svm['C'],gamma=grid_params_svm['gamma'])
    model_svm.fit(X_train, y_train)
    models.append(model_svm)
    
    #naive bayes
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    models.append(model_nb)
    
    #DT
    grid_params_dt=gridsearch_dt(X_val, y_val)
    model_dtc = DecisionTreeClassifier(criterion=grid_params_dt['criterion'],max_features=grid_params_dt['max_features'],min_samples_split=grid_params_dt['min_samples_split'],min_samples_leaf=grid_params_dt['min_samples_leaf'], random_state = 123)
    model_dtc.fit(X_train, y_train)
    models.append(model_dtc)
    
    #Random forest
    grid_params_rf=gridsearch_rf(X_val, y_val)
    model_rf = RandomForestClassifier(criterion=grid_params_rf['criterion'],min_samples_split=grid_params_rf['min_samples_split'],min_samples_leaf=grid_params_rf['min_samples_leaf'],n_estimators = grid_params_rf['n_estimators'], random_state = 123)
    model_rf.fit(X_train, y_train)
    models.append(model_rf)
    
    #AdaBoost
    model_ad = AdaBoostClassifier(n_estimators=100, random_state=0)
    model_ad.fit(X_train, y_train)
    models.append(model_ad)
    
    model_dictionary = dict(zip(classifiers_name,models))
    
    return model_dictionary


def models_train(dataset,X,Y):

    if dataset == "liver_disease":
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.25, random_state=42)
    elif dataset == "diabetes":
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    models=[]
    classifiers_name=["Logistic Regression","SVM","Decision Tree","Random Forest","AdaBoost"]
    #logistic regression
    if dataset == "breast_cancer"or dataset == "liver_disease":
        model_lr = LogisticRegression()
        model_lr.fit(X_train,y_train)
    else:
        grid_params_lr=gridsearch_logistic(X_val, y_val)
        model_lr = LogisticRegression(C=grid_params_lr['C'],penalty=grid_params_lr['penalty'])
        model_lr.fit(X_train,y_train)
    models.append(model_lr)
    
    #svm
    if dataset == "breast_cancer"or dataset == "liver_disease":
        model_svm = SVC()
        model_svm.fit(X_train, y_train)
    else:
        grid_params_svm=gridsearch_svm(X_val, y_val)
        model_svm = SVC(C=grid_params_svm['C'],gamma=grid_params_svm['gamma'])
        model_svm.fit(X_train, y_train)
    models.append(model_svm)


    #DT
    if dataset == "breast_cancer"or dataset == "liver_disease":
        model_dtc = DecisionTreeClassifier(random_state = 123)
        model_dtc.fit(X_train, y_train)
    else:
        grid_params_dt=gridsearch_dt(X_val, y_val)
        model_dtc = DecisionTreeClassifier(criterion=grid_params_dt['criterion'],max_features=grid_params_dt['max_features'],min_samples_split=grid_params_dt['min_samples_split'],min_samples_leaf=grid_params_dt['min_samples_leaf'], random_state = 123)
        model_dtc.fit(X_train, y_train)
    models.append(model_dtc)

    #Random forest
    if dataset == "breast_cancer"or dataset == "liver_disease":
        model_rf = RandomForestClassifier(random_state = 123)
        model_rf.fit(X_train, y_train)
    else:
        grid_params_rf=gridsearch_rf(X_val, y_val)
        model_rf = RandomForestClassifier(criterion=grid_params_rf['criterion'],min_samples_split=grid_params_rf['min_samples_split'],min_samples_leaf=grid_params_rf['min_samples_leaf'],n_estimators = grid_params_rf['n_estimators'], random_state = 123)
        model_rf.fit(X_train, y_train)
    models.append(model_rf)
    
    #AdaBoost
    model_ad = AdaBoostClassifier(n_estimators=100, random_state=0)
    model_ad.fit(X_train, y_train)
    models.append(model_ad)
    
    model_dictionary = dict(zip(classifiers_name,models))
    
    return model_dictionary


def models_train_Approach1(dataset,X,Y):

    if dataset == "heart_disease":
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.15, random_state=42)
    elif dataset == "diabetes":
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.275, random_state=42)
    else:
        X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
    
        
    models=[]
    classifiers_name=["Logistic Regression","SVM","Decision Tree","Random Forest","AdaBoost"]
    #logistic regression
    if dataset == "breast_cancer"or dataset == "liver_disease":
        model_lr = LogisticRegression()
        model_lr.fit(X_train,y_train)
    else:
        grid_params_lr=gridsearch_logistic(X_val, y_val)
        model_lr = LogisticRegression(C=grid_params_lr['C'],penalty=grid_params_lr['penalty'])
        model_lr.fit(X_train,y_train)
    models.append(model_lr)
    
    #svm
    if dataset == "breast_cancer"or dataset == "liver_disease":
        model_svm = SVC()
        model_svm.fit(X_train, y_train)
    else:
        grid_params_svm=gridsearch_svm(X_val, y_val)
        model_svm = SVC(C=grid_params_svm['C'],gamma=grid_params_svm['gamma'])
        model_svm.fit(X_train, y_train)
    models.append(model_svm)


    #DT
    if  dataset == "breast_cancer"or dataset == "liver_disease":
        model_dtc = DecisionTreeClassifier(random_state = 123)
        model_dtc.fit(X_train, y_train)
    else:
        grid_params_dt=gridsearch_dt(X_val, y_val)
        model_dtc = DecisionTreeClassifier(criterion=grid_params_dt['criterion'],max_features=grid_params_dt['max_features'],min_samples_split=grid_params_dt['min_samples_split'],min_samples_leaf=grid_params_dt['min_samples_leaf'], random_state = 123)
        model_dtc.fit(X_train, y_train)
    models.append(model_dtc)

    #Random forest
    if dataset == "breast_cancer"or dataset == "liver_disease":
        model_rf = RandomForestClassifier(random_state = 123)
        model_rf.fit(X_train, y_train)
    else:
        grid_params_rf=gridsearch_rf(X_val, y_val)
        model_rf = RandomForestClassifier(criterion=grid_params_rf['criterion'],min_samples_split=grid_params_rf['min_samples_split'],min_samples_leaf=grid_params_rf['min_samples_leaf'],n_estimators = grid_params_rf['n_estimators'], random_state = 123)
        model_rf.fit(X_train, y_train)
    models.append(model_rf)
    
    #AdaBoost
    model_ad = AdaBoostClassifier(n_estimators=100, random_state=0)
    model_ad.fit(X_train, y_train)
    models.append(model_ad)
    
    model_dictionary = dict(zip(classifiers_name,models))
    
    return model_dictionary

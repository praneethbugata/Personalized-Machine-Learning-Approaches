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

def models_test_approach_2(X_test,y_test,models,key_max_list,dataset,n,method):
    classifiers_name=[]
    auc_list=[]
    mae_list=[]
    auc=[]
    mae=[]
    for i in range(len(models)):
        key_max = key_max_list[i]
        classifiers_name.append(key_max)


        #Logistic Regression
        if key_max == "Logistic Regression":

            model_lr=models[i]['Logistic Regression']
            lr_pred = model_lr.predict(X_test)

            lr_mae = mean_absolute_error(y_test,lr_pred)
            lr_score = accuracy_score(y_test,lr_pred)
            auc.append(lr_score)
            mae.append(lr_mae)
        #SVM        
        if key_max == "SVM":

            model_svm=models[i]['SVM']
            svm_pred = model_svm.predict(X_test)

            svm_mae = mean_absolute_error(y_test,svm_pred)
            svm_score = accuracy_score(y_test,svm_pred)
            auc.append(svm_score)
            mae.append(svm_mae)
		

        
        #DT
        if key_max == "Decision Tree":
            model_dt = models[i]['Decision Tree']
            dtc_pred = model_dt.predict(X_test)

            dtc_mae = mean_absolute_error(y_test,dtc_pred)
            dtc_score = accuracy_score(y_test,dtc_pred)
            auc.append(dtc_score)
            mae.append(dtc_mae)
        
        #Random Forest
        if key_max == "Random Forest":
            model_rf = models[i]['Random Forest']
            rf_pred = model_rf.predict(X_test)

            rf_mae = mean_absolute_error(y_test,rf_pred)
            rf_score = accuracy_score(y_test,rf_pred)
            auc.append(rf_score)
            mae.append(rf_mae)

        #Adaboost
        if key_max == "AdaBoost":
            model_ad=models[i]['AdaBoost']
            ad_pred = model_ad.predict(X_test)

            ad_mae = mean_absolute_error(y_test,ad_pred)
            ad_score=accuracy_score(y_test,ad_pred)
            auc.append(ad_score)
            mae.append(ad_mae)
        
    #auc_dict=dict(zip(classifiers_name,auc))
    #auc_dict = {classifiers_name[i]: auc[i] for i in range(len(classifiers_name))}
    res = {} 
    for key in classifiers_name: 
        for value in auc: 
            res[key] = value 
            auc.remove(value) 
            break
    auc_dict=res
    mae_dict=dict(zip(classifiers_name,mae))
    
    auc_list.append(auc_dict)
    mae_list.append(mae_dict)
    return auc_list,mae_list
def models_test(X_test,y_test,models,dataset,n,method):
    classifiers_name=["Logistic Regression","SVM","Decision Tree","Random Forest","AdaBoost"]
    auc_list=[]
    mae_list=[]
    for i in range(len(models)):
        auc=[]
        mae=[]
        
        #Logistic Regression
        model_lr=models[i]['Logistic Regression']
        lr_pred = model_lr.predict(X_test)
    
        lr_mae = mean_absolute_error(y_test,lr_pred)
        lr_score = accuracy_score(y_test,lr_pred)
        auc.append(lr_score)
        mae.append(lr_mae)
        
        #SVM
        model_svm=models[i]['SVM']
        svm_pred = model_svm.predict(X_test)
    
        svm_mae = mean_absolute_error(y_test,svm_pred)
        svm_score = accuracy_score(y_test,svm_pred)
        auc.append(svm_score)
        mae.append(svm_mae)
        

        
        #DT
        model_dt = models[i]['Decision Tree']
        dtc_pred = model_dt.predict(X_test)
    
        dtc_mae = mean_absolute_error(y_test,dtc_pred)
        dtc_score = accuracy_score(y_test,dtc_pred)
        auc.append(dtc_score)
        mae.append(dtc_mae)
        
        #Random Forest
        model_rf = models[i]['Random Forest']
        rf_pred = model_rf.predict(X_test)
    
        rf_mae = mean_absolute_error(y_test,rf_pred)
        rf_score = accuracy_score(y_test,rf_pred)
        auc.append(rf_score)
        mae.append(rf_mae)
        
        #Adaboost
        model_ad=models[i]['AdaBoost']
        ad_pred = model_ad.predict(X_test)
    
        ad_mae = mean_absolute_error(y_test,ad_pred)
        ad_score=accuracy_score(y_test,ad_pred)
        auc.append(ad_score)
        mae.append(ad_mae)
        
        auc_dict=dict(zip(classifiers_name,auc))
        mae_dict=dict(zip(classifiers_name,mae))
        
        auc_list.append(auc_dict)
        mae_list.append(mae_dict)
        
    for j in range(len(auc_list)):
        auc_list_keys=list(auc_list[j].keys())
        auc_list_values=list(auc_list[j].values())
        train_person="Person_"+str(j)
        method="Personalized_Approach_2 model"
        
        plotting_accuracy_p2(auc_list_keys,auc_list_values,method,dataset,train_person,n)
    
        mae_list_keys=list(mae_list[j].keys())
        mae_list_values=list(mae_list[j].values())
        train_person="Person_"+str(j)
        
        
        plotting_mae_p2(mae_list_keys,mae_list_values,method,dataset,train_person,n)
    
    return auc_list,mae_list

def models_test_single(X_test,y_test,models,dataset,n,method):
    classifiers_name=["Logistic Regression","SVM","Decision Tree","Random Forest","AdaBoost"]
    auc_list=[]
    mae_list=[]
    
    auc=[]
    mae=[]

    #Logistic Regression
    model_lr=models['Logistic Regression']
    lr_pred = model_lr.predict(X_test)

    lr_mae = mean_absolute_error(y_test,lr_pred)
    lr_score = accuracy_score(y_test,lr_pred)
    auc.append(lr_score)
    mae.append(lr_mae)

    #SVM
    model_svm=models['SVM']
    svm_pred = model_svm.predict(X_test)

    svm_mae = mean_absolute_error(y_test,svm_pred)
    svm_score = accuracy_score(y_test,svm_pred)
    auc.append(svm_score)
    mae.append(svm_mae)



    #DT
    model_dt = models['Decision Tree']
    dtc_pred = model_dt.predict(X_test)

    dtc_mae = mean_absolute_error(y_test,dtc_pred)
    dtc_score = accuracy_score(y_test,dtc_pred)
    auc.append(dtc_score)
    mae.append(dtc_mae)

    #Random Forest
    model_rf = models['Random Forest']
    rf_pred = model_rf.predict(X_test)

    rf_mae = mean_absolute_error(y_test,rf_pred)
    rf_score = accuracy_score(y_test,rf_pred)
    auc.append(rf_score)
    mae.append(rf_mae)

    #Adaboost
    model_ad=models['AdaBoost']
    ad_pred = model_ad.predict(X_test)

    ad_mae = mean_absolute_error(y_test,ad_pred)
    ad_score=accuracy_score(y_test,ad_pred)
    auc.append(ad_score)
    mae.append(ad_mae)

    auc_dict=dict(zip(classifiers_name,auc))
    mae_dict=dict(zip(classifiers_name,mae))

    auc_list.append(auc_dict)
    mae_list.append(mae_dict)
        
    for j in range(len(auc_list)):
        auc_list_keys=list(auc_list[j].keys())
        auc_list_values=list(auc_list[j].values())
        
        
        
        plotting_accuracy(auc_list_keys,auc_list_values,method,dataset,n)
    
        mae_list_keys=list(mae_list[j].keys())
        mae_list_values=list(mae_list[j].values())
        
        
        
        plotting_mae(mae_list_keys,mae_list_values,method,dataset,n)
    
    return auc_list,mae_list


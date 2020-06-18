from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from train_test_split_pr import train_test_split_dataset_personalized 

def rf_train(x_train,y_train):
    models = []
    regressors = ['Random Forest Regression']


    #random forest Regression
    rf_clf = RandomForestRegressor(max_depth=2, random_state=0)
    rf_clf.fit(x_train,y_train)
    models.append(rf_clf)
    model_dictionary = dict(zip(regressors,models))

    return model_dictionary
    
def rf_test(x_test,y_test,models):

    mae_list=[]
    person_list=[]
    for i in range(len(models)):
        
        #random forest Regression
        rf_clf=models[i]['Random Forest Regression']
        rf_predcited = rf_clf.predict(x_test)
        rf_mae = mean_absolute_error(y_test,rf_predcited)
        person = 'person'+str(i+1)
        person_list.append(person)
        mae_list.append(rf_mae)
        
    minpos = mae_list.index(min(mae_list))   
    mae_dict = dict(zip(person_list,mae_list))
    #mae_final.append(mae_dict)
    
    min_person = min(mae_dict, key=mae_dict.get)
    
    return mae_dict,min_person,minpos

def re_train_rf(i,dataset,train,chunks,rf_minpos,rf_models_list,x_test,y_test):
    upd_train = train[rf_minpos].append(chunks[i-1])
    train.append(upd_train)
    x_train, y_train = train_test_split_dataset_personalized(dataset,train[-1])
    rf_model = rf_train(x_train,y_train)
    rf_models_list.append(rf_model)
    #predict                 
    rf_mae,rf_min_person,rf_minpos= rf_test(x_test,y_test,rf_models_list)
    
    return rf_mae,rf_min_person,rf_minpos,rf_models_list,train
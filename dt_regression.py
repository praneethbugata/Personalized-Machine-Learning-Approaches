from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from train_test_split_pr import train_test_split_dataset_personalized 

def dt_train(x_train,y_train):
    models = []
    regressors = ['Decision Tree Regression']


    #decision tree Regression
    dt_clf = DecisionTreeRegressor(random_state=0)
    dt_clf.fit(x_train,y_train)
    models.append(dt_clf)
    model_dictionary = dict(zip(regressors,models))

    return model_dictionary
    
def dt_test(x_test,y_test,models):

    mae_list=[]
    person_list=[]
    for i in range(len(models)):
        
        #random forest Regression
        dt_clf=models[i]['Decision Tree Regression']
        dt_predcited = dt_clf.predict(x_test)
        dt_mae = mean_absolute_error(y_test,dt_predcited)
        person = 'person'+str(i+1)
        person_list.append(person)
        mae_list.append(dt_mae)
        
    minpos = mae_list.index(min(mae_list))   
    mae_dict = dict(zip(person_list,mae_list))
    #mae_final.append(mae_dict)
    
    min_person = min(mae_dict, key=mae_dict.get)
    
    return mae_dict,min_person,minpos

def re_train_dt(i,dataset,train,chunks,dt_minpos,dt_models_list,x_test,y_test):
    upd_train = train[dt_minpos].append(chunks[i-1])
    train.append(upd_train)
    x_train, y_train = train_test_split_dataset_personalized(dataset,train[-1])
    dt_model = dt_train(x_train,y_train)
    dt_models_list.append(dt_model)
    #predict                 
    dt_mae,dt_min_person,dt_minpos= dt_test(x_test,y_test,dt_models_list)
    

    return dt_mae,dt_min_person,dt_minpos,dt_models_list,train
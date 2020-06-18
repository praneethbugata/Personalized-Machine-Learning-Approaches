from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from train_test_split_pr import train_test_split_dataset_personalized 

def ab_train(x_train,y_train):
    models = []
    regressors = ['AdaBoost Regression']


    #decision tree Regression
    ab_clf = AdaBoostRegressor(random_state=0)
    ab_clf.fit(x_train,y_train)
    models.append(ab_clf)
    model_dictionary = dict(zip(regressors,models))

    return model_dictionary
    
def ab_test(x_test,y_test,models):

    mae_list=[]
    person_list=[]
    for i in range(len(models)):
        
        #adaboost Regression
        ab_clf = models[i]['AdaBoost Regression']
        ab_predcited = ab_clf.predict(x_test)
        ab_mae = mean_absolute_error(y_test,ab_predcited)
        person = 'person'+str(i+1)
        person_list.append(person)
        mae_list.append(ab_mae)
        
    minpos = mae_list.index(min(mae_list))   
    mae_dict = dict(zip(person_list,mae_list))
    #mae_final.append(mae_dict)
    
    min_person = min(mae_dict, key=mae_dict.get)
    
    return mae_dict,min_person,minpos

def re_train_ab(i,dataset,train,chunks,ab_minpos,ab_models_list,x_test,y_test):
    upd_train = train[ab_minpos].append(chunks[i-1])
    train.append(upd_train)
    x_train, y_train = train_test_split_dataset_personalized(dataset,train[-1])
    ab_model = ab_train(x_train,y_train)
    ab_models_list.append(ab_model)
    #predict                 
    ab_mae,ab_min_person,ab_minpos= ab_test(x_test,y_test,ab_models_list)
    

    return ab_mae,ab_min_person,ab_minpos,ab_models_list,train
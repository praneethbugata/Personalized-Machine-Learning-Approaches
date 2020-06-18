from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from train_test_split_pr import train_test_split_dataset_personalized


def lasso_train(x_train,y_train):
    models = []
    regressors = ['Lasso Regression']


    #Lasso Regression
    lasso_clf = linear_model.Lasso(alpha=0.1)
    lasso_clf.fit(x_train,y_train)
    models.append(lasso_clf)
    model_dictionary = dict(zip(regressors,models))

    return model_dictionary
    
def lasso_test(x_test,y_test,models):

    mae_list=[]
    person_list=[]
    for i in range(len(models)):
        
        #Lasso Regression
        lasso_clf=models[i]['Lasso Regression']
        lasso_predcited = lasso_clf.predict(x_test)
        lasso_mae = mean_absolute_error(y_test,lasso_predcited)
        person = 'person'+str(i+1)
        person_list.append(person)
        mae_list.append(lasso_mae)
        
    minpos = mae_list.index(min(mae_list))   
    mae_dict = dict(zip(person_list,mae_list))
    #mae_final.append(mae_dict)
    
    min_person = min(mae_dict, key=mae_dict.get)
    
    return mae_dict,min_person,minpos


def re_train_lasso(i,dataset,train,chunks,lasso_minpos,lasso_models_list,x_test,y_test):

    upd_train = train[lasso_minpos].append(chunks[i-1])
    train.append(upd_train)
    x_train, y_train = train_test_split_dataset_personalized(dataset,train[-1])
    lasso_model = lasso_train(x_train,y_train)
    lasso_models_list.append(lasso_model)
    #predict                 
    lasso_mae,lasso_min_person,lasso_minpos= lasso_test(x_test,y_test,lasso_models_list)
    

    return lasso_mae,lasso_min_person,lasso_minpos,lasso_models_list,train


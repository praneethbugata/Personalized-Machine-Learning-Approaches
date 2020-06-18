from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from train_test_split_pr import train_test_split_dataset_personalized 

def lr_train(x_train,y_train):
    models = []
    regressors = ['Linear Regression']


    #lr Regression
    lr_model = LinearRegression()
    lr_model.fit(x_train,y_train)
    models.append(lr_model)
    model_dictionary = dict(zip(regressors,models))

    return model_dictionary
    
def lr_test(x_test,y_test,models):


    mae_list=[]
    person_list=[]
    for i in range(len(models)):
        
        #linear Regression
        lr_model=models[i]['Linear Regression']
        lr_predcited = lr_model.predict(x_test)
        lr_mae = mean_absolute_error(y_test,lr_predcited)
        person = 'person'+str(i+1)
        person_list.append(person)
        mae_list.append(lr_mae)
        
    minpos = mae_list.index(min(mae_list))   
    mae_dict = dict(zip(person_list,mae_list))
    #mae_final.append(mae_dict)
    min_person = min(mae_dict, key=mae_dict.get)
    
    return mae_dict,min_person,minpos

def re_train_lr(i,dataset,train,chunks,lr_minpos,lr_models_list,x_test,y_test):
    upd_train = train[lr_minpos].append(chunks[i-1])
    train.append(upd_train)
    x_train, y_train = train_test_split_dataset_personalized(dataset,train[-1])
    lr_model = lr_train(x_train,y_train)
    lr_models_list.append(lr_model)
    #predict                 
    lr_mae,lr_min_person,lr_minpos= lr_test(x_test,y_test,lr_models_list)
    

    return lr_mae,lr_min_person,lr_minpos,lr_models_list,train
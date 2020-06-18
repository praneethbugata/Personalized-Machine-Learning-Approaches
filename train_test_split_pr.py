import random
def train_test_split_dataset_personalized(dataset,new_df2):
    
    random.seed(2411)
    if dataset == "breast_cancer":
        X = new_df2.iloc[:,0:30]
        Y = new_df2.iloc[:,30]
        Y=Y.astype('int')
        
    elif dataset == "heart_disease":
        X = new_df2.iloc[:,0:22]
        Y = new_df2.iloc[:,22]
        Y=Y.astype('int')
        
    elif dataset == "liver_disease":
        X = new_df2.iloc[:,0:11]
        Y = new_df2.iloc[:,11]
        Y=Y.astype('int')
    elif dataset == "thyroid":
        X = new_df2.iloc[:,0:23]
        Y = new_df2.iloc[:,23]
        Y=Y.astype('int')
    elif dataset == "diabetes":
        X = new_df2.iloc[:,0:8]
        Y = new_df2.iloc[:,8]
        Y=Y.astype('int')
        
    return X,Y  
import seaborn as sns #for plotting
import matplotlib.pyplot as plt

def plotting_accuracy(classifiers,acc,method,dataset,n):
    
    sns.set()
    plt.figure(figsize=(14,6))
    plt.ylabel("Accuracy(%)")
    plt.xlabel("Algorithms")
    sns.barplot(x=classifiers, y=acc, palette="deep")
    len1=max(acc)
    for line in range(len(classifiers)):
         plt.text(line-0.15, # x
                  len1, # y
                 "{:.2f}%".format(acc[line]*100),
                 horizontalalignment='left',
                  size='large',
                 color="black",
                 )
    plt.title(method+" accuracy for "+dataset+" of Test Person "+str(n))
    plt.show()


def plotting_mae(classifiers,mae,method,dataset,n):
    sns.set()
    plt.figure(figsize=(14,6))
    plt.ylabel("MAE")
    plt.xlabel("Algorithms")
    sns.barplot(x=classifiers, y=mae, palette="deep")
    len1=max(mae)
    for line in range(len(classifiers)):
         plt.text(line-0.15, # x
                  len1, # y
                 "{:.2f}".format(mae[line]), 
                 horizontalalignment='left',
                  size='large',
                 color="black",
                 )
    plt.title(method+" MAE for "+dataset+" of Test Person "+str(n))
    plt.show()

def plotting_accuracy_p2(classifiers,acc,method,dataset,person,n):
    
    sns.set()
    plt.figure(figsize=(14,6))
    plt.ylabel("Accuracy(%)")
    plt.xlabel("Algorithms")
    sns.barplot(x=classifiers, y=acc, palette="deep")
    len1=max(acc)
    for line in range(len(classifiers)):
         plt.text(line-0.15, # x
                  len1, # y
                 "{:.2f}%".format(acc[line]*100),
                 horizontalalignment='left',
                  size='large',
                 color="black",
                 )
    plt.title(method+"_ accuracy of "+person+" for "+dataset+" of Test Person_"+str(n))
    plt.show()


def plotting_mae_p2(classifiers,mae,method,dataset,person,n):
    sns.set()
    plt.figure(figsize=(14,6))
    plt.ylabel("MAE")
    plt.xlabel("Algorithms")
    sns.barplot(x=classifiers, y=mae, palette="deep")
    len1=max(mae)
    for line in range(len(classifiers)):
         plt.text(line-0.15, # x
                  len1, # y
                 "{:.2f}".format(mae[line]), 
                 horizontalalignment='left',
                  size='large',
                 color="black",
                 )
    plt.title(method+"_ mae of "+person+" for "+dataset+" of Test Person_"+str(n))
    plt.show()

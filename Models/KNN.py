import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

def loadData():
    # load our data and separate it into train and labels
    train_data_df=pd.read_csv("./Data/traindata_temp.csv",sep=",")
    train_data_df = train_data_df.sample(frac=1).reset_index(drop=True)
    train_labels_df=train_data_df[train_data_df.columns[0:1]]
    train_data_df.drop(train_data_df.columns[[0]], axis=1)
    train_data=train_data_df.values
    
    train_labels_temp=train_labels_df.values
    train_labels=np.array([int(t[0]) for t in train_labels_temp])

    test_data_df=pd.read_csv("./Data/testdata_temp.csv",sep=",")
    test_data=test_data_df.values
    test_labels_df = test_data_df[test_data_df.columns[0:1]]
    test_data_df.drop(test_data_df.columns[[0]], axis=1)
    test_data=test_data_df.values
    test_labels=test_labels_df.values
    
    #split test data into validation set of four batch
    return train_data,train_labels,test_data,test_labels

 
# loading the iris dataset
X_train,y_train,X_test,y_test=loadData()
  
# training a KNN classifier

knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
 
# accuracy on X_test
accuracy = knn.score(X_test, y_test)
print (accuracy)
 
# creating a confusion matrix
test_pred = knn.predict(X_test) 
cm = confusion_matrix(y_test, test_pred)


acc_score=accuracy_score(y_test, test_pred)
fscore_val=f1_score(y_test, test_pred, average=None)
test_precision_val=precision_score(y_test, test_pred, average=None)
test_recall_val=recall_score(y_test, test_pred, average=None)

print("Classes test acc_score at step", ":", acc_score)
print("Classes test f1score at step", ":", fscore_val)
print("Classes test precision at step", ":", test_precision_val)
print("Classes test recall at step", ":", test_recall_val)



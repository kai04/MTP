import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def roccurve(n_classes,y_test,y_pred):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
#        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

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

#X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
#Y = np.array([1, 1, 1, 2, 2, 2])
X,Y,testX,test_true=loadData()

#clf = GaussianNB()
#clf.fit(X, Y)
#GaussianNB(priors=None)
#print(clf.predict([[-0.8, -1]]))

clf_pf = GaussianNB()
clf_pf.partial_fit(X, Y, np.unique(Y))
test_pred=clf_pf.predict(testX)

acc_score=accuracy_score(test_true, test_pred)
fscore_val=f1_score(test_true, test_pred, average=None)
test_precision_val=precision_score(test_true, test_pred, average=None)
test_recall_val=recall_score(test_true, test_pred, average=None)

print("Classes test acc_score at step", ":", acc_score)
print("Classes test f1score at step", ":", fscore_val)
print("Classes test precision at step", ":", test_precision_val)
print("Classes test recall at step", ":", test_recall_val)

#roccurve(14,test_true,test_pred)



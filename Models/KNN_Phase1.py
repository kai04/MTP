import scipy
import sklearn
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
import pandas as pd
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#read train data
data = pd.read_csv('./Phase1_Data/train.csv')
class1 = data.iloc[:,-1]
del data["Raga Class"]

#read train data
testdata = pd.read_csv('./Phase1_Data/test.csv')
true_values = testdata.iloc[:,-1]
del testdata["Raga Class"]

clf = NearestCentroid()
clf.fit(data, class1)
NearestCentroid(metric='euclidean', shrink_threshold=None)
pred_values = clf.predict(testdata)
con_mat = confusion_matrix(true_values, pred_values, [0, 1 ,2])
total_accuracy = (con_mat[0, 0] + con_mat[1, 1]) / float(np.sum(con_mat))
class1_accuracy = (con_mat[0, 0] / float(np.sum(con_mat[0, :])))
class2_accuracy = (con_mat[1, 1] / float(np.sum(con_mat[1, :])))
class3_accuracy = (con_mat[2, 2] / float(np.sum(con_mat[2, :])))
acc = accuracy_score(true_values, pred_values)
print("Accuracy:",acc)

print('Total accuracy: %.5f' % total_accuracy)
print('Class1 accuracy: %.5f' % class1_accuracy)
print('Class2 accuracy: %.5f' % class2_accuracy)
print('Class3 accuracy: %.5f' % class3_accuracy)
print('Geometric mean accuracy: %.5f' % math.sqrt((class1_accuracy * class2_accuracy)))
	

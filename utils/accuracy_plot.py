#/usr/bin/env python                                                       
# coding:utf-8                                                             

# need to install python-pygresql
import matplotlib.pyplot as plt
import sys
import os
from pylab import *
import pickle as p
import numpy as np

def save_obj(obj, name ):
    with open(name , 'wb') as f:
        p.dump(obj, f, p.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name , 'rb') as f:
        return p.load(f) 


#fnx = lambda : np.random.randint(3, 10, 10)
operation=sys.argv[1]
path=os.getcwd()
suffix=operation+".pkl"
list1=[]
for file1 in os.listdir(path):
    if file1.endswith(suffix):
#        print(file1)
        list1.append((load_obj(file1)).tolist())
        
b=tuple(list1)   
a=tuple()
for q in list1:
    a=a+tuple(q)

#print(a)
y = np.row_stack(b) 

x = np.arange(14) 
y_stack = np.cumsum(y, axis=0)  



fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(111)

#print("sizeof x:",np.shape(x))
#print("sizeof y:",np.shape(y_stack[0,:]))

ax1.plot(x, y_stack[0,:], label="KNN")
ax1.plot(x, y_stack[1,:], label="SVM")
ax1.plot(x, y_stack[2,:], label="Decision Tree")
ax1.plot(x, y_stack[3,:], label="Bayesian")
ax1.legend(loc=1)

plt.xticks(x)
plt.xlabel('Raga\'s')
plt.ylabel(operation)

colormap = plt.cm.gist_ncar 
colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]
for i,j in enumerate(ax1.lines):
    j.set_color(colors[i])
outfile=operation+"_plot.png"
plt.savefig(outfile)

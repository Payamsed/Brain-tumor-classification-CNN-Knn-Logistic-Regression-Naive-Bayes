# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 23:20:38 2022

@author: payam
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2






list_of_pics_yes = list()
for i in range(0,1500):
    train_image_yes= imread('y'+str(i)+'.jpg')
    train_y = cv2.resize(train_image_yes, (35,35))
    if len(train_y.shape) == 3:
        train_y = cv2.cvtColor(train_y, cv2.COLOR_BGR2GRAY)
    train_y = np.reshape(train_y,1225)
    train_y = np.append(train_y,1)
    list_of_pics_yes.append(train_y)
list_of_pics_yes = np.asarray(list_of_pics_yes)


# no dataset labeling(0)
list_of_pics_no = list()
for i in range(0,1500):
    train_image_no= imread('no'+str(i)+'.jpg')
    train_n = cv2.resize(train_image_no, (35,35))
    if len(train_n.shape) == 3:
        train_n = cv2.cvtColor(train_n, cv2.COLOR_BGR2GRAY)
    train_n = np.reshape(train_n,1225)
    train_n = np.append(train_n,0)
    list_of_pics_no.append(train_n)
list_of_pics_no = np.asarray(list_of_pics_no)





list_of_pics_ytest = list_of_pics_yes[300:600]


# # no_test labeling(-1)

list_of_pics_ntest = list_of_pics_no[300:600]


test_dataset = np.concatenate((list_of_pics_ntest,list_of_pics_ytest),axis = 0)

# train dataset and labels dataset

train_dataset = np.concatenate(((np.concatenate((list_of_pics_no[600:1500],list_of_pics_no[0:300]),axis = 0),
                                 np.concatenate((list_of_pics_yes[600:1500],list_of_pics_yes[0:300]),axis = 0))),axis = 0)
x_labels = train_dataset[:,-1]
y_labels = test_dataset[:,-1]

train_dataset = train_dataset[:,:-1]

test_dataset = test_dataset[:,:-1]
# normalizing

xtrainnorm = (train_dataset-(np.amin(train_dataset))+0)/((np.amax(train_dataset))-(np.amin(train_dataset)))

xtestnorm = (test_dataset-(np.amin(train_dataset))+0)/((np.amax(train_dataset))-(np.amin(train_dataset)))




# print(xtrainnorm.shape)
# print(xtestnorm.shape)
# print(train_dataset.shape)
# print(test_dataset.shape)


xtrainnorm = xtrainnorm.T
xtestnorm = xtestnorm.T

ytrain = x_labels.reshape(1,xtrainnorm.shape[1])
ytest = y_labels.reshape(1,xtestnorm.shape[1])


# print(xtrainnorm.shape)
# print(ytrain.shape)
# print(xtestnorm.shape)
# print(ytest.shape)

def sig(u):
  return 1/(1+np.exp(-u))






# logistic regression

def logistic(xtrain,ytrain,learningrate,epochs,c,x_test,y_test):
    m = xtrain.shape[1]
    n = xtrain.shape[0]
  # gaussian or not
  
  # stochastic gradient decent

    if c == 1:
      b = np.random.normal(0, 1)
      w = np.random.normal(0, 1, n)
      w = np.reshape(w, (n, 1))
    elif c== 2:
      b = np.random.uniform(0, n)
      w = np.random.uniform(0, 1, n)
      w = np.reshape(w, (n, 1))
    
    else:  
      w = np.zeros((n,1))
      b = 0
    costlist= []
    acs = []
    js = []
    for i in range(epochs):
      k = np.dot(w.T,xtrain) + b
      j = sig(k)
      # the cost is observed to see whether the algorithm is working or not
      cost = -(1/m)*np.sum(ytrain*np.log(j)+(1-ytrain)*np.log(1-j))
      dw = (1/m)*np.dot(j-ytrain,xtrain.T)
      db = (1/m)*np.sum(j-ytrain)
      w = w - learningrate*dw.T 
      b = b - learningrate*db
      ac,jp = accuracy(x_test,y_test,w,b)
      costlist.append(cost)
      acs.append(ac)
      js.append(jp)
      if(i%(epochs/10)) == 0:
          print(cost)
        

    return w, b, costlist,acs,js

def conf(y_pre, y_tes):
  con_matrix = np.zeros((2, 2), dtype=int)
  for x in range(len(y_pre)):
    if y_pre[x] == 1 and y_tes[x] == 1:
      con_matrix[1][1] += 1  # tn
    elif y_pre[x] == 1 and y_tes[x] == 0:
      con_matrix[1][0] += 1  # fn
    elif y_pre[x] == 0 and y_tes[x] == 1:
      con_matrix[0][1] += 1  # fp
    else:
      con_matrix[0][0] += 1  # tp

  return np.asarray(con_matrix)

def accuracy(x,y,w,b):
  k = np.dot(w.T,x) + b
  j = sig(k)

  j = j>0.5
  j = np.array(j,dtype = 'int64')

  acuracy = (1-np.sum(np.absolute(j-y))/y.shape[0])*100
  return acuracy,j



# print(y_labels.shape)
epochs = [2000,5000,7000,10000]
learningrate = [10**-1,10**-2,10**-3,10**-4]

for e in epochs:
    for l in learningrate:

        print("first 10 cost values for",e,"epochs and",l,"learning rate is:")
        w,b,costlist,acs,jp = logistic(xtrainnorm,x_labels,learningrate = l,epochs = e,c = 0,
                                       x_test = xtestnorm,y_test = y_labels)
        jpp = np.asarray(jp)[-1][0]
    
        c = conf(jpp,y_labels)
        print("confusion matrix is:")
        print(c)
        print("final accuracy is :",acs[-1])
        
        print(l,"learning rate",e," epochs plots:")
        plt.plot(np.arange(e),costlist)
        plt.xlabel('epochs')
        plt.ylabel('cost')
        plt.figure()
        plt.plot(np.arange(e),acs)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.figure()















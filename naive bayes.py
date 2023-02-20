# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 02:21:51 2022

@author: payam
"""

import numpy as np
from matplotlib.image import imread
import cv2

# test dataset labeling()
# import matplotlib.pyplot as plt
# nos = 110
# yess = 95




# yes_train labeling(1)

list_of_pics_yes = list()
for i in range(0,1500):
    train_image_yes= imread('y'+str(i)+'.jpg')
    train_y = cv2.resize(train_image_yes, (20,20))
    if len(train_y.shape) == 3:
        train_y = cv2.cvtColor(train_y, cv2.COLOR_BGR2GRAY)
    train_y = train_y/255
    train_y = np.reshape(train_y,400)
    train_y = np.append(train_y,1)
    list_of_pics_yes.append(train_y)
list_of_pics_yes = np.asarray(list_of_pics_yes)


# no_train labeling(2)
list_of_pics_no = list()
for i in range(0,1500):
    train_image_no= imread('no'+str(i)+'.jpg')
    train_n = cv2.resize(train_image_no, (20,20))
    if len(train_n.shape) == 3:
        train_n = cv2.cvtColor(train_n, cv2.COLOR_BGR2GRAY)
    train_n = train_n/ 255
    train_n = np.reshape(train_n,400)
    train_n = np.append(train_n,2)
    list_of_pics_no.append(train_n)
list_of_pics_no = np.asarray(list_of_pics_no)



# train dataset and labels dataset
a= [0,300,600,900,1200]
b = [300,600,900,1200,1500]

acs_a = []
for l,p in zip(a,b):
    train_dataset = np.concatenate(((np.concatenate((list_of_pics_no[p:1500],list_of_pics_no[0:l]),
                                axis = 0),np.concatenate((list_of_pics_yes[p:1500],list_of_pics_yes[0:l]),axis = 0))),axis = 0)
    train_dataset_withoutlabels = np.delete(train_dataset,-1,1)
    # print(train_dataset_withoutlabels.shape)
    list_of_pics_ytest = list_of_pics_yes[l:p]
    
    # # no_test labeling(-1)
    list_of_pics_ntest = list_of_pics_no[l:p]
    test_dataset = np.concatenate((list_of_pics_ntest,list_of_pics_ytest),axis = 0)
    test_dataset_withoutlabels = np.delete(test_dataset,-1,1)
    # print(test_dataset)
    # print(test_dataset_withoutlabels)
    # train labes
    train_labels = train_dataset[:,-1]
    train_labels_yes =np.count_nonzero((train_labels)==1)
    train_labels_no  =np.count_nonzero((train_labels)==2)
    
    # test labels
    test_labels = test_dataset[:,-1]
    
    test_labels_yes =np.count_nonzero((test_labels)==1)
    test_labels_no  =np.count_nonzero((test_labels)==2)
    
    
    
    
    #  probablity of classes
    
    prob_class =[train_labels_yes/3000,train_labels_no/3000]
    
    
    # print(test_dataset_withoutlabels.shape)
    
    # tetas for class 1
    a =1
    
    T1 = 0
    TT1 = []
    for k in range(400):
      T1 = 0
      for i, j in zip(train_dataset_withoutlabels[:, [k]], train_labels):
        if j == 1 and i != 0:
          T1+=1
      TT1.append(T1)
    TT1 = np.asarray(TT1)
    sigT1 = np.array(sum(TT1))
    TETA1 = np.log(TT1/sigT1)
    
    TETA1a = np.log((TT1+a)/(sigT1 +a*400))
    # print(TT1)
    # print(sigT1)
    # print(TETA1)
    
    # print('ssssssssss')
    # tetas for class 2
    T2 = 0
    TT2 = []
    for k in range(400):
      T2 = 0
      for i, j in zip(train_dataset_withoutlabels[:, [k]], train_labels):
        if j == 2 and i != 0:
          T2+=1
      TT2.append(T2)
    TT2 = np.asarray(TT2)
    sigT2 = np.array(sum(TT2))
    TETA2 = np.log(TT2/sigT2)
    
    
    TETA2a = np.log((TT2+a)/(sigT2 +a*400))
    # print(TT2)
    # print(sigT2)
    # print(TETA2.shape)
    
    
    
    # finding predictions
    
    y1 = (prob_class[0]) + np.dot(test_dataset_withoutlabels , TETA1a)
    y1 = np.exp(y1)
    y2 = (prob_class[1]) + np.dot(test_dataset_withoutlabels , TETA2a)
    y2 = np.exp(y2)
                                  
    
    # labeling predictions
    nos =0
    yes =0
    y_pre = []
    for o in range(600):
        if y2[o] > y1[o]:
            nos+=1
            y_pre.append(2)
        else:
            yes+=1
            y_pre.append(1)
    
    
    def conf(y_pre, y_tes):
      con_matrix = np.zeros((2, 2), dtype=int)
      for x in range(len(y_pre)):
        if y_pre[x] == 2 and y_tes[x] == 2:
          con_matrix[1][1] += 1  # tn
        elif y_pre[x] == 2 and y_tes[x] == 1:
          con_matrix[1][0] += 1  # fn
        elif y_pre[x] == 1 and y_tes[x] == 2:
          con_matrix[0][1] += 1  # fp
        else:
          con_matrix[0][0] += 1  # tp
    
      return np.asarray(con_matrix)
    # accuracy
    
    def accuracy(con_matrix):
      return ((con_matrix[0][0] + con_matrix[1][1]) * 100) / (np.sum(con_matrix))
    
    a =conf(y_pre,test_labels)
    print('confusion matrix =','\n', a)    
    print('accuracy = ','\n',accuracy(a),'%')
    acs_a.append(accuracy(a))

print("the average accuracy for 5-fold cross validation is",sum(acs_a)/5)
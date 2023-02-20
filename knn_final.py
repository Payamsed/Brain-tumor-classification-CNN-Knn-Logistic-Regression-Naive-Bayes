# from PIL import Image
import numpy as np
from matplotlib.image import imread
import cv2

# y_test dataset labeling(-1)


# list_of_pics_ytest = list()
# for i in range(301):
#     test_image_yes= imread('predy'+str(i)+'.jpg')
#     test_y = cv2.resize(test_image_yes, (25,25))
#     if len(test_y.shape) == 3:
#         test_y = cv2.cvtColor(test_y, cv2.COLOR_BGR2GRAY)
#     test_y = np.reshape(test_y,625)
#     test_y = np.append(test_y,1)
#     list_of_pics_ytest.append(test_y)
# list_of_pics_ytest = np.asarray(list_of_pics_ytest)



# list_of_pics_ntest = list()
# for i in range(341):
#     test_image_no= imread('predn'+str(i)+'.jpg')
#     test_n = cv2.resize(test_image_no, (25,25))
#     if len(test_n.shape) == 3:
#         test_n = cv2.cvtColor(test_n, cv2.COLOR_BGR2GRAY)
#     test_n = np.reshape(test_n,625)
#     test_n = np.append(test_n,-1)
#     list_of_pics_ntest.append(test_n)
# list_of_pics_ntest = np.asarray(list_of_pics_ntest)




# yes dataset labeling(1)

list_of_pics_yes = list()
for i in range(0,1501):
    train_image_yes= imread('y'+str(i)+'.jpg')
    train_y = cv2.resize(train_image_yes, (25,25))
    if len(train_y.shape) == 3:
        train_y = cv2.cvtColor(train_y, cv2.COLOR_BGR2GRAY)
    train_y = np.reshape(train_y,625)
    train_y = np.append(train_y,1)
    list_of_pics_yes.append(train_y)
list_of_pics_yes = np.asarray(list_of_pics_yes)


# no dataset labeling(0)
list_of_pics_no = list()
for i in range(0,1501):
    train_image_no= imread('no'+str(i)+'.jpg')
    train_n = cv2.resize(train_image_no, (25,25))
    if len(train_n.shape) == 3:
        train_n = cv2.cvtColor(train_n, cv2.COLOR_BGR2GRAY)
    train_n = np.reshape(train_n,625)
    train_n = np.append(train_n,-1)
    list_of_pics_no.append(train_n)
list_of_pics_no = np.asarray(list_of_pics_no)


# y_test dataset labeling(-1)

list_of_pics_ytest = list_of_pics_yes[0:300]



# # no_test labeling(-1)

list_of_pics_ntest = list_of_pics_no[0:300]


# test_dataset = np.concatenate((list_of_pics_ntest,list_of_pics_ytest),axis = 0)

# train dataset and labels dataset

train_dataset = np.concatenate((list_of_pics_no,list_of_pics_yes),axis = 0)
x_labels = train_dataset[:,-1]
y_labels = test_dataset[:,-1]

# finding distances(euclidine)

tot = []
for t in (np.delete(test_dataset,-1,1)):
    ms= []
    for j in (np.delete(train_dataset,-1,1)):
        Sy = np.sqrt(np.sum(np.square(t-j)))
        ms.append(Sy)
        np.asarray(ms)
    tot.append(ms)
tot = np.asarray(tot)

# sorting the indicies and getting the relative labels
sortedd = np.argsort(tot)
predicted_dist_labels= []
for i in sortedd:
    
    predicted_dist_labels.append(x_labels[i])

# knn =11 and voting about no or yes based on the the majority
k = 7
yes_lists =[]
no_lists = []
for f in range(640):
    yes =np.count_nonzero((predicted_dist_labels[f][0:k])==1)  
    yes_lists.append(yes)
    no  =np.count_nonzero((predicted_dist_labels[f][0:k])==-1)
    no_lists.append(no)
# print(yes_lists)
count_no = 0
count_yes= 0

y_pre = []
for b,d in zip(no_lists,yes_lists):
    if b>d:
        count_no+=1
        y_pre.append(-1)
    elif b ==d:
        count_yes+=1
        y_pre.append(-1)
    else:
        count_yes+=1
        y_pre.append(1)


# conf_matrix

def conf(y_pre, y_tes):
  con_matrix = np.zeros((2, 2), dtype=int)
  for x in range(len(y_pre)):
    if y_pre[x] == -1 and y_tes[x] == -1:
      con_matrix[1][1] += 1  # true negative
    elif y_pre[x] == -1 and y_tes[x] == 1:
      con_matrix[1][0] += 1  # false negative
    elif y_pre[x] == 1 and y_tes[x] == -1:
      con_matrix[0][1] += 1  # false positive
    else:
      con_matrix[0][0] += 1  # true positive

  return np.asarray(con_matrix)


# accuracy

def accuracy(con_matrix):
  return ((con_matrix[0][0] + con_matrix[1][1]) * 100) / (np.sum(con_matrix))

a = conf(y_pre, y_labels)
print('confusion matrix =','\n', a)    
print('accuracy = ','\n',accuracy(a),'%')

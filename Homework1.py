#!/usr/bin/env python
# coding: utf-8

# # Homework 1 Template
# This is the template for the first homework assignment.
# Below are some function templates which we require you to fill out.
# These will be tested by the autograder, so it is important to not edit the function definitions.
# The functions have python docstrings which should indicate what the input and output arguments are.

# ## Instructions for the Autograder
# When you submit your code to the autograder on Gradescope, you will need to comment out any code which is not an import statement or contained within a function definition.

# In[9]:


# # Uncomment and run this code if you want to verify your `sklearn` installation.
# # If this cell outputs 'array([1])', then it's installed correctly.

# from sklearn import tree
# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf = clf.fit(X, y)
# clf.predict([[2, 2]])


# In[192]:


# Uncomment this code to see how to visualize a decision tree. This code should
# be commented out when you submit to the autograder.
# If this cell fails with
# an error related to `pydotplus`, try running `pip install pydotplus`
# from the command line, and retry. Similarly for any other package failure message.
# If you can't get this cell working, it's ok - this part is not required.
#
# This part should be commented out when you submit it to Gradescope

#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#import pydotplus
#
#dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True,
#                feature_names=['feature1', 'feature2'],
#                class_names=['0', '1'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())


# In[193]:


# This code should be commented out when you submit to the autograder.
# This cell will possibly download and unzip the dataset required for this assignment.
# It hasn't been tested on Windows, so it will not run if you are running on Windows.

#import os
#
#if os.name != 'nt':  # This is the Windows check
#    if not os.path.exists('badges.zip'):
#        # If your statement starts with "!", then the command is run in bash, not python
#        !wget https://www.seas.upenn.edu/~cis519/fall2018/assets/HW/HW1/badges.zip
#        !mkdir -p badges
#        !unzip badges.zip -d badges
#        print('The data has saved in the "badges" directory.')
#else:
#    print('Sorry, I think you are running on windows. '
#          'You will need to manually download the data')


# In[388]:


import numpy as np
import string

def compute_features(name):
    """
    Compute all of the features for a given name. The input
    name will always have 3 names separated by a space.
    
    Args:
        name (str): The input name, like "bill henry gates".
    Returns:
        list: The features for the name as a list, like [0, 0, 1, 0, 1].
    """
    first, middle, last = name.split()
    split_name =[first,middle,last]
    char_arr=[]
    feature=[]
#X_data=pd.DataFrame(x)
#Y_data=pd.DataFrame(y)
#cont = 'bill- henry gates' # put your data X in this 
    
    for i in range(3) :
        if len(split_name[i])==5 :
            for j in range(5):
                if split_name[i][j].isalpha():
                    index=ord(split_name[i][j])-97
                    char_arr=np.zeros(26)
                    char_arr[index]=1   
                    char_arr=char_arr.tolist()
                    feature =feature+char_arr
                else:
                    char_arr=np.zeros(26)
                    char_arr=char_arr.tolist()
                    feature =feature+char_arr
        else:
            for j in range(5):
                char_arr=np.zeros(26)
                char_arr=char_arr.tolist()
                feature =feature+char_arr
    return feature


# In[390]:


from sklearn.tree import DecisionTreeClassifier

# The `max_depth=None` construction is how you specify default arguments
# in python. By adding a default argument, you can call this method in a couple of ways:
#     
#     train_decision_tree(X, y)
#     train_decision_tree(X, y, 4) or train_decision_tree(X, y, max_depth=4)
#
# In the first way, max_depth is automatically set to `None`, otherwise it is 4.

# part (iii) 
def train_decision_tree(X, y, max_depth=None):
    """
    Trains a decision tree on the input data using the information gain criterion
    (set the criterion in the constructor to 'entropy').
    
    Args:
        X (list of lists): The features, which is a list of length n, and
                           each item in the list is a list of length d. This
                           represents the n x d feature matrix.
        y (list): The n labels, one for each item in X.
        max_depth (int): The maximum depth the decision tree is allowed to be. If
                         `None`, then the depth is unbounded.
    Returns:
        DecisionTreeClassifier: the learned decision tree.
        """
    clf = DecisionTreeClassifier(criterion='entropy',max_depth=max_depth,random_state=11)
    learnedtree_dec = clf.fit(X, y)
    return(learnedtree_dec)

# #part (iv)
# train_decision_tree(X,y,4)

# #part (v)
# train_decision_tree(X,y,8)


# In[418]:


from sklearn.linear_model import SGDClassifier

def train_sgd(X, y, learning_rate='optimal'):
    """
    Trains an `SGDClassifier` using 'log' loss on the input data.
    
    Args:
        X (list of lists): The features, which is a list of length n, and
                           each item in the list is a list of length d. This
                           represents the n x d feature matrix.
        y (list): The n labels, one for each item in X.
        learning_rate (str): The learning rate to use. See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
    Returns:
        SGDClassifier: the learned classifier.
    """
    sgdclf = SGDClassifier(alpha=0.002,tol=1e-2,loss='log',learning_rate='optimal',random_state=0)
    learnedtree_sgd = sgdclf.fit(X,y)
    return(learnedtree_sgd)


# In[275]:



# from sklearn.linear_model import SGDClassifier

# def train_sgd(X, y, learning_rate='optimal'):
#     """
#     Trains an `SGDClassifier` using 'log' loss on the input data.
    
#     Args:
#         X (list of lists): The features, which is a list of length n, and
#                            each item in the list is a list of length d. This
#                            represents the n x d feature matrix.
#         y (list): The n labels, one for each item in X.
#         learning_rate (str): The learning rate to use. See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
#     Returns:
#         SGDClassifier: the learned classifier.
#     """
#     sgdclf = SGDClassifier(alpha = 0.002,loss='log',learning_rate='optimal',random_state=0)
#     learnedtree_sgd = sgdclf.fit(X, y)
#     return(learnedtree_sgd)



# In[419]:



def train_sgd_with_stumps(X, y):
    """
    Trains an `SGDClassifier` using 'log' loss on the input data. The classifier will
    be trained on features that are computed using decision tree stumps.
    
    This function will return two items, the `SGDClassifier` and list of `DecisionTreeClassifier`s
    which were used to compute the new feature set. If `sgd` is the name of your `SGDClassifier`
    and `stumps` is the name of your list of `DecisionTreeClassifier`s, then writing
    `return sgd, stumps` will return both of them at the same time.
    
    Args:
        X (list of lists): The features, which is a list of length n, and
                           each item in the list is a list of length d. This
                           represents the n x d feature matrix.
        y (list): The n labels, one for each item in X.
    Returns:
        SGDClassifier: the learned classifier.
        List[DecisionTree]: the decision stumps that were used to compute the features
                            for the `SGDClassifier`.
    """
    # This is an example for how to return multiple arguments
    # in python. If you write `a, b = train_sgd_with_stumps(X, y)`, then
    # a will be 1 and b will be 2.
    
    stumps = []
#     X_pred=[]
    prediction =[]
    X_new=[]
    X_out=[]
    X_pred=[]
    for i in range(200):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,shuffle=True)
        stump_tree = train_decision_tree(X_train, y_train,8)
        stumps.append(stump_tree)   
        X_out.append(stump_tree.predict(X))
#    for j in range(len(X_out)):
#        if X_out[j]=='+':
#           X_out[j]=1 
#        else:
#           X_out[j]=0
#        X_new=int(float(X_out[j])) 
#        X_pred.append(X_new)
   
    #prediction of 200 steps
#     for j in range(200):
#         prediction.append(predict_clf(clf,X_train))
#     prediction=np.transpose(X_pred)
    X_pred=np.transpose(np.asarray(X_out)).tolist()
#    prediction=np.transpose(X_pred).tolist()
   # prediction=float(prediction)
    sgd=train_sgd(X_pred, y, learning_rate='optimal')          
    return sgd, stumps


# In[420]:


# The input to this function can be an `SGDClassifier` or a `DecisionTreeClassifier`.
# Because they both use the same interface for predicting labels, the code can be the same
# for both of them.

def predict_clf(clf, X):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import numpy as np 
    import matplotlib.pyplot as plt
    import pandas as pd
    """
    Predicts labels for all instances in `X` using the `clf` classifier. This function
    will be the same for `DecisionTreeClassifier`s and `SGDClassifier`s.
    
    Args:
        clf: (`SGDClassifier` or `DecisionTreeClassifier`): the trained classifier.
        X (list of lists): The features, which is a list of length n, and
                           each item in the list is a list of length d. This
                           represents the n x d feature matrix.
    Returns:
        List[int]: the predicted labels for each instance in `X`.
    """
    return clf.predict(X)


# In[421]:


# The SGD-DT classifier can't use the same function as the SGD or decision trees
# because it requires an extra argument
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def predict_sgd_with_stumps(sgd, stumps, X):
    """
    Predicts labels for all instances `X` using the `SGDClassifier` trained with
    features computed from decision stumps. The input `X` will be a matrix of the
    original features. The stumps will be used to map `X` from the original features
    to the features that the `SGDClassifier` were trained with.
    
    Args:
        sgd (`SGDClassifier`): the classifier that was trained with features computed
                               using the input stumps.
        stumps (List[DecisionTreeClassifier]): a list of `DecisionTreeClassifier`s that
                                               were used to train the `SGDClassifier`.
        X (list of lists): The features that were used to train the stumps (i.e. the original
                           feature set).
    Returns:
        List[int]: the predicted labels for each instance in `X`.
        
    """
    X_out=[]
    X_final=[]
    X_pred=[]
    X_new=[]
    for i in stumps:   
        X_out.append(i.predict(X))
    X_pred = np.transpose(np.asarray(X_out)) 
    X_new = X_pred.astype(int) 
    X_final=predict_clf(sgd, X_new)
    return X_final
    
    


# In[422]:


# # # # Write the rest of your code here. Anything from here down should be commented
# # # # out when you submit to the autograder
# # # #cross validation procedure 

# # # #calling train_sgd_With stumps 
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import numpy as np 
# import matplotlib.pyplot as plt
# import pandas as pd

# data=pd.read_csv('train.fold-0.txt',sep='\t', header= None)
# x0=data.iloc[:,1].values
# #x0=pd.DataFrame(x0)
# y0=data.iloc[:,0].values
# data=pd.read_csv('train.fold-1.txt',sep='\t' , header= None)
# x1=data.iloc[:,1].values
# y1=data.iloc[:,0].values
# data=pd.read_csv('train.fold-2.txt',sep='\t',header= None)
# x2=data.iloc[:,1].values
# y2=data.iloc[:,0].values
# data=pd.read_csv('train.fold-3.txt',sep='\t',header= None)
# x3=data.iloc[:,1].values
# y3=data.iloc[:,0].values
# data=pd.read_csv('train.fold-4.txt',sep='\t',header= None)
# x4=data.iloc[:,1].values
# y4=data.iloc[:,0].values
# # Define/declare variables 
# Names=[x0,x1,x2,x3,x4]
# labels=[y0,y1,y2,y3,y4]
# labels_tot=[]
# feature_tot=[]
# feature_test=[]

# test_list=[]
# feature_label_test=[]
# train_input=[]
# train_label=[]
# test_input=[]
# feature_label=[]
# feature_test=[]
# feature_label_y=[]
# test_name_x=[]
# test_label_y=[]
# Train_tree=[]
# Train_sgd=[]
# Test_tree=[]
# Test_sgd=[] 
# final_label=[]
# stump_test=[]
# stump_train=[]
# final_label_test=[]
# label_in=[]
# Train_tree4=[]
# Train_tree8=[]
# Test_tree4=[]
# Test_tree8=[]
# accuracy=[]

# for i in range(5):
#     test_name=Names.pop(i)
#     test_labels=labels.pop(i)
#     for u in range(4):    
#         for j in range(len(x0)):
#             feature_tot.append(compute_features(Names[u][j]))   
#         feature_label=labelconversion(labels[u])
        
#         train_label=train_label +feature_label
        
#         train_input=train_input+feature_tot
#         feature_tot=[]
#         label_in=[]
#         feature_label=[]
#     sdgclassifier=train_sgd(train_input,train_label,learning_rate='optimal')
#     decisionclassifier=train_decision_tree(train_input, train_label)
#     pred_y_1=predict_clf(decisionclassifier,train_input)
#     pred_y_2=predict_clf(sdgclassifier,train_input)
#     Train_tree.append(accuracy_score(train_label,pred_y_1))
#     Train_sgd.append((accuracy_score(train_label,pred_y_2)))
    
#     decclass4=train_decision_tree(train_input, train_label,max_depth=4)
#     decclass8=train_decision_tree(train_input, train_label,max_depth=8)
#     pred_1=predict_clf(decclass4,train_input)
#     pred_2=predict_clf(decclass8,train_input)
#     Train_tree4.append(accuracy_score(train_label,pred_1))
#     Train_tree8.append(accuracy_score(train_label,pred_2))
    
#     for v in range(len(test_name)):
#         feature_test.append(compute_features(test_name[v])) 
# #        
#    # test_name_x=test_name_x+feature_test
#     feature_label_y=labelconversion(test_labels)
   
#     pred_y_3=predict_clf(decisionclassifier,feature_test)
#     #print(pred_y_3)
#     pred_y_4=predict_clf(sdgclassifier,feature_test)
   
#     Test_tree.append(accuracy_score(feature_label_y,pred_y_3))
#     Test_sgd.append(accuracy_score(feature_label_y,pred_y_4))
    
#     pred_y_5=predict_clf(decclass4,feature_test)
#     pred_y_6=predict_clf(decclass8,feature_test)
#     Test_tree4.append(accuracy_score(feature_label_y,pred_y_5))
#     Test_tree8.append(accuracy_score(feature_label_y,pred_y_6))
    
#     sgdd,stumpss=train_sgd_with_stumps(train_input,train_label)
#     final_label=predict_sgd_with_stumps(sgdd,stumpss,train_input)
  
#     final_label_test=predict_sgd_with_stumps(sgdd,stumpss,feature_test)
#     stump_train.append(accuracy_score(train_label,final_label))
#     stump_test.append(accuracy_score(feature_label_y,final_label_test))
#     Names.insert(i,test_name)
#     labels.insert(i,test_labels)
#     train_input=[]
#     train_label=[]
#     feature_test=[]
#     feature_label_y=[]
#     #test_labels=[]
#     #test_name=[]
#     #test_labels=[]
#     #test_label_y=[]
#     #print('Test name X is')

# print(np.mean(Test_tree))
# print(np.mean(Test_sgd))
# print(np.mean(Test_tree4))
# print(np.mean(Test_tree8))
# print(np.mean(stump_test))
 


# In[ ]:





# In[423]:


# def labelconversion(y):
#     ynew=[]
#     a=np.asarray(y)
#     a[a=='+']=1
#     a[a=='-']=0
#     ynew=a.tolist()
#     return ynew


# In[424]:


# import pandas as pd 

# train = pd.read_csv('train.txt',sep= '\t', header = None)
# test = pd.read_csv('test.unlabeled.txt')

# X=[]
# x_name = train.iloc[:,1].values.tolist()
# test_data= test.iloc[:,0].values.tolist()
# print(len(test_data))
# y_label= train.iloc[:,0].values.tolist()
# y=labelconversion(y_label)

# for o in range(len(x_name)):
#     X.append(compute_features(x_name[o]))


# features_test_u= []
# for j in range(len(test_data)):
#     features_test_u.append(compute_features(test_data[j]))


# model_tree = train_decision_tree(X, y)
# model_sgd  = train_sgd(X, y)
# model_4dec4 = train_decision_tree(X, y,max_depth=4)
# model_8dec8 = train_decision_tree(X, y,max_depth=8)
# model_stumps,stumps = train_sgd_with_stumps(X,y)

# pred_y = predict_clf(model_tree, features_test_u)

# pred_dec8= predict_clf(model_8dec8,features_test_u)

# pred_dec4= predict_clf(model_4dec4,features_test_u)

# pred_sgd= predict_clf(model_sgd,features_test_u)

# pred_stumps = predict_sgd_with_stumps(model_stumps, stumps,features_test_u)

# with open('sgd.txt' , 'w',encoding= 'utf8') as file:
#     for k in range(len(pred_sgd)):
#         if pred_sgd[k]=='0':
#             file.write("%s\n"%'-')
            
#         elif pred_sgd[k]=='1':
#              file.write("%s\n"%'+')
            

# with open('sgd-dt.txt' , 'w',encoding= 'utf8') as file:
#     for k in range(len(pred_stumps)):
#         if pred_stumps[k]=='0':
#             file.write("%s\n"%'-')
            
            
#         elif pred_stumps[k] =='1':
#              file.write("%s\n"%'+')
            
        

# with open('dt.txt' , 'w',encoding= 'utf8') as file:
#      for k in range(len(pred_y)):
#         if pred_y[k]=='0':
#             file.write("%s\n"%'-')
            
            
#         elif pred_y[k] =='1':
#              file.write("%s\n"%'+')
                  
# with open('dt-8.txt' , 'w',encoding= 'utf8') as file:
#      for k in range(len(pred_dec8)):
#         if  pred_dec8[k]=='0':
#              file.write("%s\n"%'-')         
            
#         elif pred_dec8[k] =='1':
#               file.write("%s\n"%'+')
            
# with open('dt-4.txt' , 'w',encoding= 'utf8') as file:
#     for k in range(len(pred_dec4)):
#         if pred_dec4[k]=='0':
#              file.write("%s\n"%'-')           
#         elif pred_dec4[k]=='1':
#              file.write("%s\n"%'+')
            


# In[205]:


# import pandas as pd
# train = pd.read_csv('train.txt',sep= '\t', header = None)
# test = pd.read_csv('test.unlabeled.txt')

# x = train.iloc[:,1].values.tolist()
# test_x= test.iloc[:,0].values.tolist()
# y = train.iloc[:,0].values.tolist()


# In[ ]:





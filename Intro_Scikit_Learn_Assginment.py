# -*- coding: utf-8 -*-
"""
#Purpose: Below application code is designed to read csv data file and perform
          classification using methods of scikit-learn module  . 
#Process: 1. Read training,test and target data.
          2. Divide the training data in two parts.Train the model on first part and
             then verify the model on second data. If everything goes fine move to next
             step.
          3. Do PCA Evaluation for dimensionality reduction.
          4. Apply KNN classification algorithm.

#Indicators: First 12 variables.
#Author:  Aakash Parwani
#Date:    25th Jan 2017

"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA

############################## Section 1: Read data ###################################################

working_directory =r"D:\Aakash_Documents\MS_Collections\AcceptanceFromSaintPeters\ClassStuff\DS_680_MrktgAnalytic\KaggleCompetitions\ScikitLearn_Compt1\Data\\"

#fetch training data
train_data = np.genfromtxt(open(working_directory + 'train.csv','rb'), delimiter=',')

#fetch target variable data
target_data = np.genfromtxt(open(working_directory + 'trainLabels.csv','rb'), delimiter=',')

#fetch testing data
test_data = np.genfromtxt(open(working_directory + 'test.csv','rb'), delimiter=',')

############################################################################################

############################## Section 2: Define functions #################################

"""
function 1: This funtion will divide the training data in 2 parts (80 & 20 percent).
            Model will be trained on first part and validation will be performed on
            second part. 
            Basically it will help us to understand whether selected model will 
            perform better on final test data or not.
"""
def dividedata(train_data,target_data):
    train,test,train_target,test_target = train_test_split(train_data,target_data,test_size=0.2,random_state=42)
    return train,test,train_target,test_target


"""
function 2: This function will help us to perform classification on data. Here we 
            are using KNN algorithm (supervised learning) which is a common 
            classification scheme and distance measure parameter is 12.
"""
def Knnclassifier(train,test,train_target,test_target):
    kclass = KNeighborsClassifier(n_neighbors=12,algorithm='kd_tree',weights=
                                    'uniform',p=1)
    kclass.fit(train,train_target)
    res = kclass.predict(train)
    print classification_report(train_target,res)
    res1 = kclass.predict(test)
    print classification_report(test_target,res1)
    return kclass
    
"""
function 3: In this function we will do Principal Component Analysis(PCA) it is very
            important function because if we will check our dataset there are around 
            40 features or variables. And not every variable is that important or
            able to define the target variable. In such cases it becomes important
            to first recognize important variables that can define the maximum variance.
            Basically, this function will help us in Dimensionality Reduction.
"""
def dopca(train,train_target,test,test_data):
    pca = PCA(n_components=12,whiten=True)
    train = pca.fit_transform(train,train_target)
    test = pca.transform(test)
    test_data = pca.transform(test_data)
    return train,test,test_data
############################################################################################    

##################### Section 3: use functions and predict the output #######################
train,test,train_target,test_target = dividedata(train_data,target_data)

train,test,test_data = dopca(train,train_target,test,test_data)

est =Knnclassifier(train,test,train_target,test_target)

res = est.predict(test_data)
idcol = np.arange(start=1,stop=9001)
res2 = np.column_stack((idcol,res))

np.savetxt(working_directory + 'prediction.csv',res2,fmt='%d',delimiter=",")
############################################################################################    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 17:04:21 2023

@author: wavbrain
"""


from scipy.stats import entropy
from sklearn.decomposition import PCA
from sklearn import decomposition
from threading import Thread
from Utils import normalize_segments,FeatureExtractor_All,ModelFitter_OneClassSVM,DimensionReductor_PCA,DataOrganizer_RandomCrop,ModelRuntime_OneClassSVM, ModelDesigner_OneClassSVM

from sklearn import svm
import os
import os.path
from os import walk
import csv
import numpy as np
import pickle as pk
from joblib import dump, load
from matplotlib import pyplot as plt



def csvToWindow(fname,size):
    """
    This function is used to import raw data from csv files (especially for learning)
    """
    i=0
    window=[]
    windowSegment = []        
    with open(fname, newline='') as f:            
        csvreader=csv.reader(f, delimiter=',', quotechar='|')
        for row in csvreader:
            i+=1
            window.append(float(row[1]))
            if (i==size):
                windowSegment.append(window)
                window=[]
                i=0
    return windowSegment




def getSegments(ListFiles,SizeSegment,subjectDir):
    """
    This function is used to extracts data segments according to a segment size (SizeSegment)
    """
    Segments = []
    for fn in ListFiles:
        targetFile = Datadir+subjectDir+fn
        SS = csvToWindow(targetFile,SizeSegment) 
        Segments += SS
    
    return Segments

    

Datadir = './data/'
LearningDir = '/learning/'
TestDir = '/test/'

#size of a data segment
SizeSegment = 512
#parameter of the one class svm classifier
gamma=  0.3
#parameter of the one class svm classifier
nu = 0.1
#parameter of the one class svm classifier
kernel="rbf"
#number of principal components to keep
NC_pca = 2
#wavelet decomposition level
Dec_levels = 5


#get learning files to import raw data
ListFiles = os.listdir(Datadir+LearningDir)
ListFiles.sort()
#import raw data and decompose it into segments of size SizeSegment
Segments_learning = getSegments(ListFiles,SizeSegment,LearningDir)



#-------------- Learning -----------#
#normalize the segment data
Segments_normalized = normalize_segments(Segments_learning)
#initialize default labels  (nominal)
Labels = np.ones(len(Segments_normalized))
#extract features
Features = FeatureExtractor_All(Segments_normalized,'db3',Dec_levels)
#apply dimension reduction using principal component analysis
PCA_Features_learning,pca  = DimensionReductor_PCA(Features,None,NC_pca)
#reorganize data as learning and test sets 
Features_train, Features_test, Labels_Train, Labels_Test = DataOrganizer_RandomCrop(PCA_Features_learning,Labels,Percentage=0.2)
#create a one class SVM classifier with indicated parameters (kernel,nu,gamma)
classifier = ModelDesigner_OneClassSVM(kernel,nu,gamma)
#fit the classifier using training data and get fitted classifier
FittedClassifier = ModelFitter_OneClassSVM(Features_train,classifier)
#apply classifier to training data to get estimated labels 
labels = ModelRuntime_OneClassSVM(FittedClassifier,Features_train)
#caculate classification error
diff = sum(labels != Labels_Train)
error_rate_learning = diff / len(Labels_Train)

   
#---------------- test ------------#
#get test files to import raw data
ListFiles = os.listdir(Datadir+TestDir)
ListFiles.sort()
#import raw data and decompose it into segments of size SizeSegment
Segments_test = getSegments(ListFiles,SizeSegment,TestDir)
#normalize data
Segments_normalized = normalize_segments(Segments_test)
#extract features
Features = FeatureExtractor_All(Segments_normalized,'db3',Dec_levels)
PCA_Features,pca  = DimensionReductor_PCA(Features,pca,NC_pca)
#apply dimension reduction using principal component analysis
PCA_Features_all = np.concatenate([PCA_Features,Features_test])
#initialize test labels (anomaly)
Labels = -1*np.ones(len(Segments_test))
#merge anomaly labels with those for test (nominal) 
Labels = np.concatenate([Labels,Labels_Test])
#apply classifier to get estimated labels
labels_closed = ModelRuntime_OneClassSVM(FittedClassifier,PCA_Features_all)
#caculate classification error on the test set
diff = sum(labels_closed != Labels)
error_rate_closed = diff / len(Labels)




Prec_learn = 1 - error_rate_learning
Prec_test = 1 - error_rate_closed


print(100*Prec_learn)
print(100*Prec_test)











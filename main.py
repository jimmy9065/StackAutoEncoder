import sys
import os.path
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.io as sio

import sklearn.preprocessing as skp
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.decomposition import PCA

from AutoEncoder import NonLinearAE

def testSVM(data,labels):
    clf2 = svm.SVC(kernel='linear', C=1)
    clf3 = svm.SVC(kernel='rbf', C=1)

    scores = cross_val_score(clf2, data, labels, cv=10)
    print("Linear")
    print(scores)
    print(sum(scores)/10)

    scores = cross_val_score(clf3, data, labels, cv=10)
    print("RBF")
    print(scores)
    print(sum(scores)/10)

def testLogistic(data,labels):
    lg = LogisticRegression()
    scores = cross_val_score(lg, data, labels, cv=5)
    print("Logistic")
    print(scores,"mean=%0.3f"%sum(scores)/5)

def testRF(data,labels):
    rf=RF(100,n_jobs=-1)
    score=cross_val_score(rf,data,labels,cv=5)
    print(score)
    print(np.mean(score))

def normalize(data):
    minV=data.min(0)
    data -= minV
    maxV=data.max(0)
    data/=maxV
    return data

def main():
    np.random.seed(0)

    data=pd.read_pickle("data-knn.pk1").values
    labels=(data[:,-1]-1).tolist()
    data=data[:,:-1]

    pca=PCA()
    data=pca.fit_transform(data)
    print(data.shape)

    data=normalize(data)

    testLogistic(data,labels)

    t=time.time()
    encoding,w1=NonLinearAE(data,0,1,500,learning_rate=0.1,max_iter=1000,denoise=True,noise=0.2,Sparcity=False)
    testLogistic(encoding,labels)
    encoding,w1=NonLinearAE(encoding,0,1,300,learning_rate=0.1,max_iter=1000,denoise=False,noise=0.2,Sparcity=True)
    testLogistic(encoding,labels)
    encoding,w1=NonLinearAE(encoding,0,1,200,learning_rate=0.1,max_iter=1000,denoise=False,noise=0.2,Sparcity=True)
    testLogistic(encoding,labels)
    encoding,w1=NonLinearAE(encoding,0,1,100,learning_rate=0.1,max_iter=1000,denoise=False,noise=0.2,Sparcity=True)
    testLogistic(encoding,labels)
    elapsed=int(time.time()-t)
    testLogistic(encoding,labels)
    print('elapsed= %d min:%d seconds' % (int(elapsed/60),elapsed%60))

if __name__=="__main__":
    main()

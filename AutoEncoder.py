import sys
import os.path
import time

import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
import scipy.io as sio

import sklearn.preprocessing as skp
from sklearn import svm
from sklearn.model_selection import cross_val_score

n_hidden=1000

learning_rate=0.05
max_iter=2000
drate=500

build=True
test=True
matlab=True

def normalize(x):
    mean=tf.div(tf.reduce_sum(x,1,keep_dims=True),4)
    center=tf.subtract(x,mean)
    norm=tf.sqrt(tf.reduce_sum(tf.pow(center,2),1,keep_dims=True))
    normalize=tf.div(center,norm)
    return normalize

def PCCloss(X1,X2):
    bX1=normalize(X1)
    bX2=normalize(X2)

    tbX1=tf.expand_dims(bX1,1)
    tbX2=tf.expand_dims(bX2,2)
    tproduct=tf.batch_matmul(tbX1,tbX2)
    product=tf.squeeze(tproduct)
    loss=tf.negative(product)
    return loss

def getEncoder(data,learning_rate=0.1):
    n_features=data.shape[1]

    X=tf.placeholder("float", [None,n_features])

    #init parameter
    weights={
            'encoder_h':tf.Variable(tf.random_normal([n_features, n_hidden])),
            'decoder_h':tf.Variable(tf.random_normal([n_hidden, n_features])),
            }

    biases={
            'encoder_b':tf.Variable(tf.random_normal([n_hidden])),
            'decoder_b':tf.Variable(tf.random_normal([n_features])),
            }


    #build graph
    wi=weights['encoder_h']
    #Hlayer=tf.nn.sigmoid(tf.add(tf.matmul(X,wi), biases['encoder_b']))
    #Hlayer=tf.nn.relu(tf.add(tf.matmul(X,wi), biases['encoder_b']))
    Hlayer=tf.nn.bias_add(tf.matmul(X,wi), biases['encoder_b'])

    wo=tf.transpose(wi)
    #wo=weights['decoder_h']
    #X_pred=tf.nn.sigmoid(tf.add(tf.matmul(Hlayer,wo), biases['decoder_b']))
    X_pred=tf.nn.bias_add(tf.matmul(Hlayer,wo), biases['decoder_b'])

    cost=tf.reduce_mean(PCCloss(X,X_pred))
    #cost=tf.reduce_mean(tf.pow(X-X_pred,2))
    #optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
    optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    #init for session
    init=tf.global_variables_initializer()

    sess=tf.Session()
    sess.run(init)

    #run
    print("rate=",learning_rate)
    for epoch in range(max_iter):
        if (epoch+1 % drate==0):
            #learning_rate/=2.0
            print("rate=",learning_rate)
            #optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
            optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        _, c=sess.run([optimizer, cost],feed_dict={X:data})

        if epoch % 200 ==0:
            print("Epoch:","%04d" % (epoch+1),"cost=", c)#"{:9f}".format(c))
        #print("Epoch:","%04d" % (epoch+1),"cost=", c)#"{:9f}".format(c))

    print("Finished")

    encoded=sess.run(Hlayer,feed_dict={X:data})
    return encoded

def testSVM(data,odata,labels):
    clf2 = svm.SVC(kernel='linear', C=1)

    #print("[ 0.45945946  0.59459459  0.51351351  0.45945946  0.62162162  0.54054054  0.52777778  0.48571429  0.57142857  0.6]")
    print("0.5374")

    scores = cross_val_score(clf2, data, labels, cv=10)
    print(scores)
    print(sum(scores)/10)

def main():
    tf.set_random_seed(0)
    data=pd.read_pickle("data-knn.pk1").values
    labels=(data[:,-1]-1).tolist()
    #data=np.load('ndata.pk1')
    #labels=(data[:,-1]).tolist()
    data=data[:,:-1]

    #build=False
    test=False
    matlab=False

    if os.path.isfile('encoder.pk1') and not build:
        encoder=np.load('encoder.pk1')
    else:
        t=time.time()
        encoder=getEncoder(data,learning_rate)
        elapsed=time.time()-t
        print('elapsed=',elapsed)
        f=open('encoder.pk1','wb')
        np.save(f,encoder)
        f.close()

    if test:
        testSVM(encoder,data,labels)

    if matlab:
        D=np.column_stack([encoder,labels])
        sio.savemat('AE_features',{'D':D})

if __name__=="__main__":
    main()

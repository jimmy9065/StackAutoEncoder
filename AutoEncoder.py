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

n_hidden1=1024
max_iter1=2000

n_hidden=[1024,512,1024,512,256]
max_iter=[2000,4000,3000,4000,4000]

Max_depth=4

learning_rate1=0.5
learning_rate2=0.04

labels=[]

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

def testSVM(data,odata,labels):
    clf2 = svm.SVC(kernel='linear', C=1)

    #print("[ 0.45945946  0.59459459  0.51351351  0.45945946  0.62162162  0.54054054  0.52777778  0.48571429  0.57142857  0.6]")
    print("0.5374")

    scores = cross_val_score(clf2, data, labels, cv=10)
    print(scores)
    print(sum(scores)/10)

def NonLinearAE(data,depth):
    print("depth=",depth)
    n_features=data.shape[1]
    X=tf.placeholder(dtype=tf.float32, shape=(None,n_features))
    dropX=tf.nn.dropout(X,0.1) #adding noise

    w_encoder=tf.Variable(tf.random_normal([n_features, n_hidden[depth]]))
    w_decoder=tf.Variable(tf.random_normal([n_hidden[depth], n_features]))
    b_encoder=tf.Variable(tf.random_normal([n_hidden[depth]]))
    b_decoder=tf.Variable(tf.random_normal([n_features]))

    #hidden=tf.nn.relu(tf.nn.bias_add(tf.matmul(dropX,w_encoder), b_encoder))
    #y_pred=tf.nn.relu(tf.nn.bias_add(tf.matmul(hidden,w_decoder), b_decoder))
    hidden=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(dropX,w_encoder), b_encoder))
    y_pred=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden,w_decoder), b_decoder))
    y_true=X

    #change cost
    #cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred,y_true))
    #cost=tf.reduce_mean(tf.pow(y_true-y_pred,2))
    cost=tf.reduce_mean(PCCloss(y_true,y_pred))
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate1).minimize(cost)
    optimizer=tf.train.AdamOptimizer(learning_rate2).minimize(cost)

    # start to run
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    for epoch in range(max_iter[depth]):
        batches=divideData(data)
        co=0
        for batch in batches:
            _, c=sess.run([optimizer, cost],feed_dict={X:batch})
            co+=c
        co/=3

        if epoch % 200 ==0:
            print("Depth: ", depth, " Epoch:", "%04d" % (epoch+1),"cost=", co)#"{:9f}".format(co))


    print("finish training")
    encoded=sess.run(hidden,feed_dict={dropX:data})

    testSVM(encoded,None,labels)
    print("extract hidden in this layer=",depth)
    w=sess.run(w_encoder)
    print("extract weights in this layer=",depth)

    if depth<Max_depth:
        print("go deeper",depth)
        encoded,ws=NonLinearAE(encoded,depth+1)
    else:
        print("get to the bottom, layer=",depth)
        ws=[]

    ws.append(w)

    print("exit this layer=",depth)
    return encoded,ws

def LinearAE(data):
    n_features=data.shape[1]
    X=tf.placeholder(dtype=tf.float32, shape=(None,n_features))
    dropX=tf.nn.dropout(X,0.1) #adding noise

    w_encoder=tf.Variable(tf.random_normal([n_features, n_hidden[0]]))
    w_decoder=tf.transpose(w_encoder)
    b_encoder=tf.Variable(tf.random_normal([n_hidden[0]]))
    b_decoder=tf.Variable(tf.random_normal([n_features]))

    hidden=tf.nn.bias_add(tf.matmul(dropX,w_encoder), b_encoder)
    y_pred=tf.nn.bias_add(tf.matmul(hidden,w_decoder), b_decoder)
    y_true=X

    cost=tf.reduce_mean(PCCloss(y_true,y_pred))
    #cost=tf.reduce_mean(tf.pow(y_true-y_pred, 2))
    optimizer=tf.train.GradientDescentOptimizer(learning_rate1).minimize(cost)

    # start to run
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    for epoch in range(max_iter[0]):
        batches=divideData(data)
        co=0
        for batch in batches:
            _, c=sess.run([optimizer, cost],feed_dict={X:batch})
            co+=c
        co/=3

        if epoch % 200 ==0:
            print("Epoch:","%04d" % (epoch+1),"cost=", "{:9f}".format(co))

    print("Finished")

    encoded=sess.run(hidden,feed_dict={dropX:data})

    e,ws=NonLinearAE(encoded,1)
    w=sess.run(w_encoder)
    ws.append(w)

    return e,ws

def divideData(D):
    index=np.arange(D.shape[0])
    np.random.shuffle(index)
    D=D[index,:]
    batches=np.split(D,3)
    return batches

def main():
    global labels

    tf.set_random_seed(0)
    np.random.seed(0)
    data=pd.read_pickle("data-knn.pk1").values
    labels=(data[:,-1]-1).tolist()
    #data=np.load('ndata.pk1')
    #labels=(data[:,-1]).tolist()

    data=data[:,:-1]

    #build=False
    #test=False
    matlab=False

    if (not build) and os.path.isfile('encoder.pk1'):
        encoder=np.load('encoder.pk1')[0]
    else:
        t=time.time()
        encoder,weights=LinearAE(data)
        #encoder,weights=NonLinearAE(data,0)
        elapsed=int(time.time()-t)
        print('elapsed= %d min:%d seconds' % (int(elapsed/60),elapsed%60))
        f=open('encoder.pk1','wb')
        np.save(f,[encoder,weights])
        f.close()

    if test:
        testSVM(encoder,data,labels)

    if matlab:
        D=np.column_stack([encoder,labels])
        sio.savemat('AE_features',{'D':D})

if __name__=="__main__":
    main()

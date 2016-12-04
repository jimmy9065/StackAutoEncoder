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

n_hidden=500

learning_rate=0.1
max_iter=5000

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
    loss=tf.mul(tf.constant(-1.0,dtype=tf.float32),
                            tf.reduce_sum(product))
    return loss

def getEncoder(data,learning_rate=0.1):
    n_features=data.shape[1]

    X=tf.placeholder("float", [None,n_features])

    weights={
            'encoder_h':tf.Variable(tf.random_normal([n_features, n_hidden])),
            'decoder_h':tf.Variable(tf.random_normal([n_hidden, n_features])),
            }

    biases={
            'encoder_b':tf.Variable(tf.random_normal([n_hidden])),
            'decoder_b':tf.Variable(tf.random_normal([n_features])),
            }

    def encoder(x):
        layer=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h']),
                                                biases['encoder_b']))

        return layer

    def decoder(x):
        layer=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h']),
                                                biases['decoder_b']))

        return layer

    encoder_op=encoder(X)
    decoder_op=decoder(encoder_op)

    y_pred=decoder_op
    y_true=X

    cost=tf.reduce_min(PCCloss(y_true,y_pred))
    #optimizer=tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
    optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init=tf.global_variables_initializer()

    #sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess=tf.Session()
    sess.run(init)

    print("rate=",learning_rate)
    for epoch in range(max_iter):
        if (epoch+1 % 500==0):
            learning_rate/=2.0
            print("rate=",learning_rate)
            optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

        _, c=sess.run([optimizer, cost],feed_dict={X:data})

        if epoch % 100 ==0:
            print("Epoch:","%03d" % (epoch+1),"cost=", c)#"{:9f}".format(c))

    print("Finished")

    encode_decode=sess.run(y_pred,feed_dict={X:data})
    return encode_decode

def testSVM(data,odata,labels):
    #clf1 = svm.SVC(kernel='linear', C=1)
    #scores = cross_val_score(clf1, odata, labels, cv=10)

    clf2 = svm.SVC(kernel='linear', C=1)

    #print(scores)
    #print(sum(scores)/10)
    print("[ 0.45945946  0.59459459  0.51351351  0.45945946  0.62162162  0.54054054  0.52777778  0.48571429  0.57142857  0.6]")
    print("0.537410982411")

    scores = cross_val_score(clf2, data, labels, cv=10)
    print(scores)
    print(sum(scores)/10)

def main():
    tf.set_random_seed(0)
    data=pd.read_pickle("data-knn.pk1").values
    labels=(data[:,-1]-1).tolist()
    data=data[:,:-1]

    build=False
    build=True
    test=True
    #test=False
    matlab=True
    #matlab=False

    if os.path.isfile('encoder.pk1') and not build:
        encoder=np.load('encoder.pk1')
    else:
        t=time.time()
        encoder=getEncoder(data)
        elapsed=time.time()-t
        print('elapsed=',elapsed)
        f=open('encoder.pk1','wb')
        np.save(f,encoder,learning_rate)
        f.close()

    if test:
        testSVM(encoder,data,labels)

    if matlab:
        D=np.stack_column([encoder,labels])
        sio.savemat('AE_features',{'D':D})

if __name__=="__main__":
    main()

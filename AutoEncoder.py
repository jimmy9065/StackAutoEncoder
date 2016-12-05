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

n_hidden1=4048

n_hidden=[1024,512]

learning_rate1=0.5
learning_rate2=0.5
max_iter1=2000

depth=1


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

def LinearAE(data):
    n_features=data.shape[1]
    X=tf.placeholder(dtype=tf.float32, shape=(None,n_features))
    dropX=tf.nn.dropout(X,0.1)

    w_encoder=tf.Variable(tf.random_normal([n_features, n_hidden1]))
    w_decoder=tf.transpose(w_encoder)
    b_encoder=tf.Variable(tf.random_normal([n_hidden1]))
    b_decoder=tf.Variable(tf.random_normal([n_features]))

    #add noise
    hidden=tf.nn.bias_add(tf.matmul(dropX,w_encoder), b_encoder)
    y_pred=tf.nn.bias_add(tf.matmul(hidden,w_decoder), b_decoder)
    y_true=X

    cost=tf.reduce_mean(PCCloss(y_true,y_pred))
    optimizer=tf.train.GradientDescentOptimizer(learning_rate1).minimize(cost)

    # start to run
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    for epoch in range(max_iter1):
        batches=divideData(data)
        co=0
        for batch in batches:
            _, c=sess.run([optimizer, cost],feed_dict={X:batch})
            co+=c
        co/=3

        if epoch % 200 ==0:
            print("Epoch:","%03d" % (epoch+1),"cost=", "{:9f}".format(co))

    print("Finished")

    encoded=sess.run(hidden,feed_dict={dropX:data})

    return encoded

def testSVM(data,odata,labels):
    clf2 = svm.SVC(kernel='linear', C=1)

    #print("[ 0.45945946  0.59459459  0.51351351  0.45945946  0.62162162  0.54054054  0.52777778  0.48571429  0.57142857  0.6]")
    print("0.5374")

    scores = cross_val_score(clf2, data, labels, cv=10)
    print(scores)
    print(sum(scores)/10)

def divideData(D):
    index=np.arange(D.shape[0])
    np.random.shuffle(index)
    D=D[index,:]
    batches=np.split(D,3)
    return batches

def main():
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
        encoder=np.load('encoder.pk1')
    else:
        t=time.time()
        encoder=LinearAE(data)
        elapsed=int(time.time()-t)
        print('elapsed= %d min:%d seconds' % (int(elapsed/60),elapsed%60))
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

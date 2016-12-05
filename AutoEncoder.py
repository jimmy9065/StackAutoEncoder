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

n_hidden1=1000

learning_rate1=0.5
max_iter1=2000

build=True
#test=True
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

def getEncoder(data):
    n_features=data.shape[1]

    X=tf.placeholder(dtype=tf.float32, shape=(None,n_features))

    weights={
            'encoder_h1':tf.Variable(tf.random_normal([n_features, n_hidden1])),
            }

    biases={
            'encoder_b1':tf.Variable(tf.random_normal([n_hidden1])),
            'decoder_b1':tf.Variable(tf.random_normal([n_features])),
            }

    #first stack
    def encoder1(x):
        w_encoder1=weights['encoder_h1']
        layer=tf.nn.bias_add(tf.matmul(x,w_encoder1), biases['encoder_b1'])

        return layer

    def decoder1(x):
        w_decoder1=tf.transpose(weights['encoder_h1'])
        layer=tf.nn.bias_add(tf.matmul(x,w_decoder1), biases['decoder_b1'])

        return layer

    hidden1=encoder1(X)
    y1=decoder1(hidden1)

    y_true=X

    cost1=tf.reduce_mean(PCCloss(y_true,y1))
    #optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)
    print("learning_rate1=",learning_rate1)
    optimizer1=tf.train.GradientDescentOptimizer(learning_rate1).minimize(cost1)

    #for other layers to configed

    # start to run
    init=tf.global_variables_initializer()

    sess=tf.Session()
    sess.run(init)

    #first layer
    print("rate=",learning_rate1)
    for epoch in range(max_iter1):
        batches=divideData(data)
        co=0
        for batch in batches:
            _, c=sess.run([optimizer1, cost1],feed_dict={X:batch})
            co+=c

        co/=3
        if epoch % 200 ==0:
            print("Epoch:","%03d" % (epoch+1),"cost=", "{:9f}".format(co))

    #for the other layers

    print("Finished")

    encoded=sess.run(hidden1,feed_dict={X:data})
    #also provide the w_matrix
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

    data=data[:,:-1]

    #build=False
    #test=False
    matlab=False

    if (not build) and os.path.isfile('encoder.pk1'):
        encoder=np.load('encoder.pk1')
    else:
        t=time.time()
        encoder=getEncoder(data)
        elapsed=int(time.time()-t)
        print('elapsed= %d min:%d seconds' % (int(elapsed/60),elapsed%60))
        #f=open('encoder.pk1','wb')
        #np.save(f,encoder)
        #f.close()

    if test:
        testSVM(encoder,data,labels)

    if matlab:
        D=np.column_stack([encoder,labels])
        sio.savemat('AE_features',{'D':D})

if __name__=="__main__":
    main()

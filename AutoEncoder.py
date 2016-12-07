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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

n_hidden=[2000,4000,2000,1000,4000,2000,1000]
#max_iter=[2000,2000,2000,2000,2000,2000,2000]
max_iter=[500,500,500,500,500,500,500]
learning_rate=[0.05,0.05,0.05,0.05,0.05,0.05,0.05]

labels=[]

log_file=open('log','w')

def divideData(D):
    index=np.arange(D.shape[0])
    np.random.shuffle(index)
    D=D[index,:]
    batches=np.split(D,3)
    return batches

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

def addNoise(Data,rate):#X1 is the target, X2 is the output
    x,y=Data.shape
    prob=np.random.rand(x,y)
    mask=prob<rate*1
    return mask

def testSVM(data,labels):
    clf2 = svm.SVC(kernel='linear', C=1)
    clf3 = svm.SVC(kernel='rbf', C=1)

    scores = cross_val_score(clf2, data, labels, cv=10)
    print("Linear")
    print(scores)
    print(sum(scores)/10)

    log_file.write("Linear")
    log_file.write(str(scores)+"\n")
    log_file.write(str(sum(scores)/10)+"\n")

    scores = cross_val_score(clf3, data, labels, cv=10)
    print("RBF")
    print(scores)
    print(sum(scores)/10)

    log_file.write("RBF")
    log_file.write(str(scores)+"\n")
    log_file.write(str(sum(scores)/10)+"\n")

def testLogistic(data,labels,dim):
    lg = LogisticRegression()
    scores = cross_val_score(lg, data, labels, cv=10)
    print("Logistic")
    print(scores)
    print(sum(scores)/10)

    log_file.write("Logistic")
    log_file.write(str(scores)+"\n")
    log_file.write(str(sum(scores)/10)+"\n")

def NonLinearAE(data,depth,denoise=True,layers=1):
    global learning_rate2
    print("depth=",depth,"#hidden neuron=",n_hidden[depth])
    log_file.write("depth="+str(depth)+"\n")
    log_file.write("#hidden neuron="+str(n_hidden[depth])+"\n")
    n_features=data.shape[1]

    X=tf.placeholder(dtype=tf.float32, shape=(None,n_features))
    X_mask=tf.placeholder(dtype=tf.float32, shape=(None,n_features))

    if denoise:
        dropX=tf.multiply(X,X_mask)
    else:
        dropX=X

    if layers==1:
        w_encoder=tf.Variable(tf.random_normal([n_features, n_hidden[depth]]))
        w_decoder=tf.Variable(tf.random_normal([n_hidden[depth], n_features]))
        b_encoder=tf.Variable(tf.random_normal([n_hidden[depth]]))
        b_decoder=tf.Variable(tf.random_normal([n_features]))

        hidden=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(dropX,w_encoder), b_encoder))
        y_pred=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden,w_decoder), b_decoder))
    else:
        layers=n_hidden[depth]
        w_encoder1=tf.Variable(tf.random_normal([n_features, layers]))
        w_encoder2=tf.Variable(tf.random_normal([layers, layers/2]))
        w_decoder1=tf.Variable(tf.random_normal([layers/2, layers]))
        w_decoder2=tf.Variable(tf.random_normal([layers, n_features]))
        b_encoder1=tf.Variable(tf.random_normal([layers]))
        b_encoder2=tf.Variable(tf.random_normal([layers/2]))
        b_decoder1=tf.Variable(tf.random_normal([layers]))
        b_decoder2=tf.Variable(tf.random_normal([n_features]))

        hidden1=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(dropX,w_encoder1), b_encoder1))
        hidden2=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden1,w_encoder2), b_encoder2))
        hidden3=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden2,w_decoder1), b_decoder1))
        hidden4=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden3,w_decoder2), b_decoder2))
        y_pred=hidden4

    y_true=X

    #cost
    #cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred,y_true))
    cost=tf.reduce_mean(tf.square(y_true-y_pred))

    #optimizer
    optimizer=tf.train.GradientDescentOptimizer(learning_rate[depth]).minimize(cost)
    #optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate[depth]).minimize(cost)

    # start to run
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    for epoch in range(max_iter[depth]):
        batches=divideData(data)
        co=0
        for batch in batches:
            mask=addNoise(batch,0.05)
            _, c=sess.run([optimizer, cost],feed_dict={X:batch,X_mask:mask})
            co+=c
        co/=3

        learning_rate[depth]*=0.999
        optimizer=tf.train.GradientDescentOptimizer(learning_rate[depth]).minimize(cost)
        #optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate[depth]).minimize(cost)

        if epoch % 100 ==0:
            print("Depth: ", depth, " Epoch:", "%04d" % (epoch+1),"cost=", co)#"{:9f}".format(co))

    print("finish training")

    encoded=sess.run(hidden,feed_dict={dropX:data})
    w=sess.run(w_encoder)

    sess.close()
    print("start testing")
    testSVM(encoded,labels)
    testLogistic(encoded,labels,n_hidden[0])

    f=open("layer"+str(depth)+".pk1",'wb')
    np.save(f,encoded)
    f.close()

    return encoded,w

def StackAE(data):

    encoding,w1=NonLinearAE(data,0,denoise=True)
    w=w1

    encoding,w1=NonLinearAE(encoding,1,denoise=True, layers=2)
    w=np.matmul(w,w1)

    return
    encoding,w1=NonLinearAE(encoding,2,denoise=True)
    w=np.matmul(w,w1)

    encoding,w1=NonLinearAE(encoding,3,denoise=True)
    w=np.matmul(w,w1)

    encoding,w1=NonLinearAE(encoding,3,denoise=True)
    w=np.matmul(w,w1)

def main():
    global labels
    tf.set_random_seed(0)
    np.random.seed(0)

    data=pd.read_pickle("data-knn.pk1").values
    labels=(data[:,-1]-1).tolist()
    data=np.load('tdata.pk1')
    data=data[:,:-1]

    t=time.time()
    StackAE(data)
    elapsed=int(time.time()-t)
    print('elapsed= %d min:%d seconds' % (int(elapsed/60),elapsed%60))

if __name__=="__main__":
    main()

log_file.close()

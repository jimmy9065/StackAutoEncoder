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
from sklearn.ensemble import RandomForestClassifier as RF

n_hidden=[2000,4000,1000,500,4000,2000,1000]
#max_iter=[2000,2000,2000,2000,2000,2000,2000]
max_iter=[500,500,500,500,500,500,500,500,500]
#learning_rates=[0.2,0.1,0.05,0.05,0.05,0.05,0.05]

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

def testRF(data,labels):
    rf=RF(100,n_jobs=-1)
    score=cross_val_score(rf,data,labels,cv=5)
    print(score)
    print(np.mean(score))

def NonLinearAE(data,depth,layers,n_hidden,learning_rate=0.05,max_iter=500,denoise=False):
    print("depth=",depth,"#hidden neuron=",n_hidden," ",layers,"layers")
    log_file.write("depth="+str(depth)+"hidden"+str(n_hidden)+"\n")
    #log_file.write("#hidden neuron="+str(n_hidden)+" "+str(layers)+"layers"+"\n")
    n_features=data.shape[1]

    X=tf.placeholder(dtype=tf.float32, shape=(None,n_features))
    X_mask=tf.placeholder(dtype=tf.float32, shape=(None,n_features))

    if denoise:
        print("adding noise")
        dropX=tf.multiply(X,X_mask)
    else:
        print("no noise")
        dropX=X

    w_encoder=tf.Variable(tf.random_normal([n_features, n_hidden]))
    w_decoder=tf.Variable(tf.random_normal([n_hidden, n_features]))
    b_encoder=tf.Variable(tf.random_normal([n_hidden]))
    b_decoder=tf.Variable(tf.random_normal([n_features]))

    #w_encoder=tf.Variable(tf.zeros([n_features, n_hidden]))
    #w_decoder=tf.Variable(tf.zeros([n_hidden, n_features]))
    #b_encoder=tf.Variable(tf.zeros([n_hidden]))
    #b_decoder=tf.Variable(tf.zeros([n_features]))

    #hidden=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(dropX,w_encoder), b_encoder))
    #y_pred=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden,w_decoder), b_decoder))
    #hidden=tf.nn.relu(tf.nn.bias_add(tf.matmul(dropX,w_encoder), b_encoder))
    #y_pred=tf.nn.relu(tf.nn.bias_add(tf.matmul(hidden,w_decoder), b_decoder))
    hidden=tf.nn.sigmoid(tf.matmul(dropX,w_encoder))
    y_pred=tf.nn.sigmoid(tf.matmul(hidden,w_decoder))
    #hidden=tf.nn.relu(tf.matmul(dropX,w_encoder))
    #y_pred=tf.nn.relu(tf.matmul(hidden,w_decoder))

    y_true=X

    #cost
    #cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(y_pred,y_true))
    cost=tf.reduce_mean(tf.square(y_true-y_pred))

    nw=tf.sqrt(tf.reduce_sum(tf.square(w_encoder)))
    nb=tf.sqrt(tf.reduce_sum(tf.square(b_encoder)))

    #optimizer
    #optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost+nw*0.1/(n_features*n_hidden))
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer=tf.train.MomentumOptimizer(learning_rate,0.95).minimize(cost-nw*0.1)
    #optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # start to run
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    for epoch in range(max_iter):
        batches=divideData(data)
        co=0
        for batch in batches:
            mask=addNoise(batch,0.05)
            #_, c, nw=sess.run([optimizer, cost, nw],feed_dict={X:batch,X_mask:mask})
            _, c, w=sess.run([optimizer, cost, nw],feed_dict={X:batch,X_mask:mask})
            co+=c
        co/=3
        #w/=(n_features*n_hidden)

        learning_rate*=0.999
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
        #optimizer=tf.train.MomentumOptimizer(learning_rate,0.95).minimize(cost-nw*0.1)
        #optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

        if epoch % 100 ==0:
            #print("Depth: ", depth, " Epoch:", "%04d" % (epoch+1),"cost=", co, " weight=", w*0.1, " penalty=",co+w*0.1)
            print("Depth: ", depth, " Epoch:", "%04d" % (epoch+1),"cost=", co, " weight=", w)

    print("finish training")

    encoded=sess.run(hidden,feed_dict={dropX:data})
    w=sess.run(w_encoder)

    sess.close()
    print("start testing")
    testRF(encoded,labels)

    return encoded,w

def StackAE(data):

    encoding,w1=NonLinearAE(data,0,1,2000,learning_rate=0.05,max_iter=1000,denoise=True)
    w=w1

    encoding,w1=NonLinearAE(encoding,1,1,4000,learning_rate=0.05,max_iter=1000,denoise=True)
    w=np.matmul(w,w1)

    encoding,w1=NonLinearAE(encoding,2,1,2000,learning_rate=0.05,max_iter=1000)
    w=np.matmul(w,w1)

    encoding,w1=NonLinearAE(encoding,3,1,1000,learning_rate=0.05,max_iter=1000)
    w=np.matmul(w,w1)

    encoding,w1=NonLinearAE(encoding,4,1,500,learning_rate=0.05,max_iter=1000)
    w=np.matmul(w,w1)

    encoding,w1=NonLinearAE(encoding,5,1,200,learning_rate=0.05,max_iter=1000)
    w=np.matmul(w,w1)

    encoding,w1=NonLinearAE(encoding,5,1,100,learning_rate=0.05,max_iter=1000)
    w=np.matmul(w,w1)

    encoding,w1=NonLinearAE(encoding,5,1,50,learning_rate=0.05,max_iter=1000)
    w=np.matmul(w,w1)
    return

    encoding,w1=NonLinearAE(encoding,3,1,500,denoise=True)
    w=np.matmul(w,w1)

def main():
    global labels
    tf.set_random_seed(0)
    np.random.seed(0)

    data=pd.read_pickle("data-knn.pk1").values
    labels=(data[:,-1]-1).tolist()
    data=data[:,:-1]
    #data=np.load('tdata.pk1')

    t=time.time()
    StackAE(data)
    elapsed=int(time.time()-t)
    print('elapsed= %d min:%d seconds' % (int(elapsed/60),elapsed%60))

if __name__=="__main__":
    main()

log_file.close()

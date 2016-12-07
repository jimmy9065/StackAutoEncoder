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
learning_rate=[0.1,0.05,0.05,0.05,0.05,0.05,0.05]

Max_depth=4

labels=[]

build=True
test=True
matlab=True

log_file=open('log','w')

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

def NonLinearAE(data,depth,denoise=True):
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

    w_encoder=tf.Variable(tf.random_normal([n_features, n_hidden[depth]]))
    w_decoder=tf.Variable(tf.random_normal([n_hidden[depth], n_features]))
    b_encoder=tf.Variable(tf.random_normal([n_hidden[depth]]))
    b_decoder=tf.Variable(tf.random_normal([n_features]))

    hidden=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(dropX,w_encoder), b_encoder))
    y_pred=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden,w_decoder), b_decoder))
    y_true=X

    #cost
    #cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred,y_true))
    cost=tf.reduce_mean(tf.square(y_true-y_pred))

    #optimizer
    optimizer=tf.train.GradientDescentOptimizer(learning_rate[depth]).minimize(cost)
    #optimizer=tf.train.AdamOptimizer(learning_rate[depth]).minimize(cost)
    #optimizer=tf.train.AdadeltaOptimizer(learning_rate[depth]).minimize(cost)

    # start to run
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    for epoch in range(max_iter[depth]):
        batches=divideData(data)
        co=0
        for batch in batches:
            mask=addNoise(batch,0.01)
            _, c=sess.run([optimizer, cost],feed_dict={X:batch,X_mask:mask})
            co+=c
        co/=3

        learning_rate[depth]*=0.999
        optimizer=tf.train.GradientDescentOptimizer(learning_rate[depth]).minimize(cost)

        if epoch % 100 ==0:
            print("Depth: ", depth, " Epoch:", "%04d" % (epoch+1),"cost=", co)#"{:9f}".format(co))

    print("finish training")

    encoded=sess.run(hidden,feed_dict={dropX:data})
    f=open("layer"+str(depth),'wb')
    np.save(f,encoded)
    f.close()

    testSVM(encoded,labels)
    testLogistic(encoded,labels,n_hidden[depth])
    print("extract hidden in this layer=",depth)
    w=sess.run(w_encoder)
    print("extract weights in this layer=",depth)

    sess.close()
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
    global learning_rate1

    print("Layers0")
    log_file.write("depth="+str(depth)+"\n")

    if os.path.isfile('firstLayers.pk1'):
        encoder,w=np.load('firstLayers.pk1')
    else:
        n_features=data.shape[1]
        X=tf.placeholder(dtype=tf.float32, shape=(None,n_features))

        w_encoder=tf.Variable(tf.random_normal([n_features, n_hidden[0]]))
        w_decoder=tf.transpose(w_encoder)
        b_encoder=tf.Variable(tf.random_normal([n_hidden[0]]))
        b_decoder=tf.Variable(tf.random_normal([n_features]))

        hidden=tf.nn.bias_add(tf.matmul(X,w_encoder), b_encoder)
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

            if epoch >1000 and learning_rate1>0.1:
                learning_rate1=0.1
                optimizer=tf.train.GradientDescentOptimizer(learning_rate1).minimize(cost)
            if epoch>2000 and learning_rate1>0.05:
                learning_rate1=0.05
                optimizer=tf.train.GradientDescentOptimizer(learning_rate1).minimize(cost)

            batches=divideData(data)
            co=0
            for batch in batches:
                _, c=sess.run([optimizer, cost],feed_dict={X:batch})
                co+=c
            co/=len(batches)

            if epoch % 200 ==0:
                print("Epoch:","%04d" % (epoch+1),"cost=", "{:9f}".format(co))

        print("Finished")

        encoded=sess.run(hidden,feed_dict={X:data})
        w=sess.run(w_encoder)

        f=open('firstLayers','wb')
        np.save(f,[encoded,w])
        f.close()

    testSVM(encoded,labels)
    testLogistic(encoded,labels,n_hidden[0])

    e,ws=NonLinearAE(encoded,1)
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
    data=np.load('tdata.pk1')
    #labels=(data[:,-1]).tolist()

    data=data[:,:-1]

    #build=False
    #test=False
    matlab=False

    if (not build) and os.path.isfile('encoder.pk1'):
        encoder=np.load('encoder.pk1')[0]
    else:
        t=time.time()
        #encoder,weights=LinearAE(data)
        encoder,weights=NonLinearAE(data,0)
        elapsed=int(time.time()-t)
        print('elapsed= %d min:%d seconds' % (int(elapsed/60),elapsed%60))
        f=open('encoder.pk1','wb')
        np.save(f,[encoder,weights])
        f.close()

    if test:
        #print("[ 0.45945946  0.59459459  0.51351351  0.45945946  0.62162162  0.54054054  0.52777778  0.48571429  0.57142857  0.6]")
        print("0.5374")

        testSVM(encoder,labels)

        testLogistic(encoder,labels,encoder.shape[1])

    if matlab:
        D=np.column_stack([encoder,labels])
        sio.savemat('AE_features',{'D':D})

if __name__=="__main__":
    main()

log_file.close()

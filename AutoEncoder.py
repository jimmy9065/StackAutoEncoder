import numpy as np
import tensorflow as tf

def divideData(D):
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

def addNoise(Data,rate):
    x,y=Data.shape
    prob=np.random.rand(x,y)
    mask=prob<rate*1
    return mask

def NonLinearAE(data,depth,layers,n_hidden,learning_rate=0.05,max_iter=500,denoise=False,noise=0.15,activation='sigmoid',Sparcity=True):
    print("depth=",depth,"#hidden neuron=",n_hidden," ",layers,"layers")
    n_features=data.shape[1]

    X=tf.placeholder(dtype=tf.float32, shape=(None,n_features))
    X_mask=tf.placeholder(dtype=tf.float32, shape=(None,n_features))

    if denoise:
        print("adding noise with ",noise)
        dropX=tf.multiply(X,X_mask)
    else:
        print("no noise")
        dropX=X

    w_encoder=tf.Variable(tf.random_normal([n_features, n_hidden]))
    w_decoder=tf.Variable(tf.random_normal([n_hidden, n_features]))
    b_encoder=tf.Variable(tf.random_normal([n_hidden]))
    b_decoder=tf.Variable(tf.random_normal([n_features]))

    if activation=='relu':
        hidden=tf.nn.relu(tf.nn.bias_add(tf.matmul(dropX,w_encoder), b_encoder))
        y_pred=tf.nn.relu(tf.nn.bias_add(tf.matmul(hidden,w_decoder), b_decoder))
    else:
        hidden=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(dropX,w_encoder), b_encoder))
        y_pred=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(hidden,w_decoder), b_decoder))

    y_true=X

    #cost
    nA=tf.reduce_sum(tf.reduce_mean(hidden,0))
    #cost=tf.reduce_mean(tf.square(y_true-y_pred))
    cost=tf.reduce_sum(tf.reduce_mean(tf.square(y_true-y_pred),0))

    if Sparcity==True:
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost+nA*0.2)
    else:
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    #optimizer

    # start to run
    init=tf.global_variables_initializer()
    sess=tf.Session()
    sess.run(init)

    for epoch in range(max_iter+1):
        np.random.shuffle(data)
        if denoise:
            mask=addNoise(data,noise)
            _, c, na=sess.run([optimizer, cost, nA],feed_dict={X:data,X_mask:mask})
        else:
            _, c, na=sess.run([optimizer, cost, nA],feed_dict={X:data})

        #batches=divideData(data)
        #co=0
        #
        #for batch in batches:
        #    if denoise:
        #        mask=addNoise(batch,noise)
        #        _, c, na=sess.run([optimizer, cost, nA],feed_dict={X:batch,X_mask:mask})
        #    else:
        #        _, c, na=sess.run([optimizer, cost, nA],feed_dict={X:batch})
        #    co+=c
        #co/=3

        learning_rate*=0.999
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        if epoch % 100 ==0:
            #print("Depth: ", depth, " Epoch:", "%04d" % (epoch+1),"cost=", co, " Activation=", na)
            print("Depth: ", depth, " Epoch:", "%04d" % (epoch+1),"cost=", c, " Activation=", na)

    print("finish training")

    encoded=sess.run(hidden,feed_dict={dropX:data})
    w=sess.run(w_encoder)

    sess.close()
    return encoded,w

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
import time
import random
import sys

batch_size = 8

def loadData(trainPath, testPath):
    train_set = np.load(sys.argv[1])
    test_set = np.load(sys.argv[2])
    train_X = train_set[:, :-2]
    train_Y = train_set[:, -2:]
    test_X = test_set[:, :-2]
    test_Y = test_set[:, -2:]

    # transform train_X into np.array
    shape_X = np.zeros((len(train_X), 200, 200))
    for i in range(len(train_X)):
        for j in range(200):
            shape_X[i, j] = np.array(train_X[i][j][:])
    train_X = shape_X[:]
    shape_X = np.zeros((len(test_X), 200, 200))
    for i in range(len(test_X)):
        for j in range(200):
            shape_X[i, j] = np.array(test_X[i][j][:])
    test_X = shape_X[:]

    train_Y = np.array(train_Y, dtype=np.int32)
    test_Y = np.array(test_Y, dtype=np.int32)
    return train_X, train_Y, test_X, test_Y

train_X, train_Y, test_X, test_Y = loadData(sys.argv[1], sys.argv[2])

train_X = train_X.reshape(-1, 200, 200, 1)
test_X = test_X.reshape(-1, 200, 200, 1)


# after all the process the data type is in form np.array,
# X:[sampleNum, lineNum, wordVecDem, 1], Y:[sampleNum, 2]

# one iter one batch
def generatebatch(X,Y, batch_size):
    batch_xs, batch_ys = [], []
    while len(batch_xs) < batch_size/2:
        rand = np.random.randint(len(X))
        if Y[rand][0] == 0:
            batch_xs.append(X[rand])
            batch_ys.append(Y[rand])
    while len(batch_xs) < batch_size:
        rand = np.random.randint(len(X))
        if Y[rand][0] == 1:
            batch_xs.append(X[rand])
            batch_ys.append(Y[rand])
    return batch_xs, batch_ys


tf.reset_default_graph()

tf_X = tf.placeholder(tf.float32,[batch_size,200,200,1])
tf_Y = tf.placeholder(tf.float32,[batch_size,2])

# conv + relu layers
conv_filter_w1 = tf.Variable(tf.random_normal([5, 200, 1, 5]))
conv_filter_b1 =  tf.Variable(tf.random_normal([5]))
relu_feature_maps1 = tf.nn.relu(
    tf.nn.conv2d(tf_X, conv_filter_w1,strides=[1, 1, 1, 1], padding='SAME') + conv_filter_b1)

# pooling_layer
max_pool1 = tf.nn.max_pool(relu_feature_maps1,ksize=[1,5,200,1],strides=[1,4,1,1],padding='SAME')


# conv_layer
conv_filter_w2 = tf.Variable(tf.random_normal([5, 200, 5, 5]))
conv_filter_b2 =  tf.Variable(tf.random_normal([5]))
conv_out2 = tf.nn.conv2d(relu_feature_maps1, conv_filter_w2,strides=[1, 2, 1, 1], padding='SAME') + conv_filter_b2

# BN Normalization and relu layers
# mean:(kernelsNum) var:(kernelsNum)
batch_mean, batch_var = tf.nn.moments(conv_out2, [0, 1, 2], keep_dims=True)
shift = tf.Variable(tf.zeros([5]))
scale = tf.Variable(tf.ones([5]))
epsilon = 1e-3
BN_out = tf.nn.batch_normalization(conv_out2, batch_mean, batch_var, shift, scale, epsilon)
relu_BN_maps2 = tf.nn.relu(BN_out)

# pooling layer
max_pool2 = tf.nn.max_pool(relu_BN_maps2,ksize=[1,5,200,1],strides=[1,2,1,1],padding='SAME')


# graph flat
max_pool2_flat = tf.reshape(max_pool2, [-1, 50*5*200])


# softmax layer
fc_w1 = tf.Variable(tf.random_normal([50*5*200,200]))
fc_b1 =  tf.Variable(tf.random_normal([200]))
fc_out1 = tf.nn.relu(tf.matmul(max_pool2_flat, fc_w1) + fc_b1)


# output layer
out_w1 = tf.Variable(tf.random_normal([200,2]))
out_b1 = tf.Variable(tf.random_normal([2]))
pred = tf.nn.softmax(tf.matmul(fc_out1,out_w1)+out_b1)

loss = -tf.reduce_mean(tf_Y*tf.log(tf.clip_by_value(pred,1e-11,1.0)))

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

y_pred = tf.arg_max(pred,1)
bool_pred = tf.equal(tf.arg_max(tf_Y,1),y_pred)

accuracy = tf.reduce_mean(tf.cast(bool_pred,tf.float32)) # accuracy

saver = tf.train.Saver()

def makePlot(train_result, test_result):
    plt.figure()
    plt.plot(train_result)
    plt.plot(test_result)
    plt.show()

time_a = time.time()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(30):   # 1000 epoches
        batch_xs, batch_ys = generatebatch(train_X, train_Y, batch_size)
        sess.run(train_step,feed_dict={tf_X:batch_xs,tf_Y:batch_ys})
        if(step%10==0):
            res = 0
            for i in range(len(train_X) // batch_size):
                res += sess.run(accuracy,feed_dict={tf_X:train_X[batch_size*i:batch_size*(i+1)], tf_Y:train_Y[batch_size*i:batch_size*(i+1)]})
            print (step,res/(len(train_X)//batch_size))

    # save the model
    saver.save(sess, 'models/model.ckpt')
    print 'model saved.'


    train_feature = []      # output of maxpool2
    test_feature = []
    train_softmax = []
    test_softmax = []
    res_test = []       # test output
    length_test = len(test_X)
    length_train = len(train_X)


with tf.Session() as sess:
    saver.restore(sess, 'models/model.ckpt')
    start = 0
    end = start+batch_size
    # output of train feature and train softmax layer output
    while end < length_train:
        batch_xs = train_X[start:start+batch_size]
        batch_ys = train_Y[start:start+batch_size]
        pool_out = max_pool2_flat.eval(feed_dict={tf_X:batch_xs, tf_Y:batch_ys})
        softmax = fc_out1.eval(feed_dict={tf_X:batch_xs, tf_Y:batch_ys})
        for i in range(batch_size):
            train_feature.append(pool_out[i])
            train_softmax.append(softmax[i])

        start += batch_size
        end = start + batch_size
    batch_xs = train_X[length_train - batch_size:]
    batch_ys = train_Y[length_train - batch_size:]
    pool_out = max_pool2_flat.eval(feed_dict={tf_X:batch_xs, tf_Y:batch_ys})
    softmax = fc_out1.eval(feed_dict={tf_X:batch_xs, tf_Y:batch_ys})
    for i in range(length_train-start):
        train_feature.append(pool_out[i])
        train_softmax.append(softmax[i])

    start = 0
    end = start+batch_size

    # output of test feature and test result
    while end < length_test:
        batch_xs = test_X[start:start+batch_size]
        batch_ys = test_Y[start:start+batch_size]
        res_ypred = 1 - y_pred.eval(feed_dict={tf_X:batch_xs,tf_Y:batch_ys}).flatten()
        for item in res_ypred:
            res_test.append(item)
        pool_out = max_pool2_flat.eval(feed_dict={tf_X:batch_xs, tf_Y:batch_ys})
        softmax = fc_out1.eval(feed_dict={tf_X:batch_xs, tf_Y:batch_ys})
        for i in range(batch_size):
            test_feature.append(pool_out[i])
            test_softmax.append(softmax[i])

        start += batch_size
        end = start + batch_size
    batch_xs = test_X[length_test-batch_size:]
    batch_ys = test_Y[length_test-batch_size:]
    res_ypred = 1 - y_pred.eval(feed_dict={tf_X:batch_xs,tf_Y:batch_ys}).flatten()
    for i in range(length_test-start):
        res_test.append(res_ypred[i])
    pool_out = max_pool2_flat.eval(feed_dict={tf_X:batch_xs, tf_Y:batch_ys})
    softmax = fc_out1.eval(feed_dict={tf_X:batch_xs, tf_Y:batch_ys})
    for i in range(length_test-start):
        test_feature.append(pool_out[i])
        test_softmax.append(softmax[i])
    res_test = np.array(res_test)
    print res_test

    # write train feature and test feature to file
    train_feature = np.array(train_feature)
    test_feature = np.array(test_feature)
    np.save(sys.argv[3]+'feature', train_feature)
    np.save(sys.argv[4]+'feature', test_feature)
    np.save(sys.argv[3]+'softmax', train_softmax)
    np.save(sys.argv[4]+'softmax', test_softmax)
    print 'train feature and test feature and softmax layer output saved.'

testY_4eval = test_Y[:, 0]
print testY_4eval

print accuracy_score(testY_4eval,res_test.reshape(-1,1))
print precision_score(testY_4eval,res_test.reshape(-1,1))
print recall_score(testY_4eval,res_test.reshape(-1,1))
print f1_score(testY_4eval,res_test.reshape(-1,1))

time_b = time.time()

print 'time used {}'.format(time_b - time_a)
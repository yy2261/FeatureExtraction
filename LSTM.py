from __future__ import division
import tensorflow as tf  
import sys
import csv
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data 
import matplotlib.pyplot as plt
import random
      

# this is data
train_X = []
train_Y = []
test_X = []
test_Y = []
test_name = []
with open(sys.argv[1]) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        featureRow = []
        for i in range(len(row)-3):
            featureRow.append(map(float, row[i].strip('[').strip(']').split(', ')))
        train_X.append(featureRow) 
        train_Y.append([int(row[-2]), int(row[-1])])  #row[-2] = 1 if defect exists

with open(sys.argv[2]) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        featureRow = []
        for i in range(len(row)-3):
            featureRow.append(map(float, row[i].strip('[').strip(']').split(', ')))
        test_X.append(featureRow) 
        test_Y.append([int(row[-2]), int(row[-1])])  #row[-2] = 1 if defect exists
        test_name.append(row[-3])

train_X = np.array(train_X)
train_Y = np.array(train_Y)
test_X = np.array(test_X)
test_Y = np.array(test_Y)
test_name = np.array(test_name)
      
# hyperparameters  
lr = 0.0001
training_iters = 500
batch_size = 424
      
n_inputs = 200 
n_steps = 200
n_hidden_units = 200   # neurons in hidden layer  
n_classes = 2      # classes
      
# tf Graph input  
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])  
y = tf.placeholder(tf.float32, [None, n_classes])  
      
# Define weights  
weights = {  
    # (28, 128)  
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),  
    # (128, 10)  
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))  
}  
biases = {  
    # (128, )  
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),  
    # (10, )  
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))  
}  
      
      
def RNN(X, weights, biases):  
    # hidden layer for input to cell  
    #X(128 batch,28 steps,28 inputs)  
    #==>(128*28,28 inputs)  
    X = tf.reshape(X,[-1,n_inputs])      
    #==>(128 batch*28 steps,128 hidden)  
    X_in = tf.matmul(X,weights['in'])+biases['in']  
    #==>(128 batch,28 steps,128 hidden)  
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])  
    # cell
    #same to define active function  
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1,state_is_tuple=True)  
    #lstm cell is divided into two parts(c_state,m_state)  
    _init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)  
   
    #choose rnn how to work,lstm just is one kind of rnn,use lstm_cell for active function,set initial_state  
    outputs,states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)     
              
    # hidden layer for output as the final results  
    results = tf.matmul(states[1],weights['out']) + biases['out']     
          
    #unpack to list [(batch,outputs)]*steps  
    #outputs = tf.unpack(tf.transpose(outputs,[1,0,2])) # state is the last outputs  
    #results = tf.matmul(outputs[-1],weights['out']) + biases['out']  
    return results
      
      
pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))  
train_op = tf.train.AdamOptimizer(lr).minimize(cost)  

    
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  
      
init = tf.initialize_all_variables()
saver = tf.train.Saver()
train_result = []
test_result = []

with tf.Session() as sess:
    sess.run(init)
    step = 0
    for i in range(training_iters):

# make train and validate batch
        xs = []
        ys = []
        xs_test = []
        ys_test = []
        for num in range(batch_size):
            rand = np.random.randint(len(train_X))
            xs.append(train_X[rand])
            ys.append(train_Y[rand])
            rand = np.random.randint(len(test_X))
            xs_test.append(test_X[rand])
            ys_test.append(test_Y[rand])

# reshape xs and xs_test for feed_dict 
        xs = np.array(xs).reshape([-1, n_steps, n_inputs])
        ys = np.array(ys)
        xs_test = np.array(xs_test).reshape([-1, n_steps, n_inputs])
        ys_test = np.array(ys_test)

# calculate cost
        _, loss_ = sess.run([train_op, cost], feed_dict={  
            x: xs,  
            y: ys,
        })
        if step % 10 == 0:
            train_result.append(sess.run(accuracy, feed_dict={  
            x: xs,  
            y: ys,  
            }))
            test_result.append(sess.run(accuracy, feed_dict={  
            x: xs_test,  
            y: ys_test,  
            }))
            print 'loss is: '+str(loss_)
        step += 1
    saver.save(sess, 'model.ckpt')

plt.figure()
plt.plot(train_result)
plt.plot(test_result)
plt.show()

result = []
with tf.Session() as sess:
    saver.restore(sess, './model.ckpt')
    start = 0
    end = start + batch_size
    while start < len(test_X):
        result.append(sess.run(pred, feed_dict={
        x: test_X[start:end].reshape([-1, n_steps, n_inputs]),
        }))
        start += batch_size
        end = start + batch_size


tp = 0
fp = 0
fn = 0
tn = 0

for i in range(len(result)):
    for j in range(batch_size):
        predict = result[i][j]
        test = test_Y[batch_size*i+j]
        if predict[0] > predict[1] and test[0] == 1:
            tp += 1
#            print 'tp!'
#            print test_name[batch_size*i+j]
        elif predict[0] < predict[1] and test[0] == 0:
            tn += 1
        elif predict[0] > predict[1] and test[0] == 0:
            fp += 1
 #           print 'fp!'
 #           print test_name[batch_size*i+j]
        elif predict[0] < predict[1] and test[0] == 1:
            fn += 1
 #           print 'fn!'
 #           print test_name[batch_size*i+j]

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print tp
print fp
print fn
print tn

print 'precision is'
print precision
print 'recall is'
print recall


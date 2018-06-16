from __future__ import division
import tensorflow as tf  
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import random

# hyperparameters
global lr, training_iters, batch_size, n_inputs, n_steps, n_hidden_units, n_classes
lr = 0.000001
training_iters = 2000
batch_size = 20
      
n_inputs = 200
n_steps = 40
n_hidden_units = 30  # neurons in hidden layer  
n_classes = 2      # classes
      

def processData(train_set, test_set):
    global lr, training_iters, batch_size, n_inputs, n_steps, n_hidden_units, n_classes

    train_X = train_set[:, :-2]
    train_Y = train_set[:, -2:]
    test_X = test_set[:, :-2]
    test_Y = test_set[:, -2:]

    # transform train_X into np.array
    shape_X = np.zeros((len(train_X), n_steps*5, n_inputs))
    for i in range(len(train_X)):
        for j in range(n_steps*5):
            shape_X[i, j] = np.array(train_X[i][j][:])
    train_X = shape_X[:]
    shape_X = np.zeros((len(test_X), n_steps*5, n_inputs))
    for i in range(len(test_X)):
        for j in range(n_steps*5):
            shape_X[i, j] = np.array(test_X[i][j][:])
    test_X = shape_X[:]

    return train_X, train_Y, test_X, test_Y
      
def RNN(X, weights, weights_out, weights_outs, biases, biases_out, biases_outs): 
    global lr, training_iters, batch_size, n_inputs, n_steps, n_hidden_units, n_classes

    # change shape of X
    X = map(tf.reshape, X, [[-1,n_inputs]]*5)
    X_in = map(tf.matmul, X, weights)
    X_in = map(tf.add, X_in, biases)
    X_in = map(tf.reshape, X_in, [[-1,n_steps,n_hidden_units]]*5)

    # cell  
    with tf.variable_scope("first_lstm"):
    	lstm_cell_1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1,state_is_tuple=True)
    	_init_state = lstm_cell_1.zero_state(batch_size,dtype=tf.float32)
    	outputs_1,states_1 = tf.nn.dynamic_rnn(lstm_cell_1,X_in[0],initial_state=_init_state,time_major=False)
    
    with tf.variable_scope("sencond_lstm"):
    	lstm_cell_2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1,state_is_tuple=True)
    	_init_state = lstm_cell_2.zero_state(batch_size,dtype=tf.float32)
    	outputs_2,states_2 = tf.nn.dynamic_rnn(lstm_cell_2,X_in[1],initial_state=_init_state,time_major=False)

    with tf.variable_scope("third_lstm"):    	
    	lstm_cell_3 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1,state_is_tuple=True)
    	_init_state = lstm_cell_3.zero_state(batch_size,dtype=tf.float32)
    	outputs_3,states_3 = tf.nn.dynamic_rnn(lstm_cell_3,X_in[2],initial_state=_init_state,time_major=False)
    
    with tf.variable_scope("fourth_lstm"):
    	lstm_cell_4 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1,state_is_tuple=True)
    	_init_state = lstm_cell_4.zero_state(batch_size,dtype=tf.float32)
    	outputs_4,states_4 = tf.nn.dynamic_rnn(lstm_cell_4,X_in[3],initial_state=_init_state,time_major=False)

    with tf.variable_scope("fifth_lstm"):
        lstm_cell_5 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1,state_is_tuple=True)
        _init_state = lstm_cell_5.zero_state(batch_size,dtype=tf.float32)
        outputs_5,states_5 = tf.nn.dynamic_rnn(lstm_cell_5,X_in[4],initial_state=_init_state,time_major=False)


    #lstm cell is divided into two parts(c_state,m_state)
    #choose rnn how to work,lstm just is one kind of rnn,use lstm_cell for active function,set initial_state  
    # hidden layer for output as the final results
    states = [states_1[1], states_2[1], states_3[1], states_4[1], states_5[1]]
    outputs = map(tf.matmul, states, weights_outs)
    outputs = map(tf.add, outputs, biases_outs)

    output = tf.concat(axis=1, values=outputs)  
    results = tf.matmul(output, weights_out) + biases_out
    return results, output


def buildModel():
    global lr, training_iters, batch_size, n_inputs, n_steps, n_hidden_units, n_classes
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_steps*5, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Define weights

    weights = tuple([tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))] * 5)
    biases = tuple([tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]))] * 5)

    # change weights_outs y to 10 instead of 2
    weights_outs = tuple([tf.Variable(tf.random_normal([n_hidden_units, 10]))] * 5)
    biases_outs = tuple([tf.Variable(tf.constant(0.1, shape=[10, ]))] * 5)
    weights_out = tf.Variable(tf.random_normal([50, n_classes])) 
    biases_out = tf.Variable(tf.constant(0.1, shape=[n_classes, ]))

    x_1 = x[:, :n_steps, :]
    x_2 = x[:, n_steps:n_steps*2, :]
    x_3 = x[:, n_steps*2:n_steps*3, :]
    x_4 = x[:, n_steps*3:n_steps*4, :]
    x_5 = x[:, n_steps*4:, :]

    X = [x_1, x_2, x_3, x_4, x_5]
          
    pred, output = RNN(X, weights, weights_out, weights_outs, biases, biases_out, biases_outs)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return train_op, cost, x, y, accuracy, pred, output


def selectData(train_X, train_Y, test_X, test_Y):
    # input: training set and test set
    # output: xs, ys, xs_test, ys_test in np.array form
    global lr, training_iters, batch_size, n_inputs, n_steps, n_hidden_units, n_classes
    xs, ys, xs_test, ys_test = [], [], [], []
    while len(xs) < batch_size/2:
        rand = np.random.randint(len(train_X))
        if train_Y[rand][0] == 0:
            xs.append(train_X[rand])
            ys.append(train_Y[rand])
    while len(xs) < batch_size:
        rand = np.random.randint(len(train_X))
        if train_Y[rand][0] == 1:
            xs.append(train_X[rand])
            ys.append(train_Y[rand])
    for num in range(batch_size):
        rand = np.random.randint(len(test_X))
        xs_test.append(test_X[rand])
        ys_test.append(test_Y[rand])
    # reshape xs and xs_test for feed_dict
    xs = np.array(xs).reshape([-1, n_steps*5, n_inputs])
    ys = np.array(ys)
    xs_test = np.array(xs_test).reshape([-1, n_steps*5, n_inputs])
    ys_test = np.array(ys_test)
    return xs, ys, xs_test, ys_test

def makePlot(train_result, test_result):
    plt.figure()
    plt.plot(train_result)
    plt.plot(test_result)
    plt.show()

def train(train_op, cost, x, y, accuracy, train_X, train_Y, test_X, test_Y):
    global lr, training_iters, batch_size, n_inputs, n_steps, n_hidden_units, n_classes
    # init the model
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    train_result = []
    test_result = []

    with tf.Session() as sess:
        sess.run(init)
        step = 0
        for i in range(training_iters):
            xs , ys, xs_test, ys_test = selectData(train_X, train_Y, test_X, test_Y)        
    # calculate cost
            _, loss_ = sess.run([train_op, cost], feed_dict={x: xs, y: ys})
            if step % 10 == 0:
                train_result.append(sess.run(accuracy, feed_dict={x: xs, y: ys}))
                test_result.append(sess.run(accuracy, feed_dict={x: xs_test, y: ys_test}))
                print 'loss is: '+str(loss_)
            step += 1
        saver.save(sess, 'models/model.ckpt')
    makePlot(train_result, test_result)
    return saver

def predict(saver, pred, output, x, test_X, test_Y):
    global lr, training_iters, batch_size, n_inputs, n_steps, n_hidden_units, n_classes
    result = []
    result_softmax = []
    with tf.Session() as sess:
        saver.restore(sess, 'models/model.ckpt')
        start = 0
        end = start + batch_size
        while end < len(test_X):
            result_batch, softmax = sess.run([pred, output], feed_dict={
                x: test_X[start:end].reshape([-1, n_steps*5, n_inputs])})
            start += batch_size
            end = start + batch_size
            for i in range(batch_size):
                result.append(result_batch[i])
                result_softmax.append(softmax[i])
        # last result
        result_last_batch, softmax_last_batch = sess.run([pred, output], feed_dict={
            x: test_X[len(test_X)-batch_size:].reshape([-1, n_steps*5, n_inputs])})
        for i in range(len(test_X)-start):
            result.append(result_last_batch[i])
            result_softmax.append(softmax_last_batch[i])
    result_softmax = np.array(result_softmax)
    return result

def measure(result, test_Y):
    global lr, training_iters, batch_size, n_inputs, n_steps, n_hidden_units, n_classes
    
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(result)):
        predict = result[i]
        test = test_Y[i]
        if predict[0] > predict[1] and test[0] == 1:    tp += 1
        elif predict[0] < predict[1] and test[0] == 0:  tn += 1
        elif predict[0] > predict[1] and test[0] == 0:  fp += 1
        elif predict[0] < predict[1] and test[0] == 1:  fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2*precision*recall/(precision+recall)

    print tp, fp, fn, tn
    print 'precision is:'+str(precision)
    print 'recall is:'+str(recall)
    print 'f1 is:'+str(f1)

def main(train_path, test_path):
    train_set = np.load(train_path)
    test_set = np.load(test_path)
    train_X, train_Y, test_X, test_Y = processData(train_set, test_set)
    train_op, cost, x, y, accuracy, pred, output = buildModel()
    saver = train(train_op, cost, x, y, accuracy, train_X, train_Y, test_X, test_Y)
    result = predict(saver, pred, output, x, test_X, test_Y)
    measure(result, test_Y)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
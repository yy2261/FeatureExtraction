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
    shape_X = np.zeros((len(train_X), n_steps, n_inputs))
    for i in range(len(train_X)):
        for j in range(n_steps):
            shape_X[i, j] = np.array(train_X[i][j][:])
    train_X = shape_X[:]
    shape_X = np.zeros((len(test_X), n_steps, n_inputs))
    for i in range(len(test_X)):
        for j in range(n_steps):
            shape_X[i, j] = np.array(test_X[i][j][:])
    test_X = shape_X[:]
    return train_X, train_Y, test_X, test_Y
      
def RNN(x, weight, weight_out, bias, bias_out, mask): 
    global lr, training_iters, batch_size, n_inputs, n_steps, n_hidden_units, n_classes

    # calculate input of lstm
    x = tf.reshape(x, [-1, n_inputs])
    xLSTM = tf.add(tf.matmul(x, weight), bias)
    xLSTM = tf.reshape(xLSTM, [-1, n_steps, n_hidden_units])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs = []

    # cell  
    with tf.variable_scope("LSTM_LAYER"):
        state = _init_state
        for time_step in range(n_steps):
            if time_step > 0:
                tf.get_variable_scope().reuse_variables()
            (cell_output, state) = lstm_cell(xLSTM[:, time_step, :], state)
            outputs.append(cell_output)

    with tf.variable_scope("MEAN_POOLING"):
        outputs = outputs * mask
        outputs = tf.reduce_sum(outputs, 0) / (tf.reduce_sum(mask, 0))

    results = tf.matmul(outputs, weight_out) + bias_out
    return results


def buildModel():
    global lr, training_iters, batch_size, n_inputs, n_steps, n_hidden_units, n_classes
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_classes])
    mask = tf.placeholder(tf.float32, [n_steps, None, n_hidden_units])

    # Define weights
    weight = tf.Variable(tf.random_normal([n_inputs, n_hidden_units]))
    bias = tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]))

    weight_out = tf.Variable(tf.random_normal([n_hidden_units, n_classes])) 
    bias_out = tf.Variable(tf.constant(0.1, shape=[n_classes, ]))

    pred = RNN(x, weight, weight_out, bias, bias_out, mask)


    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return train_op, cost, x, y, mask, accuracy, pred


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
    xs = np.array(xs).reshape([-1, n_steps, n_inputs])
    ys = np.array(ys)
    xs_test = np.array(xs_test).reshape([-1, n_steps, n_inputs])
    ys_test = np.array(ys_test)
    return xs, ys, xs_test, ys_test

def makePlot(train_result, test_result):
    plt.figure()
    plt.plot(train_result)
    plt.plot(test_result)
    plt.show()

def train(train_op, cost, x, y, mask, accuracy, train_X, train_Y, test_X, test_Y):
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
            masks = np.array([[[1]*n_hidden_units]*batch_size]*n_steps)
    # calculate cost
            _, loss_ = sess.run([train_op, cost], feed_dict={x: xs, y: ys, mask: masks})
            if step % 10 == 0:
                train_result.append(sess.run(accuracy, feed_dict={x: xs, y: ys, mask: masks}))
                test_result.append(sess.run(accuracy, feed_dict={x: xs_test, y: ys_test, mask: masks}))
                print 'loss is: '+str(loss_)
            step += 1
        saver.save(sess, 'models/model.ckpt')
    makePlot(train_result, test_result)
    return saver

def predict(saver, pred, x, mask, test_X, test_Y):
    global lr, training_iters, batch_size, n_inputs, n_steps, n_hidden_units, n_classes
    result = []
    with tf.Session() as sess:
        saver.restore(sess, 'models/model.ckpt')
        start = 0
        end = start + batch_size
        masks = np.array([[[1]*n_hidden_units]*batch_size]*n_steps)
        while end < len(test_X):
            result_batch = sess.run(pred, feed_dict={
                x: test_X[start:end].reshape([-1, n_steps, n_inputs]), mask: masks})
            start += batch_size
            end = start + batch_size
            for i in range(batch_size):
                result.append(result_batch[i])
        # last result
        result_last_batch = sess.run(pred, feed_dict={
            x: test_X[len(test_X)-batch_size:].reshape([-1, n_steps, n_inputs]), mask: masks})
        for i in range(len(test_X)-start):
            result.append(result_last_batch[i])
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
    n_steps = len(train_X[0])
    train_op, cost, x, y, mask, accuracy, pred = buildModel()
    print 'going to train...'
    saver = train(train_op, cost, x, y, mask, accuracy, train_X, train_Y, test_X, test_Y)
    result = predict(saver, pred, x, mask, test_X, test_Y)
    measure(result, test_Y)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
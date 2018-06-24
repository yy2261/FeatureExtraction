from __future__ import division
import tensorflow as tf  
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import random

# hyperparameters
global lr, training_iters, batch_size, n_inputs, n_steps, n_hidden_units, n_classes
lr = float(sys.argv[3])
training_iters = 100
batch_size = 20
      
n_inputs = 50
n_steps = 0
n_hidden_units = int(sys.argv[4])  # neurons in hidden layer  
n_classes = 2      # classes
      

def processData(train_set, test_set):
    global lr, batch_size, n_inputs, n_steps, n_hidden_units

    train_X = train_set[:, :-2]
    train_Y = train_set[:, -2:]
    test_X = test_set[:, :-2]
    test_Y = test_set[:, -2:]

    train_mask = np.zeros((n_steps, len(train_X), n_hidden_units))
    # transform train_X into np.array
    shape_X = np.zeros((len(train_X), n_steps, n_inputs))
    for i in range(len(train_X)):
        for j in range(n_steps):
            
            shape_X[i, j] = np.array(train_X[i][j][:])
            
            if train_X[i][j] != [0.0]*n_inputs:
                train_mask[j][i] = np.ones(n_hidden_units)
    
    train_X = shape_X[:]

    # transform test_X into np.array, if length of test_X is smaller than train_X then padding
    test_mask = np.zeros((n_steps, len(test_X), n_hidden_units))
    shape_X = np.zeros((len(test_X), n_steps, n_inputs))

    for i in range(len(test_X)):
        for j in range(n_steps):
            if len(test_X[i]) > j:
                shape_X[i, j] = np.array(test_X[i][j][:])
            else:
                shape_X[i, j] = [0.0]* n_inputs

            if len(test_X[i]) > j and test_X[i][j] != [0.0] * n_inputs:
                test_mask[j][i] = np.ones(n_hidden_units)

    test_X = shape_X[:]
    return train_X, train_Y, train_mask, test_X, test_Y, test_mask
      
def RNN(x, weight, weight_out, bias, bias_out, mask): 
    global batch_size, n_inputs, n_steps, n_hidden_units

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
    global lr, n_inputs, n_steps, n_hidden_units, n_classes
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


    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=(pred+1e-10), labels=y))
    train_op = tf.train.AdamOptimizer(lr).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))  
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return train_op, cost, x, y, mask, accuracy, pred


def genTrainBatch(train_set):
    # input: training set (zip of train_X, train_Y and train_mask)
    global batch_size
    batches = len(train_set) // batch_size
    for i in range(batches):
        yield train_set[i:i+batch_size]


def makePlot(train_result, test_result):
    plt.figure()
    plt.plot(train_result)
    plt.plot(test_result)
    plt.show()

def train(train_op, cost, x, y, mask, accuracy, train_X, train_Y, train_mask):
    global training_iters
    # init the model
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    train_result = []

    with tf.Session() as sess:
        sess.run(init)
        for i in range(training_iters):
            train_set = zip(train_X, train_Y, np.swapaxes(train_mask, 0, 1))
            np.random.shuffle(train_set)
            batches = [ _ for _ in genTrainBatch(train_set)]

            losses = []
            for batch in batches:
                batch = zip(*batch)
                xs, ys, masks = batch[0], batch[1], batch[2]
                # calculate cost
                _, loss_ = sess.run([train_op, cost], feed_dict={x: xs, y: ys, mask: np.swapaxes(masks, 0, 1)})
                losses.append(loss_)
            if i % 10 == 0:
                print 'loss is: '+str(np.mean(losses))

        saver.save(sess, 'models/model.ckpt')
    return saver

def predict(saver, pred, x, mask, test_X, test_Y, test_mask):
    global batch_size, n_inputs, n_steps

    result = []
    with tf.Session() as sess:
        saver.restore(sess, 'models/model.ckpt')
        start = 0
        end = start + batch_size
        while end < len(test_X):
            result_batch = sess.run(pred, feed_dict={
                x: test_X[start:end].reshape([-1, n_steps, n_inputs]), mask: test_mask[:, start:end, :]})
            start += batch_size
            end = start + batch_size
            for i in range(batch_size):
                result.append(result_batch[i])
        # last result
        result_last_batch = sess.run(pred, feed_dict={
            x: test_X[len(test_X)-batch_size:].reshape([-1, n_steps, n_inputs]), mask: test_mask[:, len(test_X)-batch_size:, :]})
        for i in range(len(test_X)-start):
            result.append(result_last_batch[i])
    return result

def measure(result, test_Y):
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
    global n_steps
    train_set = np.load(train_path)
    test_set = np.load(test_path)
    n_steps = len(train_set[0])-2
    print 'processing data...'
    train_X, train_Y, train_mask, test_X, test_Y, test_mask = processData(train_set, test_set)
    print 'building model...'
    train_op, cost, x, y, mask, accuracy, pred = buildModel()
    print 'training...'
    saver = train(train_op, cost, x, y, mask, accuracy, train_X, train_Y, train_mask)
    print 'predicting...'
    result = predict(saver, pred, x, mask, test_X, test_Y, test_mask)
    print 'measure results...'
    measure(result, test_Y)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
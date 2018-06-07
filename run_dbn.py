import numpy as np
import tensorflow as tf
import sys

from yadlt.models.boltzmann import dbn
from yadlt.utils import datasets, utilities
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


rbm_layers = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
rbm_learning_rate = [0.001, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
rbm_num_epochs = [200]
rbm_batch_size = [50]
rbm_gibbs_k = [1]
finetune_opt = 'adam'     # sgd/adagrad/momentum/adam
finetune_loss_func = 'softmax_cross_entropy'        # softmax_cross_entropy/mse 
finetune_dropout = 1
finetune_num_epochs = 1

if __name__ == '__main__':

    utilities.random_seed_np_tf(2)

    trX, trY = np.load(sys.argv[1]), np.load(sys.argv[2])
    vlX, vlY = np.load(sys.argv[3]), np.load(sys.argv[4])

    # Create the object
    finetune_act_func = utilities.str2actfunc('relu')

    srbm = dbn.DeepBeliefNetwork(
        name='dbn',
        rbm_layers=rbm_layers,
        finetune_act_func=finetune_act_func,
        rbm_learning_rate=rbm_learning_rate,
        rbm_num_epochs=rbm_num_epochs,
        rbm_gibbs_k = rbm_gibbs_k,
        rbm_gauss_visible=True,
        rbm_stddev=50,
        momentum=0.9,
        rbm_batch_size=rbm_batch_size,
	)

    train_result, valid_result = srbm.pretrain(trX, trY, vlX, vlY)

    print train_result, valid_result

    clf=LogisticRegression(penalty='l1', C=100, solver='liblinear')

    trY = [item[0] for item in trY]
    vlY = [item[0] for item in vlY]

    clf.fit(train_result, trY)
    result = clf.predict(valid_result)

    print accuracy_score(vlY, result)
    print precision_score(vlY, result)
    print recall_score(vlY, result)
    print f1_score(vlY, result)

import numpy as np
import tensorflow as tf
import sys

from yadlt.models.boltzmann import dbn
from yadlt.utils import datasets, utilities


rbm_layers = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
rbm_learning_rate = [0.005]
rbm_num_epochs = [200]
rbm_batch_size = [150]
rbm_gibbs_k = [1]
finetune_opt = 'sgd'     # sgd/adagrad/momentum/adam
finetune_loss_func = 'softmax_cross_entropy'        # softmax_cross_entropy/mse 
finetune_dropout = 1
finetune_num_epochs = 1000

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
        rbm_stddev=0.5,
        momentum=0.9,
        rbm_batch_size=rbm_batch_size,
        finetune_learning_rate=0.01,
        finetune_num_epochs=finetune_num_epochs,
        finetune_batch_size=150,
        finetune_opt=finetune_opt,
        finetune_loss_func=finetune_loss_func,
        finetune_dropout=finetune_dropout,
	decay_step = 1400,
	decay_rate = 0.65
	)

    train_result, valid_result = srbm.pretrain(trX, trY, vlX, vlY)

    srbm.fit(trX, trY, vlX, vlY)

    accuracy, precision, recall = srbm.score(vlX, vlY)
    print('Test set accuracy: {}'.format(accuracy))
    print('Test set precision:{}'.format(precision))
    print('Test set recall:{}'.format(recall))

    train_result_finetune, _ = srbm.predict(trX, True)
    valid_result_finetune, _ = srbm.predict(vlX, True)

    np.save(sys.argv[5], train_result_finetune)
    np.save(sys.argv[6], valid_result_finetune)

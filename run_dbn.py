import numpy as np
import tensorflow as tf

from yadlt.models.boltzmann import dbn
from yadlt.utils import datasets, utilities

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('train_labels', '', 'Path to train labels .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('valid_labels', '', 'Path to valid labels .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_string('name', 'dbn', 'Name of the model.')
flags.DEFINE_string('save_predictions', '', 'Path to a .npy file to save predictions of the model.')
flags.DEFINE_string('save_layers_output_test', '', 'Path to a .npy file to save test set output from all the layers of the model.')
flags.DEFINE_string('save_layers_output_train', '', 'Path to a .npy file to save train set output from all the layers of the model.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')

# RBMs layers specific parameters
flags.DEFINE_string('rbm_layers', '256,', 'Comma-separated values for the layers in the sdae.')
flags.DEFINE_boolean('rbm_gauss_visible', False, 'Whether to use Gaussian units for the visible layer.')
flags.DEFINE_float('rbm_stddev', 0.1, 'Standard deviation for Gaussian visible units.')
flags.DEFINE_string('rbm_learning_rate', '0.001,', 'Initial learning rate.')
flags.DEFINE_string('rbm_num_epochs', '10,', 'Number of epochs.')
flags.DEFINE_string('rbm_batch_size', '32,', 'Size of each mini-batch.')
flags.DEFINE_string('rbm_gibbs_k', '1,', 'Gibbs sampling steps.')

# Supervised fine tuning parameters
flags.DEFINE_string('finetune_act_func', 'relu', 'Activation function.')
flags.DEFINE_float('finetune_learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_float('finetune_momentum', 0.9, 'Momentum parameter.')
flags.DEFINE_integer('finetune_num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('finetune_batch_size', 32, 'Size of each mini-batch.')
flags.DEFINE_string('finetune_opt', 'momentum', '["sgd", "ada_grad", "momentum", "adam"]')
flags.DEFINE_string('finetune_loss_func', 'softmax_cross_entropy', 'Loss function. ["mse", "softmax_cross_entropy"]')
flags.DEFINE_float('finetune_dropout', 1, 'Dropout parameter.')

flags.DEFINE_string('path_to_train', '/home/yy/xiajbxie/', 'path to save train features')
flags.DEFINE_string('path_to_test', '/home/yy/xiajbxie/', 'path to save test features')

# Conversion of Autoencoder layers parameters from string to their specific type
rbm_layers = utilities.flag_to_list(FLAGS.rbm_layers, 'int')
rbm_learning_rate = utilities.flag_to_list(FLAGS.rbm_learning_rate, 'float')
rbm_num_epochs = utilities.flag_to_list(FLAGS.rbm_num_epochs, 'int')
rbm_batch_size = utilities.flag_to_list(FLAGS.rbm_batch_size, 'int')
rbm_gibbs_k = utilities.flag_to_list(FLAGS.rbm_gibbs_k, 'int')

path_to_train = FLAGS.path_to_train
path_to_test = FLAGS.path_to_test

# Parameters validation
assert FLAGS.finetune_act_func in ['sigmoid', 'tanh', 'relu']
assert len(rbm_layers) > 0

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)


    def load_from_np(dataset_path):
        if dataset_path != '':
            return np.load(dataset_path)
        else:
            return None

    trX, trY = load_from_np(FLAGS.train_dataset), load_from_np(FLAGS.train_labels)
    vlX, vlY = load_from_np(FLAGS.valid_dataset), load_from_np(FLAGS.valid_labels)

    # Create the object
    finetune_act_func = utilities.str2actfunc(FLAGS.finetune_act_func)

    srbm = dbn.DeepBeliefNetwork(
        name=FLAGS.name, do_pretrain=True,
        rbm_layers=rbm_layers,
        finetune_act_func=finetune_act_func, rbm_learning_rate=rbm_learning_rate,
        rbm_num_epochs=rbm_num_epochs, rbm_gibbs_k = rbm_gibbs_k,
        rbm_gauss_visible=FLAGS.rbm_gauss_visible, rbm_stddev=FLAGS.rbm_stddev,
        momentum=FLAGS.momentum, rbm_batch_size=rbm_batch_size, finetune_learning_rate=FLAGS.finetune_learning_rate,
        finetune_num_epochs=FLAGS.finetune_num_epochs, finetune_batch_size=FLAGS.finetune_batch_size,
        finetune_opt=FLAGS.finetune_opt, finetune_loss_func=FLAGS.finetune_loss_func,
        finetune_dropout=FLAGS.finetune_dropout)

    # Fit the model (unsupervised pretraining)
    newtrX, newvlX = srbm.pretrain(trX, vlX)

    print('saving train set features...')
    np.save(path_to_train, newtrX)
    print('saving test set features...')
    np.save(path_to_test, newvlX)
    print('done.')
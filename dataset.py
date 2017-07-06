import os
os.chdir('/home/abhishek/Projects/tensorflow_11/master-thesis/my_public_repos/storn_dvbf/')

activate_this = '../../.././venv/bin/activate_this.py'
execfile(activate_this, dict(__file__ = activate_this))

import pickle
import numpy as np
# from numpy.random import *
import gym

import utilities

class Datasets(object):
    pass

class Dataset(object):
    def __init__(self, features):
        # assert features.shape[0] == labels.shape[0], ("features.shape: %s labels.shape: %s" % (features.shape, labels.shape))
        self._num_examples = features.shape[0]

        features = features.astype(np.float32)
        # features = np.multiply(features - 130.0, 1.0 / 70.0) # [130.0 - 200.0] -> [0 - 1]
        self._features = features
        # self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def full_data(self):
        return self._features

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._features = self._features[perm]
            # self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._features[:,start:end,:] # , self._labels[start:end]

# shuffle data and its label in association
def corresponding_shuffle(data):
    random_indices = np.random.permutation(len(data))
    _data = np.zeros_like(data)
    for i, j in enumerate(random_indices):
        _data[i] = data[j]
    return _data

# save dataset as a pickle file
def save_as_pickle(filename, dataset):
    with open(filename, "wb") as f:
        pickle.dump(dataset, f)

# entry point
if __name__ == '__main__':
    datasets = Datasets()

    env = gym.make('Pendulum-v0')

    n_samples = 1000
    n_timesteps = 100
    # is the reward handled as observation?
    learned_reward = True
    
    X, U = utilities.rollout(env, n_samples, n_timesteps, learned_reward=learned_reward, fn_action=None)
    X_mean = X.reshape((-1, X.shape[2])).mean(0)
    X = X - X_mean
    X_std = X.reshape((-1, X.shape[2])).std(0)
    X = X / X_std
    # 4 dimensions and the control signal combined would be the input variable. 
    XU = np.concatenate((X, U), -1)

    # shuffle
    data = corresponding_shuffle(XU)

    # split data
    N_train = np.floor(n_samples * 2 * 1).astype(np.int32)
    N_validation = np.floor(N_train * 0).astype(np.int32)
    x_train, x_test = np.split(data, [N_train])
    # y_train, y_test = np.split(target, [N_train])
    x_validation = x_train[:N_validation]
    # y_validation = y_train[:N_validation]

    # create dataset
    datasets.train = Dataset(x_train)
    datasets.test = Dataset(x_test)
    datasets.validation = Dataset(x_validation)

    # save as a pickle file
    save_as_pickle('./pickled_data/XU.pkl', datasets)

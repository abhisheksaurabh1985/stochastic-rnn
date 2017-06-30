import os
os.chdir('/home/abhishek/Projects/tensorflow_11/master-thesis/my_public_repos/storn_dvbf/')

activate_this = '../../.././venv/bin/activate_this.py'
execfile(activate_this, dict(__file__ = activate_this))

import numpy as np
import tensorflow as tf
import gym

# os.chdir(".././my_public_repos/storn_dvbf/")

#%%
from storn import STORN
from utils import * # Authored by Max
from utilities import *
#%% 
env = gym.make('Pendulum-v0')
#%%
# Dataset parameters
n_samples = 1000
n_timesteps = 100
# is the reward handled as observation?
learned_reward = True

#%%
n_latent_dim = 2
HU_enc = 100
HU_dec = 100
mb_size = n_samples
#%% 
# Initial dataset creation
# X.shape: (100, 1000, 4); U.shape:(100, 1000,1). The 4 dimensions correspond to
# cosine and sine of angle alpha, angular velocity and reward. 
# U is the one dimensional control signal at each time step. 
X, U = rollout(env, n_samples, n_timesteps, learned_reward=learned_reward, fn_action=None)
X_mean = X.reshape((-1, X.shape[2])).mean(0)
X = X - X_mean
X_std = X.reshape((-1, X.shape[2])).std(0)
X = X / X_std
# 4 dimensions and the control signal combined would be the input variable. 
XU = np.concatenate((X, U), -1)

#%% ENCODER
X_dim = XU.shape[2] # Input data dimension 
_X, z = inputs(X_dim, n_latent_dim, n_timesteps)
nne = STORN(X_dim, n_timesteps, HU_enc, HU_dec, n_latent_dim, mb_size)
z_mu, z_logvar = nne.encoder_rnn(_X)
z_var = tf.exp(z_logvar)
#%% SAMPLING
# Sample the latent variables from the posterior using z_mu and z_logvar. 
# Reparametrization trick is implicit in this step. Reference: Section 3 Kingma et al (2013).
z0 = nne.reparametrize_z(z_mu, z_var)
#%% DECODER
x_recons = nne.decoder_rnn(z0)


    

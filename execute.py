import os
os.chdir('/home/abhishek/Desktop/Projects/tensorflow_11/master-thesis/my_public_repos/storn_dvbf/')

activate_this = '../../.././venv/bin/activate_this.py'
execfile(activate_this, dict(__file__ = activate_this))

import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # To build TF from source. Supposed to speed up the execution by 4-8x. 
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
from tensorflow.python import debug as tf_debug
#%%
from storn import STORN
from dataset import *
import utilities
import train
import helper_functions
#%% 
# env = gym.make('Pendulum-v0')
#%%
# Dataset parameters
n_samples = 1000
n_timesteps = 100
# is the reward handled as observation?
learned_reward = True
#%% NN params
n_latent_dim = 2
HU_enc = 100
HU_dec = 100
mb_size = 3
learning_rate = 0.0001
training_epochs = 5
display_step = 1
#%%
XU = pickle.load(open('./pickled_data/XU.pkl', "rb"))
#%% ENCODER
X_dim = XU.train.full_data.shape[2] # Input data dimension 
_X, z = utilities.inputs(X_dim, n_latent_dim, n_timesteps)
nne = STORN(X_dim, n_timesteps, HU_enc, HU_dec, n_latent_dim, mb_size)
z_mu, z_logvar = nne.encoder_rnn(_X)
z_var = tf.exp(z_logvar)
#%% SAMPLING
# Sample the latent variables from the posterior using z_mu and z_logvar. 
# Reparametrization trick is implicit in this step. Reference: Section 3 Kingma et al (2013).
z0 = nne.reparametrize_z(z_mu, z_var)
#%% DECODER
x_recons = nne.decoder_rnn(z0)
#%% LOSS
loss_op = utilities.vanilla_vae_loss(_X, x_recons, z_mu, z_var)
solver = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)
#%% TF Session
sess = tf.InteractiveSession()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(tf.global_variables_initializer())
# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
tf.summary.FileWriter("tf_logs", graph=sess.graph)
#%% TRAIN
average_cost = train.train(sess, loss_op, solver, training_epochs, n_samples, learning_rate, 
                     mb_size, display_step, _X, XU)

#%% PLOTS
# Prepare data for plotting
cos_actual = helper_functions.sliceFrom3DTensor(XU.train.next_batch(10), 0)
sine_actual = helper_functions.sliceFrom3DTensor(XU.train.next_batch(10), 1)
w_actual =  helper_functions.sliceFrom3DTensor(XU.train.next_batch(10), 2) # Angular velocity omega
reward_actual = helper_functions.sliceFrom3DTensor(XU.train.next_batch(10), 1)
    


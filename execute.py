import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # To build TF from source. Supposed to speed up the execution by 4-8x. 
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
# from tensorflow.python import debug as tf_debug

from storn import STORN
from dataset import *
import utilities
import train
import helper_functions
import plots


# Dataset parameters
n_samples = 1000
n_timesteps = 100
learned_reward = True # is the reward handled as observation?
# NN params
n_latent_dim = 2
HU_enc = 100
HU_dec = 100
mb_size = 400
learning_rate = 0.0001
training_epochs = 5
display_step = 1
model_path = "./output_models/model.ckpt" # Manually create the directory

# DATASET
XU = pickle.load(open('./pickled_data/XU.pkl', "rb"))
shuffled_data = pickle.load(open('./pickled_data/shuffled_data.pkl', "rb"))
datasets = pickle.load(open('./pickled_data/datasets.pkl', "rb"))

# ENCODER
X_dim = datasets.train.full_data.shape[2] # Input data dimension 
_X, z = utilities.inputs(X_dim, n_latent_dim, n_timesteps)
nne = STORN(X_dim, n_timesteps, HU_enc, HU_dec, n_latent_dim, mb_size)
z_mu, z_logvar = nne.encoder_rnn(_X)
z_var = tf.exp(z_logvar)

# SAMPLING
# Sample the latent variables from the posterior using z_mu and z_logvar. 
# Reparametrization trick is implicit in this step. Reference: Section 3 Kingma et al (2013).
z0 = nne.reparametrize_z(z_mu, z_var)

# DECODER
x_recons = nne.decoder_rnn(z0)

# LOSS
loss_op = utilities.vanilla_vae_loss(_X, x_recons, z_mu, z_var)
solver = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

# Initializing the TensorFlow variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

sess = tf.InteractiveSession()
# sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(init)
# sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
tf.summary.FileWriter("tf_logs", graph=sess.graph)

# TRAINING
average_cost = train.train(sess, loss_op, solver, training_epochs, n_samples,
                           learning_rate, mb_size, display_step, _X, datasets)
save_path = saver.save(sess, model_path)
print "Model saved in file: %s" % save_path

# Restore model
saver.restore(sess, model_path)
print("Model restored from file: %s" % save_path)




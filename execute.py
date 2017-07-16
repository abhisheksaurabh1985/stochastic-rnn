import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # To build TF from source. Supposed to speed up the execution by 4-8x. 
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
from tensorflow.python import debug as tf_debug
from matplotlib import pyplot as plt

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
HU_enc = 128
HU_dec = 120
mb_size = 8
learning_rate = 0.0001
training_epochs = 4
display_step = 1
model_path = "./output_models/model.ckpt" # Manually create the directory
logs_path = './tf_logs/'

# DATASET
XU = pickle.load(open('./pickled_data/XU.pkl', "rb"))
shuffled_data = pickle.load(open('./pickled_data/shuffled_data.pkl', "rb"))
datasets = pickle.load(open('./pickled_data/datasets.pkl', "rb"))

# ENCODER
X_dim = datasets.train.full_data.shape[2] # Input data dimension 
_X, z = utilities.inputs(X_dim, n_latent_dim, n_timesteps)
nne = STORN(X_dim, n_timesteps, HU_enc, HU_dec, n_latent_dim, mb_size)
z_mu, z_logvar = nne.encoder_rnn(_X) # Shape:(T,B,z_dim)
z_var = tf.exp(z_logvar)

# SAMPLING
# Sample the latent variables from the posterior using z_mu and z_logvar. 
# Reparametrization trick is implicit in this step. Reference: Section 3 Kingma et al (2013).
z0 = nne.reparametrize_z(z_mu, z_var)

# DECODER
x_recons = nne.decoder_rnn(z0) # Shape: (T,B,x_dim)

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

# Summary to monitor cost tensor
# tf.summary.scalar("loss", loss_op)

# Create summary to visualise weights
# for var in tf.trainable_variables():
#     tf.summary.histogram(var.name, var)

# Merge all summaries into a single op
# merged_summary_op = tf.summary.merge_all()
tf.summary.FileWriter(logs_path, graph=sess.graph)

# TRAINING
average_cost = train.train(sess, loss_op, solver, training_epochs, n_samples,
                           learning_rate, mb_size, display_step, _X, datasets)
# save_path = saver.save(sess, model_path)
# print "Model saved in file: %s" % save_path
# sess.close()

# RESTORE MODEL
# sess = tf.InteractiveSession()
# sess.run(init)
# saver_rest = tf.train.import_meta_graph('./output_models/model.ckpt.meta') # Describes the saved graph structure
# saver_rest.restore(sess, model_path)
# print("Model restored from file: %s" % save_path)

# RECONSTRUCTION
x_sample = datasets.train.next_batch(mb_size)
print "x_sample.shape", x_sample.shape

# latent_for_x_sample = nne.get_latent(sess, _X, x_sample)
# print "latent sample shape", latent_for_x_sample.shape

x_reconstructed = nne.reconstruct(sess, _X, x_sample)
print "x_reconstructed type", type(x_reconstructed)
print "x_reconstructed shape", x_reconstructed.shape

#PLOTS
##Prepare data for plotting
cos_actual = helper_functions.sliceFrom3DTensor(x_sample, 0)
sine_actual = helper_functions.sliceFrom3DTensor(x_sample, 1)
w_actual =  helper_functions.sliceFrom3DTensor(x_sample, 2) # Angular velocity omega
reward_actual = helper_functions.sliceFrom3DTensor(x_sample, 3)

cos_recons = helper_functions.sliceFrom3DTensor(x_reconstructed , 0)
sine_recons = helper_functions.sliceFrom3DTensor(x_reconstructed, 1)
w_recons =  helper_functions.sliceFrom3DTensor(x_reconstructed, 2) # Angular velocity omega
reward_recons = helper_functions.sliceFrom3DTensor(x_reconstructed, 3)

# Plot cosine: actual, reconstruction and generative sampling
time_steps = range(n_timesteps)

plt.figure()
plots.line_plot_2d(time_steps, cos_actual[:,0])
plots.line_plot_2d(time_steps, cos_recons[:,0])
plt.title("Cosine at all time steps")
plt.show()

plt.figure()
plots.line_plot_2d(time_steps, sine_actual[:,0])
plots.line_plot_2d(time_steps, sine_recons[:,0])
plt.title("Sine at all time steps")
plt.show()

plt.figure()
plots.line_plot_2d(time_steps, w_actual[:,0])
plots.line_plot_2d(time_steps, w_recons[:,0])
plt.title("Angular velocity at all time steps")
plt.show()

plt.figure()
plots.line_plot_2d(time_steps, reward_actual[:,0])
plots.line_plot_2d(time_steps, reward_recons[:,0])
plt.title("Reward at all time steps")
plt.show()
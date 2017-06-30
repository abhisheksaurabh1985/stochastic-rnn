import tensorflow as tf

def inputs(D, Z, time_steps):
    """
    D: Input dimension
    Z: Latent space dimension
    """
    X = tf.placeholder(tf.float32, shape = [time_steps, None, D], 
                       name = 'input_data')
    z = tf.placeholder(tf.float32, shape = [time_steps, None, Z], 
                       name = 'latent_var')
    return X, z

def vanilla_vae_loss(x, x_reconstr, z_mu, z_var):
    reconstr_loss = -tf.reduce_sum(x * tf.log(1e-10 + x_reconstr)
                       + (1-x) * tf.log(1e-10 + 1 - x_reconstr), 1) 
    latent_loss = -0.5 * tf.reduce_sum(1 + tf.log(z_var) - tf.square(z_mu) - \
                         tf.exp(tf.log(z_var)), 1)
    loss = tf.reduce_mean(reconstr_loss + latent_loss)
    return loss

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

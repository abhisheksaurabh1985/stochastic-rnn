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

#def reconstr_vae_loss_per_tstep(x, x_reconstr):
#    reconstr_loss = -tf.reduce_mean(x * tf.log(1e-10 + x_reconstr)
#                       + (1-x) * tf.log(1e-10 + 1 - x_reconstr), 1)
#    return reconstr_loss
#
#def latent_loss_per_tstep(z_mu, z_var):
#    latent_loss = -0.5 * tf.reduce_mean(1 + tf.log(z_var) - tf.square(z_mu) - \
#                         tf.exp(tf.log(z_var)), 1)
#    return latent_loss

#def vanilla_vae_loss(x, x_reconstr, z_mu, z_var):
#    time_steps = x.shape[0]
#    init_reconstr_loss = tf.zeros((time_steps, 1))
#    reconstr_loss = tf.scan(reconstr_vae_loss_per_tstep, [x, x_reconstr], 
#                            initializer = init_reconstr_loss)
#    print "reconstr_loss shape:", reconstr_loss.get_shape()
#    init_latent_loss = tf.zeros((time_steps, 1))
#    latent_loss = tf.scan(latent_loss_per_tstep, [z, z_mu], 
#                          initializer = init_latent_loss)
#    print "latent_loss shape:", latent_loss.get_shape()
#    loss = tf.reduce_mean(reconstr_loss + latent_loss)
#    return loss
#
def vanilla_vae_loss(x, x_reconstr, z_mu, z_var):
    print "x shape:", x.get_shape()
    print "x_reconstr:", x_reconstr.get_shape()
    print "z_mu:", z_mu.get_shape()
    print "z_var:", z_var.get_shape()
    reconstr_loss = -tf.reduce_mean(x * tf.log(1e-10 + x_reconstr)
                       + (1-x) * tf.log(1e-10 + 1 - x_reconstr), axis = None)
    print "reconstr loss:", reconstr_loss.get_shape()
    latent_loss = -0.5 * tf.reduce_mean(1 + tf.log(z_var) - tf.square(z_mu) - \
                         tf.exp(tf.log(z_var)), axis = None)
    print "latent loss:", latent_loss.get_shape()
    loss = tf.reduce_mean(reconstr_loss + latent_loss)
    return loss

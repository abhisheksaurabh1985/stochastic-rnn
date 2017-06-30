import numpy as np
import tensorflow as tf

class STORN(object):
    def __init__(self, data_dim, time_steps, n_hidden_units_enc, n_hidden_units_dec, 
                 n_latent_dim, batch_size, learning_rate = 0.001, 
                 mu_init = 0, sigma_init = 1):
        self.data_dim = data_dim
        self.time_steps = time_steps
        self.n_hidden_units_enc = n_hidden_units_enc
        self.n_hidden_units_dec = n_hidden_units_dec
        self.n_latent_dim = n_latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        
        # Initializers for encoder parameters
        self.init_wxhe = tf.random_normal((self.n_hidden_units_enc, self.data_dim), 
                                           mean = self.mu_init, 
                                           stddev = self.sigma_init)
        self.init_whhe = tf.random_normal((self.n_hidden_units_enc, self.n_hidden_units_enc), 
                                           mean = self.mu_init, 
                                           stddev = self.sigma_init)
        self.init_bhe = tf.zeros((self.n_hidden_units_enc, 1))
        self.init_whmu = tf.random_normal((self.n_latent_dim, self.n_hidden_units_enc), 
                                           mean = self.mu_init, 
                                           stddev = self.sigma_init)
        self.init_bhmu = tf.zeros(self.n_latent_dim, 1)
        self.init_whsigma = tf.random_normal((self.n_latent_dim, self.n_hidden_units_enc), 
                                              mean = self.mu_init, 
                                              stddev = self.sigma_init)
        self.init_bhsigma = tf.zeros(self.n_latent_dim, 1)
        
        # Initializers for the decoder parameters
        self.init_dec_wzh = tf.random_normal((self.n_hidden_units_dec, self.n_latent_dim),  
                                      mean = self.mu_init, 
                                      stddev = self.sigma_init)
        self.init_dec_bzh = tf.zeros((self.n_hidden_units_dec, 1))
        self.init_dec_whhd = tf.random_normal((self.n_hidden_units_dec, self.n_hidden_units_dec),  
                                      mean = self.mu_init, 
                                      stddev = self.sigma_init)
        self.init_dec_wxhd = tf.random_normal((self.n_hidden_units_dec, self.data_dim),  
                                      mean = self.mu_init, 
                                      stddev = self.sigma_init)
        self.init_dec_bhd = tf.zeros((self.n_hidden_units_dec, 1))
        self.init_dec_whx = tf.random_normal((self.data_dim, self.n_hidden_units_dec),  
                                      mean = self.mu_init,  
                                      stddev = self.sigma_init)
        self.dec_bhx = tf.zeros((self.data_dim, 1))

    def encoding_step(self, h_t, x_t):
        print "self W_xhe shape:", self.W_xhe.get_shape()
        print "self W_hhe shape:", self.W_hhe.get_shape()
        print "h_t shape:", h_t.get_shape()
        print "x_t shape:", x_t.get_shape()
        print "b_he shape:", self.b_he.get_shape()
        first_term = tf.tensordot(self.W_xhe, tf.cast(x_t, tf.float32), axes=[[1],[1]])
        print "first_term shape", first_term.get_shape()
        second_term = tf.tensordot(self.W_hhe, tf.cast(h_t, tf.float32), axes=[[1],[1]])
        print "second_term shape", second_term.get_shape()
        output_encoding_step = tf.transpose(tf.tanh(first_term + second_term + self.b_he))
        print "output_encoding_shape", output_encoding_step.get_shape()
        return output_encoding_step
        
    def reparametrize_z(self, z_mu, z_var):
        eps = tf.random_normal(shape=tf.shape(z_mu), mean= 0, stddev= 1)
        print "eps shape", eps.get_shape()
        z = tf.add(z_mu, tf.multiply(tf.sqrt(z_var), eps))
        print "z shape", z.get_shape()
        return z 

    def encoder_rnn(self, x):
        # Parameters of the encoder network
        with tf.variable_scope('encoder_rnn'): 
#            self.W_xhe = tf.Variable('W_xhe', dtype = tf.float32, initializer = self.init_wxhe)
#            self.W_hhe = tf.Variable('W_hhe', dtype = tf.float32, initializer = self.init_whhe)
#            self.b_he = tf.Variable('b_he', dtype = tf.float32, initializer = self.init_bhe) 
#            self.W_hmu = tf.Variable('W_hmu', dtype = tf.float32, initializer = self.init_whmu)
#            self.b_hmu = tf.Variable('b_hmu', dtype = tf.float32, initializer = self.init_bhmu)
#            self.W_hsigma = tf.Variable('W_hsigma', dtype = tf.float32, initializer = self.init_whsigma)
#            self.b_hsigma = tf.Variable('b_hsigma', dtype = tf.float32, initializer = self.init_bhsigma)

            self.W_xhe = tf.Variable(initial_value = self.init_wxhe, name = "W_xhe", dtype = tf.float32)
            self.W_hhe = tf.Variable(initial_value = self.init_whhe, name = "W_hhe", dtype = tf.float32)
            self.b_he = tf.Variable(initial_value = self.init_bhe, name = "b_he", dtype = tf.float32) 
            self.W_hmu = tf.Variable(initial_value = self.init_whmu, name = "W_hmu", dtype = tf.float32)
            self.b_hmu = tf.Variable(initial_value = self.init_bhmu, name = "b_hmu", dtype = tf.float32)
            self.W_hsigma = tf.Variable(initial_value = self.init_whsigma, name = "W_hsigma", dtype = tf.float32)
            self.b_hsigma = tf.Variable(initial_value = self.init_bhsigma, name = "b_hsigma", dtype = tf.float32)
            # Number of time steps
            states_0 = tf.zeros([self.batch_size, self.n_hidden_units_enc], tf.float32) 
            print "states_0 shape", states_0.get_shape()
            states = tf.scan(self.encoding_step, x, initializer = states_0, name = 'states')
            print "states shape", states.get_shape()
            # Reshape states
            _states = tf.reshape(states, [-1, self.n_hidden_units_enc])
            print "_states shape", _states.get_shape()
            print "W_hmu shape", self.W_hmu.get_shape()
            print "b_hmu shape", self.b_hmu.get_shape()

            # Parameters of the distribution
            self.mu_encoder = tf.tensordot(self.W_hmu, _states, axes=[[1],[1]])
            print "mu_encoder shape", self.mu_encoder.get_shape()
            self.mu_encoder = tf.reshape(tf.transpose(self.mu_encoder), (self.time_steps, self.batch_size, -1))             
            print "mu_encoder 3D shape", self.mu_encoder.get_shape()
            self.log_sigma_encoder = tf.tensordot(self.W_hsigma, _states, axes=[[1],[1]])
            print "log_sigma_encoder shape", self.log_sigma_encoder.get_shape()
            self.log_sigma_encoder = tf.reshape(tf.transpose(self.log_sigma_encoder), (self.time_steps, self.batch_size, -1))
            print "log_sigma_encoder shape", self.log_sigma_encoder.get_shape()
            return self.mu_encoder, self.log_sigma_encoder
                        
    def get_recons_x(self, x, rec_states):
        print "Decoding rec_states shape:", rec_states.get_shape()
        print "Decoding x shape:", x.get_shape()
        print "Decoding step W_hx shape:", self.W_hx.get_shape()
        print "Decoding step b_hx shape:", self.b_hx.get_shape()        
        x = tf.transpose(tf.nn.sigmoid(tf.tensordot(self.W_hx, rec_states, 
                                              axes = [[1],[0]]) + self.b_hx))
        print "Decoding step output shape:", x.get_shape()
        return x        

    def get_decoder_recurrent_state(self, h_t, z_t):
        print "Decoding step z_t shape:", z_t.get_shape()
        print "Decoding step h_t shape:", h_t.get_shape()
        print "Decoding step W_hhd shape:", self.W_hhd.get_shape()
        print "Decoding step W_xhd shape:", self.W_xhd.get_shape()
        print "Decoding step b_hd shape:", self.b_hd.get_shape()
        recurrent_states = tf.tanh(tf.tensordot(self.W_hhd, h_t, axes = [[1],[0]]) + \
                    tf.tensordot(self.W_zh, z_t, axes = [[1],[1]]) + self.b_zh)
        return recurrent_states
        
    def decoder_rnn(self, z):
        # Parameters of the decoder network
        with tf.variable_scope('decoder_rnn'):
            self.W_zh = tf.Variable(initial_value = self.init_dec_wzh, name = "W_zh", dtype = tf.float32)
            self.b_zh = tf.Variable(initial_value = self.init_dec_bzh, name = "b_zh", dtype = tf.float32)
            self.W_hhd = tf.Variable(initial_value = self.init_dec_whhd, name = "W_hhd", dtype = tf.float32)
            self.W_xhd = tf.Variable(initial_value = self.init_dec_wxhd, name = "W_xhd", dtype = tf.float32)
            self.b_hd = tf.Variable(initial_value = self.init_dec_bhd, name = "b_hd", dtype = tf.float32)
            self.W_hx = tf.Variable(initial_value = self.init_dec_whx, name = "W_hx", dtype = tf.float32)
            self.b_hx = tf.Variable(initial_value = self.dec_bhx, name = "b_hx", dtype = tf.float32) 
            # Initial recurrent state
            print "z0 first time step shape:", z[0,:,:].get_shape()
            initial_recurrent_state = tf.tanh(tf.tensordot(self.W_zh, z[0,:,:], axes=[[1],[1]]) + self.b_zh)
            print "Initial recurrent state shape:", initial_recurrent_state.get_shape()
            recurrent_states = tf.scan(self.get_decoder_recurrent_state, z, initializer = initial_recurrent_state, name = 'recurrent_states')
            recons_init_x = tf.zeros([self.batch_size, self.data_dim], tf.float32)
            recons_x = tf.scan(self.get_recons_x, recurrent_states, initializer = recons_init_x, name = 'recons_x')
            print "recurrent_states shape:", recurrent_states.get_shape()
            return recons_x                
                
                

                               
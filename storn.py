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
        # print "x_t extended shape:", tf.expand_dims(x_t, -1).get_shape()
        first_term = tf.tensordot(self.W_xhe, tf.cast(x_t, tf.float32), axes=[[1],[1]])
        print "first_term shape", first_term.get_shape()
        second_term = tf.tensordot(self.W_hhe, tf.cast(h_t, tf.float32), axes=[[1],[1]])
        print "second_term shape", second_term.get_shape()
        output_encoding_step = tf.transpose(tf.tanh(first_term + second_term + self.b_he))
        print "output_encoding_shape", output_encoding_step.get_shape()
        # print "W_xhe * x_t", (tf.matmul(self.W_xhe, tf.expand_dims(tf.cast(x_t, tf.float32),1))).get_shape()
        # encoding_step_output = tf.tanh(tf.matmul(self.W_xhe, tf.cast(x_t, tf.float32)) + \
        #                            tf.matmul(self.W_hhe, tf.cast(h_t, dtype = tf.float32)) + self.b_he)
        return output_encoding_step
        
    def reparametrize_z(self, z_mu, z_var):
        # z_var = tf.exp(self.log_sigma_encoder)
        eps = tf.random_normal(shape=tf.shape(z_mu), mean= 0, stddev= 1)
        print "eps shape", eps.get_shape()
        z = tf.add(z_mu, tf.multiply(tf.sqrt(z_var), eps))
        print "z shape", z.get_shape()
        return z 

    def decoding_step(self, x_t, h_t):
        """
        x_t: previous_output
        h_t: current_input
        """
        print "Decoding step x_t shape:", x_t.get_shape()
        print "Decoding step h_t shape:", h_t.get_shape()
        print "Decoding step W_hhd shape:", self.W_hhd.get_shape()
        print "Decoding step W_xhd shape:", self.W_xhd.get_shape()
        print "Decoding step b_hd shape:", self.b_hd.get_shape()
        h = tf.tanh(tf.tensordot(self.W_hhd, h_t) + tf.tensordot(self.W_xhd, x_t) + self.b_hd)
        print "Decoding step h shape:", h.get_shape()
        print "Decoding step W_hx shape:", self.W_hhd.get_shape()
        print "Decoding step b_hx shape:", self.b_hx.get_shape()        
        output_decoding_step = tf.nn.sigmoid(tf.tensordot(self.W_hx, h) + self.b_hx)
        print "Decoding step output shape:", output_decoding_step.get_shape()
        return output_decoding_step
    
    def encoder_rnn(self, x):
        # Parameters of the encoder network
        with tf.variable_scope('encoder_rnn', reuse = True): 
            self.W_xhe = tf.get_variable('W_xhe', dtype = tf.float32, initializer = self.init_wxhe)
            self.W_hhe = tf.get_variable('W_hhe', dtype = tf.float32, initializer = self.init_whhe)
            self.b_he = tf.get_variable('b_he', dtype = tf.float32, initializer = self.init_bhe) 
            self.W_hmu = tf.get_variable('W_hmu', dtype = tf.float32, initializer = self.init_whmu)
            self.b_hmu = tf.get_variable('b_hmu', dtype = tf.float32, initializer = self.init_bhmu)
            self.W_hsigma = tf.get_variable('W_hsigma', dtype = tf.float32, initializer = self.init_whsigma)
            self.b_hsigma = tf.get_variable('b_hsigma', dtype = tf.float32, initializer = self.init_bhsigma)
        
            # Number of time steps
            # T = x.shape[0]
            states_0 = tf.zeros([self.batch_size, self.n_hidden_units_enc], tf.float32) # (T + 1, self.n_hidden_units_enc))
            # states_0 = tf.zeros([None, self.n_hidden_units_enc], tf.float32) # (T + 1, self.n_hidden_units_enc))
            print "states_0 shape", states_0.get_shape()
            states = tf.scan(self.encoding_step, x, initializer = states_0, name = 'states')
            print "states shape", states.get_shape()
            # Reshape states
            _states = tf.reshape(states, [-1, self.n_hidden_units_enc])
            # _states = tf.reshape(states, [self.batch_size,-1])
            print "_states shape", _states.get_shape()
            print "W_hmu shape", self.W_hmu.get_shape()
            print "b_hmu shape", self.b_hmu.get_shape()
            # print "reshaped _states for cal mu and sigma", np.reshape(_states, [self.n_hidden_units_enc, -1]).get_shape()
            # Parameters of the distribution
            self.mu_encoder = tf.tensordot(self.W_hmu, _states, axes=[[1],[1]])
            print "mu_encoder shape", self.mu_encoder.get_shape()
            self.mu_encoder = tf.reshape(tf.transpose(self.mu_encoder), (self.time_steps, self.batch_size, -1))             
            print "mu_encoder 3D shape", self.mu_encoder.get_shape()
            # mu_encoder = tf.matmul(_states, tf.transpose(self.W_hmu)) + self.b_hmu
            # self._mu_encoder = tf.reshape(mu_encoder, [self.time_steps, None, -1])
            # log_sigma_encoder = tf.matmul(self.W_hsigma, _states) + self.b_hsigma  
            # self._log_sigma_encoder = tf.reshape(log_sigma_encoder, [self.time_steps, None, -1])            
            self.log_sigma_encoder = tf.tensordot(self.W_hsigma, _states, axes=[[1],[1]])
            print "log_sigma_encoder shape", self.log_sigma_encoder.get_shape()
            self.log_sigma_encoder = tf.reshape(tf.transpose(self.log_sigma_encoder), (self.time_steps, self.batch_size, -1))
            print "log_sigma_encoder shape", self.log_sigma_encoder.get_shape()
            # self._log_sigma_encoder =              
            # self.z = self.reparametrize_z()
            return self.mu_encoder, self.log_sigma_encoder
                        
    def decoder_rnn(self, z):
        # Parameters of the decoder network
        with tf.variable_scope('decoder_rnn', reuse = True):
            self.W_zh = tf.get_variable('W_zh', dtype = tf.float32, initializer = self.init_dec_wzh)
            self.b_zh = tf.get_variable('b_zh', dtype = tf.float32, initializer = self.init_dec_bzh)
            self.W_hhd = tf.get_variable('W_hhd', dtype = tf.float32, initializer = self.init_dec_whhd)
            self.W_xhd = tf.get_variable('W_xhd', dtype = tf.float32, initializer = self.init_dec_wxhd)
            self.b_hd = tf.get_variable('b_hd', dtype = tf.float32, initializer = self.init_dec_bhd)
            self.W_hx = tf.get_variable('W_hx', dtype = tf.float32, initializer = self.init_dec_whx)
            self.b_hx = tf.get_variable('b_hx', dtype = tf.float32, initializer = self.dec_bhx) 
            
            recons_x_0 = tf.zeros([self.batch_size, self.data_dim], tf.float32) # (T + 1, self.n_hidden_units_enc))                        
            
            recons_x = tf.scan(self.decoding_step, z, initializer = recons_x_0, name = 'recons_x')
            return recons_x                
                
                

                               
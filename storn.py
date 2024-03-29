import tensorflow as tf


class STORN(object):
    def __init__(self, data_dim, time_steps, n_hidden_units_enc, n_hidden_units_dec, n_latent_dim, batch_size,
                 learning_rate=0.001, mu_init=0, sigma_init=0.01):
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
                                          mean=self.mu_init,
                                          stddev=self.sigma_init)
        self.init_whhe = tf.random_normal((self.n_hidden_units_enc, self.n_hidden_units_enc),
                                          mean=self.mu_init,
                                          stddev=self.sigma_init)
        self.init_bhe = tf.zeros((self.n_hidden_units_enc, 1))
        self.init_whmu = tf.random_normal((self.n_latent_dim, self.n_hidden_units_enc),
                                          mean=self.mu_init,
                                          stddev=self.sigma_init)
        self.init_bhmu = tf.zeros(self.n_latent_dim, 1)
        self.init_whsigma = tf.random_normal((self.n_latent_dim, self.n_hidden_units_enc),
                                             mean=self.mu_init,
                                             stddev=self.sigma_init)
        self.init_bhsigma = tf.zeros(self.n_latent_dim, 1)

        # Initializers for the decoder parameters
        self.init_dec_wzh = tf.random_normal((self.n_hidden_units_dec, self.n_latent_dim),
                                             mean=self.mu_init,
                                             stddev=self.sigma_init)
        self.init_dec_bzh = tf.zeros((self.n_hidden_units_dec, 1))
        self.init_dec_whhd = tf.random_normal((self.n_hidden_units_dec, self.n_hidden_units_dec),
                                              mean=self.mu_init,
                                              stddev=self.sigma_init)
        self.init_dec_wxhd = tf.random_normal((self.n_hidden_units_dec, self.data_dim),
                                              mean=self.mu_init,
                                              stddev=self.sigma_init)
        self.init_dec_bhd = tf.zeros((self.n_hidden_units_dec, 1))
        self.init_dec_whx = tf.random_normal((self.data_dim, self.n_hidden_units_dec),
                                             mean=self.mu_init,
                                             stddev=self.sigma_init)
        self.dec_bhx = tf.zeros((self.data_dim, 1))

    def encoding_step(self, h_t, x_t):
        """
        The encoder has one set of recurrent connection. State h_{t+1} is calculated based on the previous
        recurrent state h_t and current input x_{t+1}.

        This function is called by the tf.scan function inside encoder_rnn.

        :param h_t: Refers to the previous output. At the start this has a shape (mb_size, n_hidden_units_enc).
        :param x_t: Refers to the current_input. At the start is has the shape (?,5).

        :return output_encoding_step: Output at the end of each run. Has a shape (6,100).
        """
        print "self W_xhe shape:", self.W_xhe.get_shape()
        print "self W_hhe shape:", self.W_hhe.get_shape()
        print "h_t shape:", h_t.get_shape()
        print "x_t shape:", x_t.get_shape()
        print "b_he shape:", self.b_he.get_shape()
        first_term = tf.tensordot(self.W_xhe, tf.cast(x_t, tf.float32), axes=[[1], [1]], name="enc_first_term")
        print "first_term shape", first_term.get_shape()
        second_term = tf.tensordot(self.W_hhe, tf.cast(h_t, tf.float32), axes=[[1], [1]], name="enc_second_term")
        print "second_term shape", second_term.get_shape()
        output_encoding_step = tf.transpose(tf.tanh(first_term + second_term + self.b_he),  name="output_encoding_step")
        print "output_encoding_shape", output_encoding_step.get_shape()
        return output_encoding_step

    def reparametrize_z(self, z_mu, z_var):
        """
        Sampling from a normal distribution with the mean and sigma given.

        :param z_mu: Mean of the distribution for each item in the mini-batch for each time step.
                     Has a shape (T,B,D) where T, B and D refer to time step, batch size and dimension respectively.
        :param z_var: Standard deviation for each item in the mini-batch for each time step. Dimension
                      same as that of z_mu.
        :return: self.z: Sampled (aka reparametrized) z with shape (T,B,D).
        """
        eps = tf.random_normal(shape=tf.shape(z_mu), mean=0, stddev=1)  # Shape: (100,6,2)
        print "eps shape", eps.get_shape()
        self.z = tf.add(z_mu, tf.multiply(tf.sqrt(z_var), eps))  # Shape: (100,6,2)
        print "z shape", self.z.get_shape()
        return self.z

    def encoder_rnn(self, x):
        """
        RNN as an encoder network in STORN. For a given input x it returns a compressed representation
        in the latent space.

        :param x: Input time series data with dimension (T,B,D).

        :return self.mu_encoder: Mean of the data for each item in the batch at each time step. Has a shape (T,B,D).
        :return self.log_sigma_encoder: Standard deviation of the data for each item in the batch at each time step. Has
         a shape (T,B,D).
        """
        # Parameters of the encoder network
        with tf.variable_scope('encoder_rnn'):
            self.W_xhe = tf.Variable(initial_value=self.init_wxhe, name="W_xhe", dtype=tf.float32)
            self.W_hhe = tf.Variable(initial_value=self.init_whhe, name="W_hhe", dtype=tf.float32)
            self.b_he = tf.Variable(initial_value=self.init_bhe, name="b_he", dtype=tf.float32)
            self.W_hmu = tf.Variable(initial_value=self.init_whmu, name="W_hmu", dtype=tf.float32)
            self.b_hmu = tf.Variable(initial_value=self.init_bhmu, name="b_hmu", dtype=tf.float32)
            self.W_hsigma = tf.Variable(initial_value=self.init_whsigma, name="W_hsigma", dtype=tf.float32)
            self.b_hsigma = tf.Variable(initial_value=self.init_bhsigma, name="b_hsigma", dtype=tf.float32)

            # Number of time steps
            states_0 = tf.zeros([tf.shape(x)[1], self.n_hidden_units_enc], tf.float32, name="enc_states_0")
            print "states_0 shape", states_0.get_shape()
            print "x shape", x.get_shape()  # (100, ?, 5)
            states = tf.scan(self.encoding_step, x, initializer=states_0, name='states')
            print "states shape", states.get_shape()  # shape:(timeSteps,miniBatchSize,nUnitsEncoder)

            # Reshape states
            _states = tf.reshape(states, [-1, self.n_hidden_units_enc],
                                 name="encoder_states")  # Shape:(timeSteps * miniBatchSize, nUnitsEncoder)
            print "_states shape", _states.get_shape()
            print "W_hmu shape", self.W_hmu.get_shape()
            print "b_hmu shape", self.b_hmu.get_shape()

            # Parameters of the distribution
            self.mu_encoder = tf.tensordot(self.W_hmu, _states, axes=[[1], [1]], name="mu_encoder")
            print "mu_encoder shape", self.mu_encoder.get_shape()  # Shape:(z_dim, timeSteps*miniBatchSize)
            self.mu_encoder = tf.reshape(tf.transpose(self.mu_encoder),
                                         (self.time_steps, self.batch_size, -1),
                                         name="reshaped_mu_encoder")  # Shape:(timeSteps, miniBatchSize, z_dim)
            print "mu_encoder 3D shape", self.mu_encoder.get_shape()
            self.log_sigma_encoder = tf.tensordot(self.W_hsigma, _states,
                                                  axes=[[1], [1]],
                                                  name="log_sigma_encoder")  # Shape:(z_dim, timeSteps*miniBatchSize)
            print "########"
            print "log_sigma_encoder shape", self.log_sigma_encoder.get_shape()
            print "########"
            self.log_sigma_encoder = tf.reshape(tf.transpose(self.log_sigma_encoder),
                                                (self.time_steps, self.batch_size, -1),
                                                name="reshaped_log_sigma_encoder")  # Shape:(timeSteps, miniBatchSize, z_dim)
            print "########"
            print "log_sigma_encoder shape", self.log_sigma_encoder.get_shape()
            print "########"
            return self.mu_encoder, self.log_sigma_encoder

    def decoding_step(self, previous_output, z_t):
        """
        Returns the recurrent state at the previous time step and the reconstruction from the data in the latent space.
        Executed by the tf.scan function in decoder_rnn for as many times as there are time steps (eqv to the first
        dimension of z when the function call is made). Data at each time step is used one at a time. Therefore, the
        dimension of z_t is (mini_batch,nDimLatentSpace).

        First input is the previous output, initialized at the time of function call. Second is the current input i.e.
        input in the latent space which is to be reconstructed.

        :param previous_output: Recurrent states and the data to be reconstructed are the previous output which will be
        calculated at each time step based on the input at each time step i.e. z_t.
        :param z_t: Point in the latent space which will be reconstructed. Shape in each iteration of tf.scan is
        (mini_batch, nDimLatentSpace). tf.scan will iterate as many times as there are time steps.
        :return h: Recurrent state outputted at the previous time step i.e. $h_{t-1}$. Has a shape
        (mini_batch, numUnitsDecoder).
        :return x: Reconstruction at each time step. Has a shape (mini_batch, data_dimensions).
        """
        # First element is the initial recurrent state. Second is the initial reconstruction, which isn't needed.
        h_t, _ = previous_output
        print "Decoding step z_t shape", z_t.get_shape()
        print "Decoding step h_t shape", h_t.get_shape()
        print "Decoding step W_hhd shape", self.W_hhd.get_shape()
        # print "Decoding step W_xhd shape", self.W_xhd.get_shape()
        print "Decoding step b_hd shape", self.b_hd.get_shape()

        # W_hhd:(100,100); h_t:(100,6); W_xhd: (100,5); x_t:(6,5); b_hd:(100,1)
        h = tf.transpose(tf.tanh(tf.tensordot(self.W_hhd, h_t, axes=[[1], [1]], name="dec_rec_first_term_h") +
                                 tf.tensordot(self.W_zh, z_t, axes=[[1], [1]], name="dec_rec_second_term_h") +
                                 self.b_hd), name="decoding_step_tr_h")
        print "Decoding step h shape", h.get_shape()
        print "Decoding step W_hx shape", self.W_hx.get_shape()
        print "Decoding step b_hx shape", self.b_hx.get_shape()
        x = tf.transpose(tf.identity(tf.tensordot(self.W_hx, h, axes=[[1], [1]]) + self.b_hx), name="x_recons")
        print "Decoding step x shape", x.get_shape()
        return h, x

    def decoder_rnn(self, z):
        """
        Returns the input reconstructed from the compressed data obtained from the encoder.

        :param z: Compressed data obtained from the encoder post reparametrization. Has a shape (T,B,D), where D is the
        number of dimensions in the latent space.

        :return self.recons_x: Reconstructed input of shape (T,B,D) where D is the original number of dimensions.
        """
        # Parameters of the decoder network
        with tf.variable_scope('decoder_rnn'):
            self.W_zh = tf.Variable(initial_value=self.init_dec_wzh, name="W_zh", dtype=tf.float32)  # Weights for z_t.
            # self.b_zh = tf.Variable(initial_value=self.init_dec_bzh, name="b_zh", dtype=tf.float32)
            self.W_hhd = tf.Variable(initial_value=self.init_dec_whhd, name="W_hhd", dtype=tf.float32)  # W_rec
            # self.W_xhd = tf.Variable(initial_value=self.init_dec_wxhd, name="W_xhd", dtype=tf.float32)
            self.b_hd = tf.Variable(initial_value=self.init_dec_bhd, name="b_hd", dtype=tf.float32)
            self.W_hx = tf.Variable(initial_value=self.init_dec_whx, name="W_hx", dtype=tf.float32)  # Weights for x_t
            self.b_hx = tf.Variable(initial_value=self.dec_bhx, name="b_hx", dtype=tf.float32)

            # Initial recurrent state
            print "z0 first time step shape:", z[0, :, :].get_shape()
            # Compute initial state of the decoding RNN with one set of weights.
            print "z shape decoder_rnn ", z.get_shape()
            print "self.W_zh in decoder_rnn shape", self.W_zh.get_shape()
            # print "self.b_zh in decoder_rnn shape", self.b_zh.get_shape()
            print "reshaped transpose tf.tensordot(self.W_zh, z, axes=[[1],[2]]::", \
                tf.reshape(tf.tensordot(self.W_zh, z, axes=[[1], [2]]),
                           [self.time_steps, -1, self.batch_size]).get_shape()

            # Iterate over each item in the latent space. Initializer will be the first recurrrent state i.e. h0 and the
            # reconstruction at the first time step.
            initial_recurrent_state = tf.random_normal((self.batch_size, self.n_hidden_units_dec),
                                                       mean=0,
                                                       stddev=1,
                                                       name="dec_init_rec_state")
            recons_init_x = tf.random_normal((self.batch_size, self.data_dim),
                                             mean=0,
                                             stddev=1,
                                             name="dec_recons_init_x")
            _, self.recons_x = tf.scan(self.decoding_step,
                                       z, initializer=(initial_recurrent_state, recons_init_x),
                                       name='recons_x')
            print "recons x shape", self.recons_x.get_shape()
            return self.recons_x

    def reconstruct(self, sess, x, data):
        return sess.run(self.recons_x, feed_dict={x: data})

    def get_latent(self, sess, x, data):
        return sess.run(self.z, feed_dict={x: data})

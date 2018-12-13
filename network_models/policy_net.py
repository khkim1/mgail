import tensorflow as tf
import numpy as np
from network_models.loss import l2_loss, l2_loss_masked


class Policy_net:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """

        ob_space = env.observation_space
        act_space = env.action_space

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape), name='obs')

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=act_space.n, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=layer_3, units=act_space.n, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=20, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class Model_Policy_net:
    def __init__(self, name: str, env, obs_dim, act_dim, num_hidden=50, depth=4, lr=1e-4):
        """
        :param name: string
        :param env: gym env
        """

        ob_space = env.observation_space
        act_space = env.action_space

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.)
        init = tf.random_normal_initializer(stddev=0.1)
        #init = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + [obs_dim], name='obs')
            self.tv = tf.placeholder(dtype=tf.float32, shape=[None] + [act_dim+1], name='tv')

            with tf.variable_scope('policy_net'):
                #layer_1 = tf.layers.dense(inputs=self.obs, units=64, activation=tf.tanh)
                #layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.tanh)

                l = tf.layers.dense(inputs=self.obs,
                    units=num_hidden,
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer,
                    kernel_initializer=init,
                    bias_initializer=init,
                    activation=tf.nn.relu)

                for i in range(depth-1):
                    l = tf.layers.dense(inputs=l,
                                        units=num_hidden,
                                        kernel_regularizer=regularizer,
                                        bias_regularizer=regularizer,
                                        kernel_initializer=init,
                                        bias_initializer=init,
                                        activation=tf.nn.relu)

                self.means = tf.layers.dense(inputs=l, units=act_dim, activation=None,
                                             kernel_initializer=init,
                                             bias_initializer=init,)

                log_vars = tf.get_variable(name='logvars',
                                                shape=(100, act_dim),
                                                dtype=tf.float32,
                                                initializer=tf.constant_initializer(0.),
                                                trainable=True)

                log_std = tf.get_variable(name='logstd',
                                                shape=(200, act_dim),
                                                dtype=tf.float32,
                                                initializer=tf.constant_initializer(0.),
                                                trainable=True)

                self.log_vars = tf.reduce_sum(log_vars, axis=0, keep_dims=True) + [-2., -2., -2., -2.]
                self.log_std = tf.reduce_sum(log_std, axis=0, keep_dims=True)

                self.diff = self.means - self.obs[:, :4]


            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=64, activation=tf.nn.relu)
                layer_2 = tf.layers.dense(inputs=layer_1, units=64, activation=tf.nn.relu)
                #layer_3 = tf.layers.dense(inputs=layer_2, units=64, activation=tf.nn.relu)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            # Get action samples
            batch = tf.shape(self.obs)[0]

            use_var = True
            if use_var:
                batch_log_vars = tf.tile(self.log_vars, [batch, 1])
                std = tf.exp(batch_log_vars / 2.0)
            else:
                batch_log_std = tf.tile(self.log_std, [batch, 1])
                std = tf.exp(batch_log_std)

            eps = tf.random_normal(shape=(batch, act_dim))
            assert std.get_shape().as_list() == eps.get_shape().as_list() == self.means.get_shape().as_list()

            self.act_stochastic = self.means + std * eps # Reparametrization trick
            self.act_deterministic = self.means

            self.scope = tf.get_variable_scope().name

            # Direct supervised learning
            self.loss = l2_loss_masked(self.act_stochastic, self.tv)
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

    def step(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def train_sl(self, given, tv):
        return tf.get_default_session().run([self.train_op, self.loss], feed_dict={self.obs: given, self.tv: tv})

    def get_sl_loss(self, given, tv):
        return tf.get_default_session().run(self.loss, feed_dict={self.obs: given, self.tv: tv})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)


class Model_net:
    def __init__(self, name, env, num_obs_per_state, lr, depth, num_hidden):
        """
        :param name: string
        :param env: gym env
        """

        ob_space = env.observation_space
        act_space = env.action_space

        assert type(name) == str

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.)

        with tf.variable_scope(name):
            # Observation is state + action
            self.given = tf.placeholder(dtype=tf.float32, shape=[None] + [5*num_obs_per_state], name='given')
            self.tv = tf.placeholder(dtype=tf.float32, shape=[None] + [4+1], name='tv')

            with tf.variable_scope('model_net'):
                l = tf.layers.dense(inputs=self.given,
                                    units=num_hidden,
                                    kernel_regularizer=regularizer,
                                    bias_regularizer=regularizer,
                                    activation=tf.nn.relu)

                for i in range(depth-1):
                    l = tf.layers.dense(inputs=l,
                                        units=num_hidden,
                                        kernel_regularizer=regularizer,
                                        bias_regularizer=regularizer,
                                        activation=tf.nn.relu)

                self.pred = tf.layers.dense(inputs=l, units=4, activation=None)
                #self.state_mean = tf.layers.dense(inputs=layer_3, units=4, activation=tf.nn.softmax)
                #self.state_sigma = tf.layers.dense(inputs=layer_3, units=4, activation=tf.nn.softmax)
                #self.pred = tf.layers.dense(inputs=layer_3, units=4, activation=tf.nn.softmax)

            self.loss = l2_loss_masked(self.pred, self.tv)
            #self.loss = l2_loss(self.pred, self.tv)
            self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)
            self.scope = tf.get_variable_scope().name

    def step(self, given, stochastic=True):
        return tf.get_default_session().run(self.pred, feed_dict={self.given: given})

    def train_sl(self, given, tv):
        return tf.get_default_session().run([self.train_op, self.loss], feed_dict={self.given: given, self.tv: tv})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

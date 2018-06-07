import tensorflow as tf
import numpy as np
import scipy.optimize
from sklearn.utils import shuffle


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)


# =============================================
#               Policy Network
# =============================================
class PolicyNet(object):
    def __init__(self, config, env, scope='policy'):
        self.scope = scope
        self.net_size = config.policy_net_size
        self.init_log_var = config.init_log_var
        self.obs_dim = env.ob_dim + 1  # +1 for time dim
        self.act_dim = env.ac_dim

        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.scope):
            self._placeholders()
            self._policy_net()
            self._log_probs()
            self._kl_entropy()
            self._sample()
            self._losses()

    def _placeholders(self):
        # observations and actions, recorded and taken, with old the policy
        self.obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_dim), name='obs_ph')
        self.act_ph = tf.placeholder(tf.float32, shape=(None, self.act_dim), name='act_ph')
        self.adv_ph = tf.placeholder(tf.float32, shape=(None,), name='adv_ph')
        # log_vars and means with pi_old (previous step's policy parameters):
        self.old_log_vars_ph = tf.placeholder(tf.float32, (self.act_dim,), 'old_log_vars_ph')
        self.old_means_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'old_means_ph')

    def _policy_net(self):
        """
        constructs the main network of the policy. includes two parts:
            * the part that spits out the mean
            * variance on top of it
        """
        # mean
        out = self.obs_ph
        init_heu = self.obs_dim
        for i, hid_size in enumerate(self.net_size):
            name = 'h' + str(i)
            out = tf.layers.dense(out, hid_size, tf.tanh,
                                  kernel_initializer=
                                  tf.random_normal_initializer(stddev=np.sqrt(1 / init_heu)), name=name)
            init_heu = hid_size
        self.means = tf.layers.dense(out, self.act_dim, None,
                                     kernel_initializer=
                                     tf.random_normal_initializer(stddev=np.sqrt(1 / init_heu)), name="means")
        # variance
        self.log_vars = tf.get_variable('logvars', (self.act_dim,), tf.float32,
                                        tf.constant_initializer(0.0)) + self.init_log_var

    def _log_probs(self):
        """
        how probables were actions taken by the old policy, given states according to:
            * the old pi itself
            * new pi

        these probabilities should be calculated in context of a multivariate gaussian distribution, see:
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Properties
        """
        logp = -0.5 * tf.reduce_sum(self.log_vars)
        logp += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.means) /
                                     tf.exp(self.log_vars), axis=1)
        self.logp = logp

        logp_old = -0.5 * tf.reduce_sum(self.old_log_vars_ph)
        logp_old += -0.5 * tf.reduce_sum(tf.square(self.act_ph - self.old_means_ph) /
                                         tf.exp(self.old_log_vars_ph), axis=1)
        self.logp_old = logp_old

        # some constants have been dropped, due to the fact that the difference between two logs matters

    def _kl_entropy(self):
        """
        add to graph:
            1. kl divergence between old and new distributions
            2. entropy of present policy given states and actions

        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
        https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
        """
        log_det_cov_old = tf.reduce_sum(self.old_log_vars_ph)
        log_det_cov_new = tf.reduce_sum(self.log_vars)
        tr_old_new = tf.reduce_sum(tf.exp(self.old_log_vars_ph - self.log_vars))

        self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
                                       tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
                                                     tf.exp(self.log_vars), axis=1) -
                                       self.act_dim)
        self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
                              tf.reduce_sum(self.log_vars))

    def _sample(self):
        """
        sample from distribution, given observation. see:
            https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Drawing_values_from_the_distribution
        """
        self.sampled_act = (self.means +
                            tf.exp(self.log_vars / 2.0) *
                            tf.random_normal(shape=(self.act_dim,), dtype=tf.float32))

    def _losses(self):
        """
        three loss terms:
            1) standard policy gradient
            2) d_kl(pi_old || pi_new)
            3) hinge loss on [d_kl - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """

        # surrogate loss
        self.surr = -tf.reduce_mean(self.adv_ph * tf.exp(self.logp - self.logp_old))
        # kl penalty
        self.kl_pen = tf.reduce_mean(self.kl)
        # entropy penalty
        # self.entropy_pen = tf.square(tf.maximum(0.0, self.kl - 2.0 * self.kl_targ))

        # self.loss = loss1 + loss2 + loss3
        # optimizer = tf.train.AdamOptimizer(self.lr_ph)
        # self.train_op = optimizer.minimize(self.loss)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}
        return tf.get_default_session().run(self.sampled_act, feed_dict=feed_dict)

    def get_trainables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)


# =============================================
#               Value Network
# =============================================
class ValueNet(object):
    def __init__(self, config, env, logger, scope='valuefunction'):
        self.scope = scope
        self.baseline_net_size = config.baseline_net_size
        self.replay_buffer_x = None
        self.replay_buffer_y = None
        self.obs_dim = env.ob_dim + 1  # +1 for time dim
        self.epochs = 20
        self.logger = logger
        self.reg = config.reg
        self.mixfrac = config.mixfrac
        self.update_method = config.vf_update_method
        self.max_lbfgs_iter = config.max_lbfgs_iter

        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.scope):
            self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs_ph')
            self.val_ph = tf.placeholder(tf.float32, (None,), 'val_ph')
            out = self.obs_ph
            init_heu = self.obs_dim
            for i, hid_size in enumerate(self.baseline_net_size):
                name = 'h' + str(i)
                out = tf.layers.dense(out, hid_size, tf.tanh,
                                      kernel_initializer=
                                      tf.random_normal_initializer(stddev=np.sqrt(1 / init_heu)), name=name)
                init_heu = hid_size
            self.preds = tf.layers.dense(out, 1, None,
                                         kernel_initializer=
                                         tf.random_normal_initializer(stddev=np.sqrt(1 / init_heu)), name="preds")
            self.preds = tf.squeeze(self.preds)
            self.loss = tf.reduce_mean(tf.square(self.preds - self.val_ph))
            for v in tf.trainable_variables():
                self.loss += self.reg * tf.nn.l2_loss(v)
            # build gradient descent update procedure
            if self.update_method == 'GD':
                self.lr = 2e-2  # / np.sqrt(self.baseline_net_size[len(self.baseline_net_size) // 2])
                optimizer = tf.train.AdamOptimizer(self.lr)
                self.train_op = optimizer.minimize(self.loss, var_list=self.get_trainables())
            # uses l_bfgs_b optimization method of scipy
            elif self.update_method == 'LBFGS':
                self.params = self.get_trainables()  # list of all trainable variables
                self.vg = flatgrad(self.loss, self.params)  # value gradient
                self.shapes = [v.shape.as_list() for v in self.params]
                self.size_phi = np.sum([np.prod(shape) for shape in self.shapes])
                self.flat_weights = tf.concat([tf.reshape(param, [-1]) for param in self.params], axis=0)  # flattened
                self.flat_wieghts_ph = tf.placeholder(tf.float32, (self.size_phi,))
                self.assign_weights_ops = []
                start = 0
                assert len(self.params) == len(self.shapes), "messed up vf shapes"
                for i, shape in enumerate(self.shapes):
                    size = np.prod(shape)
                    param = tf.reshape(self.flat_wieghts_ph[start:start + size], shape)
                    self.assign_weights_ops.append(self.params[i].assign(param))
                    start += size
                assert start == self.size_phi, "messy vf shapes"

    def get_loss(self, x, y):
        return tf.get_default_session().run(self.loss, feed_dict={
            self.obs_ph: x,
            self.val_ph: y
        })

    def fit(self, x, y):
        if self.update_method == 'LBFGS':
            self.fit_lbfgs(x, y)
        elif self.update_method == 'GD':
            self.fit_gd(x, y)

    def fit_lbfgs(self, x, y):
        prev_phi = tf.get_default_session().run(self.flat_weights)
        y_hat = self.predict(x)
        x_train = x
        y_train = y * self.mixfrac + y_hat * (1 - self.mixfrac)
        vf_loss = self.get_loss(x_train, y_train)
        vf_error = np.mean(np.square(y_hat - y))
        # rand = np.random.random_integers(0, x.shape[0], 10)
        # print(y[rand])
        # print(y_hat[rand])
        self.logger.log({'Value Error Old': vf_error, 'Value Loss Old': vf_loss})

        def lossandgrad(phi):
            tf.get_default_session().run(self.assign_weights_ops, feed_dict={self.flat_wieghts_ph: phi})
            loss, grad = tf.get_default_session().run([self.loss, self.vg], feed_dict={
                self.obs_ph: x_train,
                self.val_ph: y_train
            })
            return loss.astype(np.float64), grad.astype(np.float64)

        phi, vf_loss, opt_info = scipy.optimize.fmin_l_bfgs_b(lossandgrad, prev_phi,
                                                              maxiter=self.max_lbfgs_iter)
        del opt_info['grad']
        print(opt_info)
        tf.get_default_session().run(self.assign_weights_ops, feed_dict={self.flat_wieghts_ph: phi})
        y_hat = self.predict(x)
        vf_error = np.mean(np.square(y_hat - y))
        self.logger.log({'Value Error Now': vf_error, 'Value Loss Now': vf_loss})

    def fit_gd(self, x, y):
        num_batches = max(x.shape[0] // 256, 1)
        batch_size = x.shape[0] // num_batches
        # if self.replay_buffer_x is None:
        x_train, y_train = x, y
        # else:
        #     x_train = np.concatenate([x, self.replay_buffer_x])
        #     y_train = np.concatenate([y, self.replay_buffer_y])
        # self.replay_buffer_x = x
        # self.replay_buffer_y = y
        y_hat = self.predict(x)
        y_train = y * self.mixfrac + y_hat * (1 - self.mixfrac)
        vf_loss = self.get_loss(x_train, y_train)
        vf_error = np.mean(np.square(y_hat - y))
        self.logger.log({'Value Error Old': vf_error, 'Value Loss Old': vf_loss})
        for e in range(self.epochs):
            x_train, y_train = shuffle(x_train, y_train)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                feed_dict = {self.obs_ph: x_train[start:end, :],
                             self.val_ph: y_train[start:end]}
                _, l = tf.get_default_session().run([self.train_op, self.loss], feed_dict=feed_dict)
        y_hat = self.predict(x)
        vf_loss = self.get_loss(x_train, y_train)
        vf_error = np.mean(np.square(y_hat - y))
        self.logger.log({'Value Error Now': vf_error, 'Value Loss Now': vf_loss})

    def predict(self, x):
        feed_dict = {self.obs_ph: x}
        y_hat = tf.get_default_session().run(self.preds, feed_dict=feed_dict)
        return np.squeeze(y_hat)

    def get_trainables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

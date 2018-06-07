import tensorflow as tf
import numpy as np


class PPOUpdater(object):
    def __init__(self, policy_net, config, logger):
        self.policy_net = policy_net
        self.logger = logger
        self.beta = config.init_beta
        self.eta = config.init_eta
        self.lr_multiplier = config.lr_multiplier
        self.lr = config.ppo_lr
        self.kl_targ = config.kl_targ
        self.epochs = config.update_epochs
        self._build_updater()

    def _build_updater(self):
        self._init_placeholders()
        self._build_loss_and_optimizer()

    def _init_placeholders(self):
        self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        self.lr_ph = tf.placeholder(tf.float32, (), 'learning_rate')
        # init learning rate with hueristic
        if self.lr is None:
            self.lr = 9e-4 / np.sqrt(self.policy_net.net_size[1])

    def _build_loss_and_optimizer(self):
        self.loss = \
            self.policy_net.surr + \
            tf.reduce_mean(self.beta_ph * self.policy_net.kl) + \
            self.eta_ph * tf.square(tf.maximum(0.0, self.policy_net.kl - 2.0 * self.kl_targ))
        optimizer = tf.train.AdamOptimizer(self.lr_ph)
        self.train_op = optimizer.minimize(self.loss)

    def __call__(self, observes, actions, advantages):
        feed_dict = {self.policy_net.obs_ph: observes,
                     self.policy_net.act_ph: actions,
                     self.policy_net.adv_ph: advantages,
                     self.beta_ph: self.beta,
                     self.eta_ph: self.eta,
                     self.lr_ph: self.lr * self.lr_multiplier}
        old_means_np, old_log_vars_np = tf.get_default_session().run([self.policy_net.means,
                                                                      self.policy_net.log_vars],
                                                                     feed_dict)
        feed_dict[self.policy_net.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.policy_net.old_means_ph] = old_means_np

        loss, kl, entropy = 0, 0, 0
        for e in range(self.epochs):
            # TODO: need to improve data pipeline - re-feeding data every epoch
            tf.get_default_session().run(self.train_op, feed_dict)
            loss, kl, entropy = tf.get_default_session().run([self.loss, self.policy_net.kl, self.policy_net.entropy],
                                                             feed_dict)
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
            self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
            if self.beta > 30 and self.lr_multiplier > 0.1:
                self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2:
            self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
            if self.beta < (1 / 30) and self.lr_multiplier < 10:
                self.lr_multiplier *= 1.5

        self.logger.log({'PolicyLoss': loss,
                         'PolicyEntropy': entropy,
                         'KL': kl,
                         'Beta': self.beta,
                         '_lr_multiplier': self.lr_multiplier})


def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    print([str(i.name) for i in not_initialized_vars]) # only for testing
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

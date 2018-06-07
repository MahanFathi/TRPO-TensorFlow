import tensorflow as tf
import numpy as np


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)


class TrpoUpdater(object):
    def __init__(self, policy_net, config, logger):
        self.policy_net = policy_net
        self.delta = config.delta
        self.cg_damping = config.cg_damping
        self.logger = logger
        self._build_updater()

    def _build_updater(self):
        self.params = self.policy_net.get_trainables()  # list of all trainable variables
        self._compute_policy_gradient()
        self._compute_hessian_vector_product()
        self._assign_vars_ops()

    def _compute_policy_gradient(self):
        self.pg = flatgrad(self.policy_net.surr, self.params)

    def _compute_hessian_vector_product(self):
        self.shapes = [v.shape.as_list() for v in self.params]
        self.size_theta = np.sum([np.prod(shape) for shape in self.shapes])
        self.p = tf.placeholder(tf.float32, (self.size_theta,))  # the vector
        grads = tf.gradients(self.policy_net.kl_pen, self.params)
        tangents = []
        start = 0
        for shape in self.shapes:
            size = np.prod(shape)
            tangents.append(tf.reshape(self.p[start:start + size], shape))
            start += size
        gvp = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zip(grads, tangents)])
        self.hvp = flatgrad(gvp, self.params)

    def _assign_vars_ops(self):
        """
        Create the process of assigning updated vars
        """
        self.flat_weights = tf.concat([tf.reshape(param, [-1]) for param in self.params], axis=0)  # flattened
        self.flat_wieghts_ph = tf.placeholder(tf.float32, (self.size_theta,))
        # self.assign_weights_op = tf.assign(self.flat_weights, self.flat_wieghts_ph)
        self.assign_weights_ops = []
        start = 0
        assert len(self.params) == len(self.shapes), "messed up shapes"
        for i, shape in enumerate(self.shapes):
            size = np.prod(shape)
            param = tf.reshape(self.flat_wieghts_ph[start:start + size], shape)
            self.assign_weights_ops.append(self.params[i].assign(param))
            start += size
        assert start == self.size_theta, "messy shapes"

    def assign_vars(self, theta):
        tf.get_default_session().run(self.assign_weights_ops, feed_dict={
            self.flat_wieghts_ph: theta
        })

    def get_flat_weights(self):
        return tf.get_default_session().run(self.flat_weights)

    def __call__(self, observes, actions, advantages):

        feed_dict = {self.policy_net.obs_ph: observes,
                     self.policy_net.act_ph: actions,
                     self.policy_net.adv_ph: advantages}
        old_means_np, old_log_vars_np = tf.get_default_session().run([self.policy_net.means,
                                                                  self.policy_net.log_vars],
                                                                 feed_dict)
        feed_dict[self.policy_net.old_log_vars_ph] = old_log_vars_np
        feed_dict[self.policy_net.old_means_ph] = old_means_np

        prev_theta = self.get_flat_weights()

        def get_pg():
            return tf.get_default_session().run(self.pg, feed_dict)

        def get_hvp(p):
            feed_dict[self.p] = p
            return tf.get_default_session().run(self.hvp, feed_dict) + self.cg_damping * p

        def get_loss(theta):
            self.assign_vars(theta)
            return tf.get_default_session().run([self.policy_net.surr, self.policy_net.kl], feed_dict)

        pg = get_pg()
        if np.allclose(pg, 0):
            print("Got Zero Gradient. Not Updating.")
            return 0
        stepdir = cg(get_vp=get_hvp, b=-pg)
        shs = 0.5 * stepdir.dot(get_hvp(stepdir))
        lm = np.sqrt(shs / self.delta)
        fullstep = stepdir / lm
        success, theta = linesearch(get_loss, prev_theta, fullstep, -pg.dot(fullstep), self.delta)
        # print("success\n") if success else print("nope\n")
        self.assign_vars(theta)
        policy_loss, kl_pen = tf.get_default_session().run([self.policy_net.surr, self.policy_net.kl_pen], feed_dict)
        self.logger.log({'Policy Loss': policy_loss, 'KL Penalty': kl_pen})
        # with self.policy_net.Graph.as_default():
        #     params = tf.trainable_variables()
        #     flat_pars = tf.concat([tf.reshape(param, [-1]) for param in params], axis=0)
        #     fp = tf.get_default_session().run(flat_pars)
        #     print(fp == theta)


def linesearch(f, x, fullstep, expected_improve_rate, delta, max_backtracks=10, accept_ratio=.1):
    """
    Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
    """
    fval = f(x)[0]
    for stepfrac in (.5 ** np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval, newkl = f(xnew)
        if newkl > delta:
            newfval += np.inf
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            print(stepfrac)
            return True, xnew
    return False, x


def cg(get_vp, b, cg_iters=10, residual_tol=1e-10):
    """
    Conjugate Gradient Method, approximately solves get_vp(x) = b for x
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = get_vp(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x

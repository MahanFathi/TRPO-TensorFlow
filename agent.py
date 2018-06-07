import numpy as np
import tensorflow as tf
from datetime import datetime
from net import PolicyNet, ValueNet
from trpo import TrpoUpdater
from ppo import PPOUpdater
from env import Environment
from utils import scaler, Logger
import scipy.signal


class Agent(object):
    """
    This class is responsible for sampling from environment and making train sets
    """
    def __init__(self, config, env, policy_net):
        # Set up the logger
        now = datetime.utcnow().strftime("%b-%d_%H:%M:%S")  # create unique directories
        self.logger = Logger(logname=config.env_name, now=now)
        # Initialize
        self.env = env
        self.policy_net = policy_net
        self.baseline_net = ValueNet(config, env, self.logger)
        self.scaler = scaler(env.ob_dim + 1)
        self.update_baseline = self.baseline_net.fit
        # Settings from config
        self.gamma = config.gamma
        self.lam = config.lam
        self.animate = config.animate
        self.timesteps_per_batch = config.timesteps_per_batch
        self.timestep_limit = config.timestep_limit

    def _run_episode(self, animate=None):
        """ Run single episode with option to animate
        Returns: 4-tuple of NumPy arrays
            observes: shape = (episode len, obs_dim)
            actions: shape = (episode len, act_dim)
            rewards: shape = (episode len,)
            unscaled_obs: useful for training scalar, shape = (episode len, obs_dim)
        """
        obs = self.env.reset()
        observes, actions, rewards, unscaled_obs = [], [], [], []
        done = False
        step = 0.0
        scale, offset = self.scaler.get()
        scale[-1] = 1.0  # don't scale time step feature
        offset[-1] = 0.0  # don't offset time step feature
        for _ in range(self.timestep_limit):
            obs = obs.astype(np.float32).reshape((1, -1))
            obs = np.append(obs, [[step]], axis=1)  # add time step feature
            unscaled_obs.append(obs)
            obs = (obs - offset) * scale  # center and scale observations
            observes.append(obs)
            action = self.policy_net.sample(obs).reshape((1, -1)).astype(np.float32)
            actions.append(action)
            obs, reward, done, _ = self.env.step(np.squeeze(action, axis=0), animate)
            if not isinstance(reward, float):
                reward = np.asscalar(reward)
            rewards.append(reward)
            step += 1e-2  # increment time step feature
            if done:
                break

        return (np.concatenate(observes), np.concatenate(actions),
                np.array(rewards, dtype=np.float32), np.concatenate(unscaled_obs), done)

    def _run_policy(self, batch_size=None, animate=None):
        """ Run policy and collect data for a minimum of min_steps and min_episodes

        Returns: list of trajectory dictionaries, list length = number of episodes
            'observes' : NumPy array of states from episode
            'actions' : NumPy array of actions from episode
            'rewards' : NumPy array of (un-discounted) rewards from episode
            'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
        """
        total_steps = 0
        trajectories = []
        if batch_size is None:
            batch_size = self.timesteps_per_batch
        while total_steps < batch_size:
            observes, actions, rewards, unscaled_obs, done = self._run_episode(animate)
            total_steps += observes.shape[0]
            trajectory = {'observes': observes,
                          'actions': actions,
                          'rewards': rewards,
                          'unscaled_obs': unscaled_obs,
                          'terminated': done}
            trajectories.append(trajectory)
        unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
        self.scaler.update(unscaled)  # update running statistics for scaling observations
        self.logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                    'Steps': total_steps})

        return trajectories

    def _discount(self, x, gamma):
        """ Calculate discounted forward sum of a sequence at each point """
        return scipy.signal.lfilter([1.0], [1.0, gamma], x[::-1])[::-1]

    def _add_disc_sum_rew(self, trajectories):
        """ Adds discounted sum of rewards to all time steps of all trajectories

        Args:
            trajectories: as returned by run_policy()

        Returns:
            None (mutates trajectories dictionary to add 'disc_sum_rew')
        """
        for trajectory in trajectories:
            # if self.gamma < 0.999:  # don't scale for gamma ~= 1
            #     rewards = trajectory['rewards'] * (1 - self.gamma)
            # else:
            #     rewards = trajectory['rewards']
            rewards = trajectory['rewards']
            disc_sum_rew = self._discount(rewards, self.gamma)
            trajectory['disc_sum_rew'] = disc_sum_rew
            # print(rewards)

    def _add_value(self, trajectories):
        """ Adds estimated value to all time steps of all trajectories

        Args:
            trajectories: as returned by run_policy()

        Returns:
            None (mutates trajectories dictionary to add 'values')
        """
        for trajectory in trajectories:
            observes = trajectory['observes']
            values = self.baseline_net.predict(observes)
            trajectory['values'] = values

    def _add_gae(self, trajectories):
        """ Add generalized advantage estimator.
        https://arxiv.org/pdf/1506.02438.pdf

        Args:
            trajectories: as returned by run_policy(), must include 'values'
                key from add_value().
            gamma: reward discount
            lam: lambda (see paper).
                lam=0 : use TD residuals
                lam=1 : A =  Sum Discounted Rewards - V_hat(s)

        Returns:
            None (mutates trajectories dictionary to add 'advantages')
        """
        for trajectory in trajectories:
            # if self.gamma < 0.999:  # don't scale for gamma ~= 1
            #     rewards = trajectory['rewards'] * (1 - self.gamma)
            # else:
            #     rewards = trajectory['rewards']
            rewards = trajectory['rewards']
            values = trajectory['values']
            baseline = np.append(values, 0 if trajectory['terminated'] else values[-1])
            # temporal differences
            tds = rewards + self.gamma * baseline[1:] - baseline[:-1]
            advantages = self._discount(tds, self.gamma * self.lam)
            trajectory['advantages'] = advantages

    def build_train_set(self, animate=None):
        """
        Returns: 4-tuple of NumPy arrays
            observes: shape = (N, obs_dim)
            actions: shape = (N, act_dim)
            advantages: shape = (N,)
            disc_sum_rew: shape = (N,)
        """
        trajectories = self._run_policy(animate=animate)
        self._add_value(trajectories)
        self._add_disc_sum_rew(trajectories)
        self._add_gae(trajectories)
        observes = np.concatenate([t['observes'] for t in trajectories])
        actions = np.concatenate([t['actions'] for t in trajectories])
        disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
        advantages = np.concatenate([t['advantages'] for t in trajectories])
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        return observes, actions, advantages, disc_sum_rew


class TrpoAgent(Agent):
    def __init__(self, config):
        env = Environment(config)
        self.sess = tf.Session()
        self.sess.__enter__()
        policy_net = PolicyNet(config, env)
        super().__init__(config, env, policy_net)
        self.update_policy = TrpoUpdater(policy_net, config, self.logger)
        self.sess.run(tf.global_variables_initializer())
        # warm up agent.scalar
        self._run_policy(batch_size=1000)


class PPOAgent(Agent):
    def __init__(self, config):
        self.sess = tf.Session()
        self.sess.__enter__()
        env = Environment(config)
        policy_net = PolicyNet(config, env)
        super().__init__(config, env, policy_net)
        self.update_policy = PPOUpdater(policy_net, config, self.logger)
        self.sess.run(tf.global_variables_initializer())
        # warm up agent.scalar
        self._run_policy(batch_size=1000)

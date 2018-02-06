import gym


class Environment(object):
    def __init__(self, config):
        self.animate = config.animate
        self.env = gym.make(config.env_name)
        # Make sure it is not a discrete env
        assert not isinstance(self.env.action_space, gym.spaces.Discrete)
        self.ob_dim = self.env.observation_space.shape[0]
        self.ac_dim = self.env.action_space.shape[0]

    def reset(self):
        s = self.env.reset()
        return s

    def step(self, action, animate=None):
        if animate is None:
            animate = self.animate
        s2, r, terminal, details = self.env.step(action)
        if animate:
            self.env.render()
        return s2, r, terminal, details

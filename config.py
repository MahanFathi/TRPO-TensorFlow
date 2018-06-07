from agent import TrpoAgent, PPOAgent


class config(object):

    _vf_update_methods = ['LBFGS', 'GD']

    policy_net_size = [64, 64]
    baseline_net_size = [128, 64, 32]
    init_log_var = -1   # e^-1

    delta = 1e-2    # KL divergence between old and new policy (averaged over state-space)
    cg_damping = 1e-1

    env_name = 'Hopper-v1'
    animate = False

    timestep_limit = 10000
    timesteps_per_batch = 10000
    gamma = 0.99
    lam = 0.97

    vf_update_method = _vf_update_methods[1]
    max_lbfgs_iter = 100
    reg = 5e-3
    mixfrac = 1.0   # 1.0 corresponds to overfitting

    # PPO Updater Setup #
    init_beta = 1.0
    init_eta = 50
    lr_multiplier = 1.0
    kl_targ = 0.003
    ppo_lr = None
    update_epochs = 20

    agent = TrpoAgent
    iterations = 1000

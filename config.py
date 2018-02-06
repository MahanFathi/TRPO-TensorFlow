from agent import TrpoAgent


class config(object):

    _vf_update_methods = ['LBFGS', 'GD']

    policy_net_size = [128, 64, 32]
    baseline_net_size = [128, 64, 32]
    init_log_var = -1   # e^-1

    delta = 1e-2    # KL divergence between old and new policy (averaged over state-space)
    cg_damping = 1e-1

    env_name = 'Hopper-v1'
    animate = False

    timestep_limit = 1000
    timesteps_per_batch = 10000
    gamma = 0.99
    lam = 0.97

    vf_update_method = _vf_update_methods[0]
    max_lbfgs_iter = 100
    reg = 5e-3
    mixfrac = 1.0   # 1.0 corresponds to overfitting

    agent = TrpoAgent
    iterations = 1000

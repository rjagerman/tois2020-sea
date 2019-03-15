from experiments.classification.policies.util import PolicyConfig as _PolicyConfig, _policy_from_config


_POLICIES = {}

_POLICY_MAP = {
    'boltzmann': 0,
    'epsgreedy': 1,
    'greedy': 2,
    'thompson': 3,
    'ucb': 4,
    'uniform': 5,
    'ips': 6
}


def create_policy(d, k, strategy, lr=0.1, l2=1.0, eps=0.1, tau=0.1, alpha=1.0,
                  cap=1e-6, baseline=None):
    config = _PolicyConfig(
        d,
        k,
        _POLICY_MAP[strategy],
        lr,
        l2,
        eps,
        tau,
        alpha,
        cap
    )
    return _policy_from_config(config, baseline=baseline)


def policy_from_model(model, baseline=None):
    return _policy_from_config(model.config, baseline=baseline)

from experiments.classification.policies.boltzmann import boltzmann_policy
from experiments.classification.policies.epsgreedy import epsgreedy_policy
from experiments.classification.policies.greedy import greedy_policy
from experiments.classification.policies.uniform import uniform_policy
from experiments.classification.policies.ips import ips_policy


_STRATEGY_MAP = {
    'boltzmann': lambda k, d, args: boltzmann_policy(k, d, args['lr'], args['tau'], args['w']),
    'epsgreedy': lambda k, d, args: epsgreedy_policy(k, d, args['lr'], args['eps'], args['w']),
    'greedy': lambda k, d, args: greedy_policy(k, d, args['lr'], args['w']),
    'uniform': lambda k, d, args: uniform_policy(k, d, args['lr'], args['w']),
    'ips': lambda k, d, args: ips_policy(k, d, args['baseline'], args['lr'], args['cap'], args['w'])
}

def create_policy(strategy, k, d, **args):
    defaults = {
        'w': None,
        'lr': 0.01,
        'eps': 0.05,
        'tau': 1.0,
        'cap': 0.05
    }
    defaults.update(args)
    return _STRATEGY_MAP[strategy](k, d, defaults)

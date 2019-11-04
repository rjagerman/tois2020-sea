import numpy as _np
from experiments.classification.policies.boltzmann import BoltzmannPolicy
from experiments.classification.policies.epsgreedy import EpsgreedyPolicy
from experiments.classification.policies.greedy import GreedyPolicy
from experiments.classification.policies.uniform import UniformPolicy
from experiments.classification.policies.ips import IPSPolicy
from experiments.classification.policies.sea import SEAPolicy
from experiments.classification.policies.comp import CompPolicy
from experiments.classification.policies.statistical import StatisticalPolicy, TYPE_UCB as _TYPE_UCB, TYPE_THOMPSON as _TYPE_THOMPSON


_STRATEGY_MAP = {
    'boltzmann': lambda k, d, args: BoltzmannPolicy(k, d, args['lr'], args['l2'], args['tau'], args['w']),
    'epsgreedy': lambda k, d, args: EpsgreedyPolicy(k, d, args['lr'], args['l2'], args['eps'], args['w']),
    'greedy': lambda k, d, args: GreedyPolicy(k, d, args['lr'], args['l2'], args['w']),
    'uniform': lambda k, d, args: UniformPolicy(k, d, args['lr'], args['l2'], args['w']),
    'ips': lambda k, d, args: IPSPolicy(k, d, args['baseline'], args['lr'], args['l2'], args['cap'], args['w']),
    'ucb': lambda k, d, args: StatisticalPolicy(k, d, args['l2'], args['alpha'], args['w'], draw_type=_TYPE_UCB),
    'thompson': lambda k, d, args: StatisticalPolicy(k, d, args['l2'], args['alpha'], args['w'], draw_type=_TYPE_THOMPSON),
    'sea': lambda k, d, args: SEAPolicy(k, d, args['n'], args['baseline'], args['lr'], args['l2'], args['cap'], args['w'], args['confidence'], recompute_bounds=args['recompute_bounds']),
    'comp': lambda k, d, args: CompPolicy(k, d, args['n'], args['baseline'], args['lr'], args['l2'], args['cap'], args['w'], args['confidence'], recompute_bounds=args['recompute_bounds']),
}

def create_policy(strategy, k, d, **args):
    defaults = {
        'w': None,
        'lr': 0.01,
        'l2': 1.0,
        'eps': 0.05,
        'tau': 1.0,
        'cap': 0.05,
        'alpha': 1.0,
        'confidence': 0.95,
        'recompute_bounds': _np.array([1], dtype=_np.int32)
    }
    defaults.update(args)
    return _STRATEGY_MAP[strategy](k, d, defaults)

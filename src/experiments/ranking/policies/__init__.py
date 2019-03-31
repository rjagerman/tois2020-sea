from experiments.ranking.policies.online import OnlinePolicy
from experiments.ranking.policies.ips import IPSPolicy
from experiments.ranking.policies.sea import SEAPolicy


_STRATEGY_MAP = {
    'online': lambda d, args: OnlinePolicy(d, args['lr'], args['w']),
    'ips': lambda d, args: IPSPolicy(d, args['lr'], args['baseline'], args['eta'], args['cap'], args['w']),
    'sea': lambda d, args: SEAPolicy(d, args['pairs'], args['lr'], args['baseline'], args['eta'], args['cap'], args['w'])
}

def create_policy(strategy, d, **args):
    defaults = {
        'w': None,
        'lr': 0.01,
        'eta': 1.0,
        'cap': 0.01,
        'confidence': 0.95
    }
    defaults.update(args)
    return _STRATEGY_MAP[strategy](d, defaults)

from experiments.ranking.policies.online import OnlinePolicy
from experiments.ranking.policies.ips import IPSPolicy
from experiments.ranking.policies.sea import SEAPolicy
from experiments.ranking.policies.dueling_bandits import DuelingBanditPolicy


_STRATEGY_MAP = {
    'online': lambda d, args: OnlinePolicy(d, args['lr'], args['w']),
    'ips': lambda d, args: IPSPolicy(d, args['lr'], args['baseline'], args['eta'], args['cap'], args['w']),
    'sea': lambda d, args: SEAPolicy(d, args['pairs'], args['lr'], args['baseline'], args['eta'], args['cap'], args['w'], confidence=args['confidence']),
    'meancomp': lambda d, args: SEAPolicy(d, args['pairs'], args['lr'], args['baseline'], args['eta'], args['cap'], args['w'], confidence=0.0),
    'duelingbandit': lambda d, args: DuelingBanditPolicy(d, args['lr'], args['lr_decay'], args['delta'], args['w'])
}

def create_policy(strategy, d, **args):
    defaults = {
        'w': None,
        'lr': 0.01,
        'lr_decay': 1.0,
        'delta': 0.01,
        'eta': 1.0,
        'cap': 0.01,
        'confidence': 0.95
    }
    defaults.update(args)
    return _STRATEGY_MAP[strategy](d, defaults)

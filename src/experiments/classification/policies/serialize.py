from experiments.classification.policies.boltzmann import boltzmann_policy
from experiments.classification.policies.greedy import greedy_policy
from experiments.classification.policies.uniform import uniform_policy
from experiments.classification.policies.epsgreedy import epsgreedy_policy
from experiments.classification.policies.ips import ips_policy
from collections import namedtuple
import numba
import numpy as np


_CONSTRUCTOR_MAP = {
    'BoltzmannPolicy': boltzmann_policy,
    'GreedyPolicy': greedy_policy,
    'UniformPolicy': uniform_policy,
    'EpsgreedyPolicy': epsgreedy_policy,
    'IPSPolicy': ips_policy
}


SerializedPolicy = namedtuple('SerializedPolicy', ['name', 'struct'])


def serialize_policy(policy):
    typ = numba.typeof(policy)
    data = {}
    for k in typ.struct:
        v = getattr(policy, k)
        if isinstance(v, np.ndarray):
            v = np.copy(v)
        if str(numba.typeof(v))[0:17] == 'instance.jitclass':
            v = serialize_policy(v)
        data[k] = v
    return SerializedPolicy(typ.classname, data)


def deserialize_policy(data):
    args = {}
    for k, v in data.struct.items():
        if isinstance(v, SerializedPolicy):
            v = deserialize_policy(v)
        args[k] = v
    try:
        instance = _CONSTRUCTOR_MAP[data.name](**args)
    except KeyError:
        class_name = globals()[data.name]
        instance = class_name(**args)
    return instance

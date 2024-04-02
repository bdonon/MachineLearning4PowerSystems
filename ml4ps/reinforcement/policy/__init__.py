from ml4ps.reinforcement.policy.continous_policy import ContinuousPolicy
from ml4ps.reinforcement.policy.continous_policy_no_env import ContinuousPolicyNoEnv
from ml4ps.reinforcement.policy.one_hot_delta_discrete import OneHotDeltaDiscrete
from ml4ps.reinforcement.policy.factorized_delta_discrete import FactorizedDeltaDiscrete
from ml4ps.reinforcement.policy.continous_and_discrete import ContinuousAndDiscrete
from ml4ps.reinforcement.policy.base import BasePolicy


def get_policy(policy_type, env, **policy_kwargs) -> BasePolicy:
    if policy_type == 'continuous':
        return ContinuousPolicy(env, **policy_kwargs)
    elif policy_type == 'continuous_no_env': # No Env
        return ContinuousPolicyNoEnv(**policy_kwargs)
    elif policy_type == 'one_hot_delta_discrete':
        return OneHotDeltaDiscrete(env, **policy_kwargs)
    elif policy_type == 'factorized_delta_discrete':
        return FactorizedDeltaDiscrete(env, **policy_kwargs)
    elif policy_type == 'continuous_and_discrete':
        return ContinuousAndDiscrete(env, **policy_kwargs)
    else:
        raise ValueError('Unknown policy type: {}'.format(policy_type))
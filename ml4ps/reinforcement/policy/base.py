import os
from abc import ABC
from typing import Any, Dict, Tuple

from gymnasium.vector import VectorEnv
from jax import vmap
from ml4ps.h2mg import H2MG, H2MGNormalizer


class BasePolicy(ABC):
    normalizer: H2MGNormalizer = None

    def __init__(self) -> None:
        pass

    def init(self, rng, observation):
        pass

    def log_prob(self, params: Dict, observation: H2MG, action: H2MG,
                 env: VectorEnv = None) -> Tuple[float, Any]:
        # return log probability of actions
        pass

    def sample(self, params: dict, observation: dict, rng, deterministic: bool = False,
               n_action: int = 1, env: VectorEnv = None) -> Tuple[H2MG, float, Dict]:
        # return both sample action and corresponding log probabilities
        pass

    def jit_sample(self, params: dict, observation: dict, rng, deterministic: bool = False,
                   n_action: int = 1, env: VectorEnv = None) -> Tuple[H2MG, float, Dict]:
        # return both sample action and corresponding log probabilities
        pass

    # @partial(jit, static_argnums=(0, 4, 5))
    def vmap_sample(self, params: dict, observation: H2MG, rng, deterministic: bool = False,
                    n_action: int = 1, env: VectorEnv = None) -> Tuple[H2MG, float, Dict]:
        # return both sample action and corresponding log probabilities for a given batch
        pass

    # @partial(jit, static_argnums=(0,))
    def vmap_log_prob(self, params: Dict, obs: H2MG, action: H2MG, env: VectorEnv = None) -> float:
        return vmap(self.log_prob, in_axes=(None, 0, 0), out_axes=(0, 0))(params, obs, action)

    def _build_normalizer(self, env, normalizer_args=None):
        if isinstance(env, VectorEnv):
            backend = env.get_attr("backend")[0]
            data_dir = env.get_attr("data_dir")[0]
            observation_structure = env.get_attr("observation_structure")[0]
            env_name = env.get_attr("name")[0]
        else:
            backend = env.backend
            data_dir = env.data_dir
            observation_structure = env.observation_structure
            env_name = env.name

        normalizer_dir = os.path.join(data_dir, env_name)
        normalize_path = os.path.join(normalizer_dir, 'normalizer.pkl')
        if os.path.exists(normalize_path):
            return H2MGNormalizer(filename=normalize_path)
        if normalizer_args is None:
            normalizer = H2MGNormalizer(
                backend=backend, structure=observation_structure, data_dir=data_dir)
        else:
            normalizer = H2MGNormalizer(
                backend=backend, structure=observation_structure, data_dir=data_dir,
                **normalizer_args)

        if not os.path.exists(normalize_path):
            if not os.path.exists(normalizer_dir):
                os.mkdir(normalizer_dir)
            normalizer.save(normalize_path)
        return normalizer

    def _build_normalizer_no_env(self, backend, data_dir, observation_structure, env_name, normalizer_args=None):
        normalizer_dir = os.path.join(data_dir, env_name)
        normalize_path = os.path.join(normalizer_dir, 'normalizer.pkl')
        if os.path.exists(normalize_path):
            return H2MGNormalizer(filename=normalize_path)
        if normalizer_args is None:
            normalizer = H2MGNormalizer(
                backend=backend, structure=observation_structure, data_dir=data_dir)
        else:
            normalizer = H2MGNormalizer(
                backend=backend, structure=observation_structure, data_dir=data_dir,
                **normalizer_args)

        if not os.path.exists(normalize_path):
            if not os.path.exists(normalizer_dir):
                os.mkdir(normalizer_dir)
            normalizer.save(normalize_path)
        return normalizer
import pickle
from copy import deepcopy
from functools import partial
from numbers import Number
from typing import Dict

import gymnasium
import jax
import jax.numpy as jnp
from gymnasium import spaces
from jax import jit, vmap

import ml4ps
from ml4ps.h2mg import (
    H2MG,
    H2MGSpace,
    h2mg_normal_logprob,
    h2mg_normal_sample,
    shallow_repr,
)
from ml4ps.reinforcement.policy.base import BasePolicy


class ContinuousPolicyNoEnv(BasePolicy):
    """Continuous policy for power system control.

    Attributes
        observation_space spaces.Space
        action_space spaces.Space
        normalizer ml4ps.Normalizer preprocess input
        postprocessor postprocess output parameters
        nn produce ditribution parameters from observation input.
    """

    def __init__(
        self,
        action_space=None,
        backend=None,
        data_dir=None,
        observation_structure=None,
        env_name=None,
        normalizer=None,
        normalizer_args=None,
        nn_type="h2mgnode",
        box_to_sigma_ratio=8,
        file=None,
        cst_sigma=None,
        clip_sigma=None,
        nn_args={},
    ) -> None:
        if file is not None:
            self.load(file)
        else:
            self.box_to_sigma_ratio = box_to_sigma_ratio
            self.action_space = action_space.continuous

            if normalizer is None:
                self.normalizer = self._build_normalizer_no_env(
                    backend,
                    data_dir,
                    observation_structure,
                    env_name,
                    normalizer_args=normalizer_args,
                )
            else:
                self.normalizer = normalizer

            self.mu_structure = self.action_space.add_suffix("_mu").structure
            self.log_sigma_structure = self.action_space.add_suffix(
                "_log_sigma"
            ).structure

            nn_output_structure = self.mu_structure.combine(self.log_sigma_structure)
            self.nn = ml4ps.neural_network.get(
                nn_type, output_structure=nn_output_structure, **nn_args
            )
            self.mu_0, self.log_sigma_0 = self._build_postprocessor(self.action_space)

        self.cst_sigma = cst_sigma
        self.clip_sigma = clip_sigma

    def _build_postprocessor(self, h2mg_space: H2MGSpace):
        high = h2mg_space.high
        low = h2mg_space.low
        mu_0 = H2MG.from_structure(self.mu_structure)
        mu_0.flat_array = (high.flat_array + low.flat_array) / 2.0
        log_sigma_0 = H2MG.from_structure(self.mu_structure)
        log_sigma_0.flat_array = jnp.log(
            (high.flat_array - low.flat_array) / self.box_to_sigma_ratio
        )
        return mu_0, log_sigma_0

    def _clip_log_sigma(self, log_sigma, eps=0.01):
        # soft clipping
        return (
            jax.numpy.clip((1 - eps) * log_sigma, a_min=jnp.log(self.clip_sigma))
            + eps * log_sigma
        )

    def _postprocess_distrib_params(
        self, distrib_params: H2MG, action_space=None, cst_sigma=None
    ):
        if action_space is not None:
            action_space = action_space
        else:
            action_space = self.action_space
        mu_0, log_sigma_0 = self._build_postprocessor(
            action_space
        )  # action_space vs self.action_space ?
        mu = distrib_params.extract_from_structure(self.mu_structure)
        log_sigma = distrib_params.extract_from_structure(self.log_sigma_structure)
        mu_norm = H2MG.from_structure(
            self.action_space.structure
        )  # action_space vs self.action_space ?
        mu_norm.flat_array = (
            jnp.exp(log_sigma_0.flat_array) * mu.flat_array + mu_0.flat_array
        )
        log_sigma_norm = H2MG.from_structure(
            self.action_space.structure
        )  # action_space vs self.action_space?
        log_sigma_norm.flat_array = log_sigma.flat_array + log_sigma_0.flat_array
        if self.clip_sigma is not None and isinstance(self.clip_sigma, Number):
            log_sigma_norm.flat_array = self._clip_log_sigma(log_sigma_norm.flat_array)
        if self.cst_sigma is not None and isinstance(self.cst_sigma, Number):
            log_sigma_norm.flat_array = jnp.full_like(
                log_sigma.flat_array, jnp.log(cst_sigma)
            )  # constant sigma
        return mu_norm, log_sigma_norm

    def init(self, rng, obs):
        return self.nn.init(rng, obs)

    def _check_valid_action(self, action):
        # TODO
        pass

    def _log_prob(self, params, observation, action, action_space=None, cst_sigma=None):
        mu_norm, log_sigma_norm = self._forward(
            params, observation, action_space=action_space, cst_sigma=cst_sigma
        )
        return h2mg_normal_logprob(action, mu_norm, log_sigma_norm), (
            mu_norm,
            log_sigma_norm,
        )

    def log_prob(self, params, observation, action, action_space=None):
        return self._log_prob(
            params, observation, action, action_space, cst_sigma=self.cst_sigma
        )

    def vmap_log_prob(
        self, params: Dict, obs: H2MG, action: H2MG, action_space=None
    ) -> float:
        return jit(
            vmap(self._log_prob, in_axes=(None, 0, 0, None, None), out_axes=(0, 0)),
            static_argnums=(3,),
        )(params, obs, action, action_space, self.cst_sigma)

    def _forward(self, params, observation, action_space=None, cst_sigma=None):
        observation = self.normalizer(observation)
        distrib_params = self.nn.apply(params, observation)
        mu, log_sigma = self._postprocess_distrib_params(
            distrib_params, action_space=action_space, cst_sigma=cst_sigma
        )
        return mu, log_sigma

    def _sample(
        self,
        params,
        observation: H2MG,
        rng,
        deterministic=False,
        n_action=1,
        action_space=None,
        cst_sigma=None,
    ):
        """Sample an action and return it together with the corresponding log probability."""
        mu_norm, log_sigma_norm = self._forward(
            params, observation, action_space=action_space, cst_sigma=cst_sigma
        )
        if n_action <= 1:
            action = h2mg_normal_sample(
                rng, mu_norm, log_sigma_norm, deterministic=deterministic
            )
            log_prob = h2mg_normal_logprob(action, mu_norm, log_sigma_norm)
        else:
            action = [
                h2mg_normal_sample(
                    _rng, mu_norm, log_sigma_norm, deterministic=deterministic
                )
                for _rng in jax.random.split(rng, n_action)
            ]
            log_prob = [h2mg_normal_logprob(a, mu_norm, log_sigma_norm) for a in action]
        info = self.compute_info(mu_norm, log_sigma_norm)
        return action, log_prob, info, mu_norm, log_sigma_norm

    def sample(
        self,
        params,
        observation: H2MG,
        rng,
        deterministic=False,
        n_action=1,
        action_space=None,
    ):
        """Sample an action and return it together with the corresponding log probability."""
        action, log_prob, info, _, _ = self._sample(
            params,
            observation,
            rng,
            deterministic,
            n_action,
            action_space=action_space,
            cst_sigma=self.cst_sigma,
        )
        return action, log_prob, info

    def jit_sample(
        self,
        params,
        observation: H2MG,
        rng,
        deterministic=False,
        n_action=1,
        action_space=None,
    ):
        """Sample an action and return it together with the corresponding log probability."""
        action, log_prob, info, _, _ = jit(self._sample, static_argnums=(3, 4, 5))(
            params,
            observation,
            rng,
            deterministic,
            n_action,
            action_space,
            self.cst_sigma,
        )
        return action, log_prob, info

    @partial(jit, static_argnums=(0, 4, 5, 6))
    def _vmap_sample(
        self,
        params,
        observation: spaces.Space,
        rng,
        deterministic=False,
        n_action=1,
        action_space=None,
        cst_sigma=None,
    ):
        action, log_prob, _, mu, log_sigma = vmap(
            self._sample,
            in_axes=(None, 0, 0, None, None, None, None),
            out_axes=(0, 0, 0, 0, 0),
        )(params, observation, rng, deterministic, n_action, action_space, cst_sigma)
        info = self.compute_info(mu, log_sigma)
        batch_info = self.compute_batch_info(mu, log_sigma)
        return action, log_prob, info | batch_info

    # @partial(jit, static_argnums=(6,))
    def vmap_sample(
        self,
        params,
        observation: spaces.Space,
        rng,
        deterministic=False,
        n_action=1,
        action_space=None,
    ):
        return self._vmap_sample(
            params,
            observation,
            rng,
            deterministic,
            n_action,
            action_space,
            self.cst_sigma,
        )

    @staticmethod
    def compute_info(mu: H2MG, log_sigma: H2MG) -> Dict[str, float]:
        mu = deepcopy(mu)
        max_mu = deepcopy(mu)
        min_mu = deepcopy(mu)
        log_sigma = deepcopy(log_sigma)
        mu.add_suffix("_mu")
        max_mu.add_suffix("_max_mu")
        min_mu.add_suffix("_min_mu")
        log_sigma.add_suffix("_log_sigma")
        info = shallow_repr(mu.apply(lambda x: jnp.asarray(jnp.nanmean(x))))
        info = info | shallow_repr(
            log_sigma.apply(lambda x: jnp.asarray(jnp.nanmean(x)))
        )
        info = info | shallow_repr(max_mu.apply(lambda x: jnp.asarray(jnp.nanmax(x))))
        info = info | shallow_repr(min_mu.apply(lambda x: jnp.asarray(jnp.nanmin(x))))
        return info

    @staticmethod
    def compute_batch_info(mu: H2MG, log_sigma: H2MG) -> Dict[str, float]:
        _mu = deepcopy(mu)
        mu.add_suffix("_mu_batch_std")
        mu_std_accross_batch = shallow_repr(
            mu.apply(lambda x: jnp.asarray(jnp.nanstd(x, axis=0).mean()))
        )
        _mu.add_suffix("_mu_gen_std")
        mu_std_accross_gen = shallow_repr(
            _mu.apply(lambda x: jnp.asarray(jnp.nanstd(x, axis=1).mean()))
        )
        log_sigma.add_suffix("_log_sigma_std")
        log_sigma_std_accross_batch = shallow_repr(
            log_sigma.apply(lambda x: jnp.asarray(jnp.nanstd(x, axis=0).mean()))
        )
        return mu_std_accross_batch | log_sigma_std_accross_batch | mu_std_accross_gen

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.box_to_sigma_ratio, f)
            pickle.dump(self.action_space, f)
            pickle.dump(self.normalizer, f)
            pickle.dump(self.mu_structure, f)
            pickle.dump(self.log_sigma_structure, f)
            pickle.dump(self.nn, f)
            pickle.dump(self.mu_0, f)
            pickle.dump(self.log_sigma_0, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.box_to_sigma_ratio = pickle.load(f)
            self.action_space = pickle.load(f)
            self.normalizer = pickle.load(f)
            self.mu_structure = pickle.load(f)
            self.log_sigma_structure = pickle.load(f)
            self.nn = pickle.load(f)
            self.mu_0 = pickle.load(f)
            self.log_sigma_0 = pickle.load(f)

    def entropy(self, log_prob_info: Dict, batch=True):
        (mu_norm, log_sigma_norm) = log_prob_info
        return jnp.zeros_like(log_sigma_norm.flat_array)

    def log_sigma_sum(self, h2gm_node):
        return jnp.sum(h2gm_node.flat_array)

    def entropy(self, log_prob_info: Dict, batch=True, matrix=False, axis=0):
        (mu_norm, log_sigma_norm) = log_prob_info
        if matrix:
            # (na, bs, n_obj)
            return vmap(
                vmap(self.log_sigma_sum, in_axes=0, out_axes=0), in_axes=0, out_axes=0
            )(log_sigma_norm).mean()
        if batch:
            # (bs, n_obj)
            return vmap(self.log_sigma_sum, in_axes=0, out_axes=0)(
                log_sigma_norm
            ).mean()
        else:
            raise ValueError

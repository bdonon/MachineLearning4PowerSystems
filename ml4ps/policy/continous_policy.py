from collections import defaultdict
from typing import Any, Callable, Dict, Tuple

import jax
import numpy as np
from gymnasium import spaces
from ml4ps import H2MGNODE, h2mg, Normalizer
from ml4ps.policy.base import BasePolicy
import gymnasium


def add_prefix(x, prefix):
    return transform_feature_names(x, lambda feat_name: prefix+feat_name)


def remove_prefix(x, prefix):
    return transform_feature_names(x, lambda feat_name: feat_name.removeprefix(prefix))


def tr_feat(feat_names, fn):
    if isinstance(feat_names, list):
        return list(map(fn, feat_names))
    elif isinstance(feat_names, dict):
        return {fn(feat): value for feat, value in feat_names.items()}


def transform_feature_names(_x, fn: Callable):
    x = _x.copy()
    if "local_features" in x:
        x |= {"local_features": {obj_name: tr_feat(
            obj, fn) for obj_name, obj in x["local_features"].items()}}
    if "global_features" in x:
        x |= {"global_features": tr_feat(x["global_features"], fn)}
    return x


def slice_with_prefix(_x, prefix):
    x = _x.copy()
    if "local_features" in x:
        x |= {"local_features": {obj_name: {feat.removeprefix(prefix): value for feat, value in obj.items(
        ) if feat.startswith(prefix)} for obj_name, obj in x["local_features"].items()}}
    if "global_features" in x:
        x |= {"global_features": {feat.removeprefix(
            prefix): value for feat, value in x["global_features"] if feat.startswith(prefix)}}
    return x


def combine_space(a, b):
    x = h2mg.empty_h2mg()
    for local_key, obj_name, feat_name, value in h2mg.local_features_iterator(a):
        x[local_key][obj_name][feat_name] = value
    for local_key, obj_name, feat_name, value in h2mg.local_features_iterator(b):
        x[local_key][obj_name][feat_name] = value

    for global_key,  feat_name, value in h2mg.global_features_iterator(a):
        x[global_key][feat_name] = value
    for global_key,  feat_name, value in h2mg.global_features_iterator(b):
        x[global_key][feat_name] = value

    for local_key, obj_name, addr_name, value in h2mg.local_addresses_iterator(a):
        x[local_key][obj_name][addr_name] = value

    for all_addr_key, value in h2mg.all_addresses_iterator(a):
        x[all_addr_key][value] = value
    return x


def combine_feature_names(feat_a, feat_b):
    new_feat_a = defaultdict(lambda: defaultdict(list))
    for local_key, obj_name, feat_name in h2mg.local_feature_names_iterator(feat_a):
        new_feat_a[local_key][obj_name].append(feat_name)
    for local_key, obj_name, feat_name in h2mg.local_feature_names_iterator(feat_b):
        new_feat_a[local_key][obj_name].append(feat_name)

    for local_key, obj_name, feat_name in h2mg.local_feature_names_iterator(feat_b):
        new_feat_a[local_key][obj_name] = list(
            set(new_feat_a[local_key][obj_name]))

    new_feat_a[h2mg.H2MGCategories.GLOBAL_FEATURES.value] = list()
    for global_key, feat_name in h2mg.global_feature_names_iterator(feat_a):
        new_feat_a[global_key].append(feat_name)
    for global_key, feat_name in h2mg.global_feature_names_iterator(feat_b):
        new_feat_a[global_key].append(feat_name)
    new_feat_a[h2mg.H2MGCategories.GLOBAL_FEATURES.value] = list(
        set(new_feat_a[h2mg.H2MGCategories.GLOBAL_FEATURES.value]))
    return new_feat_a


def space_to_feature_names(space: spaces.Space):
    feat_names = {}
    if "local_addresses" in list(space.keys()):
        feat_names |= {"local_addresses": {
            k: list(v) for k, v in space["local_addresses"].items()}}
    if "local_features" in list(space.keys()):
        feat_names |= {"local_features": {
            k: list(v) for k, v in space["local_features"].items()}}
    if "global_features" in list(space.keys()):
        feat_names |= {"global_features": {
            list(k) for k, _ in space["global_features"].items()}}
    return feat_names

# TODO handle nan in observation ?


class ContinuousPolicy(BasePolicy):
    """ Continuous policy for power system control.

        Attributes
            observation_space spaces.Space
            action_space spaces.Space
            normalizer ml4ps.Normalizer preprocess input
            postprocessor postprocess output parameters
            nn produce ditribution parameters from observation input.
    """

    def __init__(self, env=None, normalizer=None, normalizer_args=None, nn_type=None, **nn_args) -> None:
        self.mu_prefix = "mu_"
        self.log_sigma_prefix = "log_sigma_"
        self.action_space, self.observation_space = env.action_space, env.observation_space
        self.normalizer = normalizer or self.build_normalizer(env, normalizer_args)
        self.nn_args = nn_args
        self.build_out_features_names_struct(env.action_space)
        self.postprocessor = self.build_postprocessor(env.action_space)
        self.nn = self.build_nn(nn_type, self.out_feature_struct, **nn_args)

    @property
    def normalizer(self):
        return self._normalizer

    @normalizer.setter
    def normalizer(self, value):
        self._normalizer = value

    @property
    def nn_args(self):
        return self._nn_args

    @nn_args.setter
    def nn_args(self, value):
        self._nn_args = value

    def init(self, rng, obs):
        self.params = self.nn.init(rng, obs)
        return self.params

    def build_out_features_names_struct(self, space: spaces.Space) -> Dict:
        """Builds output feature names structure from power system space.
            The output feature correspond to the parameter of the continuous distribution.
        """

        feat_names = space_to_feature_names(self.action_space)
        self.sigma_feat_names = add_prefix(
            feat_names, self.log_sigma_prefix)  # save this
        self.mu_feature_names = add_prefix(feat_names, self.mu_prefix)
        self.out_feature_struct = combine_feature_names(
            self.sigma_feat_names, self.mu_feature_names)
        return

    def build_nn(self, nn_type: str, out_feature_struct: Dict, **kwargs):
        return H2MGNODE.make(output_feature_names=out_feature_struct, local_dynamics_hidden_size=[16],
                             global_dynamics_hidden_size=[16],
                             local_decoder_hidden_size=[16],
                             global_decoder_hidden_size=[16],
                             local_latent_dimension=4,
                             global_latent_dimension=4,
                             solver_name="Euler",
                             dt0=0.005,
                             stepsize_controller_name="ConstantStepSize",
                             stepsize_controller_kwargs={},
                             adjoint_name="RecursiveCheckpointAdjoint",
                             max_steps=4096)

    def build_postprocessor(self, action_space: spaces.Space):
        """Builds postprocessor that transform nn output into the proper range
            via affine transformation.
        """
        # TODO: add global features
        post_process_h2mg = h2mg.empty_h2mg()
        self.mu_0 = h2mg.empty_h2mg()
        self.log_sigma_0 = h2mg.empty_h2mg()
        for local_key, obj_name, feat_name in h2mg.local_feature_names_iterator(space_to_feature_names(self.action_space)):
            high = self.action_space[local_key][obj_name][feat_name].high
            low = self.action_space[local_key][obj_name][feat_name].low
            mu_0_value = np.mean(low + (high-low)/2)  # TODO add checks
            self.mu_0[local_key][obj_name][self.mu_prefix +
                                           feat_name] = mu_0_value
            sigma_0_value = (high-low)/8
            log_sigma_0_value = np.mean(
                np.log(sigma_0_value))  # TODO add checks
            self.log_sigma_0[local_key][obj_name][self.log_sigma_prefix +
                                                  feat_name] = log_sigma_0_value
            post_process_h2mg[local_key][obj_name][self.log_sigma_prefix +
                                                   feat_name] = lambda x: x+log_sigma_0_value
            post_process_h2mg[local_key][obj_name][self.mu_prefix +
                                                   feat_name] = lambda x: x+mu_0_value

        class PostProcessor:
            def __init__(self, post_process_h2mg) -> None:
                self.post_process_h2mg = post_process_h2mg

            def __call__(self, x):
                return h2mg.map_to_features(lambda target, fn: fn(target), x, self.post_process_h2mg)
        return PostProcessor(post_process_h2mg)

    def _check_valid_action(self, action):
        # TODO
        pass

    def log_prob(self, params, observation, action):
        observation = self.normalizer(observation)
        distrib_params = self.nn.apply(params, observation)
        distrib_params = self.post_process_params(distrib_params)
        return self.normal_log_prob(action, distrib_params)

    def normal_log_prob(self, action: Dict, distrib_params: Dict) -> float:
        # use onlymu and sigma and not mu_0 and sigma_0
        """Return the log probability of an action"""
        self._check_valid_action(action)
        log_probs = 0
        # log_probs = h2mg.map_to_features(self.feature_log_prob, action, mu, sigma, mu_0, sigma_0)
        if "local_features" in action:
            for k in action["local_features"]:
                for f in action["local_features"][k]:
                    action_kf = action["local_features"][k][f]
                    mu_kf = distrib_params["local_features"][k][self.mu_prefix+f]
                    log_sigma_kf = distrib_params["local_features"][k][self.log_sigma_prefix+f]
                    mu0_kf = self.mu_0["local_features"][k][f]
                    log_sigma0_kf = self.log_sigma_0["local_features"][k][f]
                    log_probs += self.feature_log_prob(action_kf, mu_kf, log_sigma_kf)
        if "global_features" in action:
            for k in action["global_features"]:
                action_kf = action["global_features"][k]
                mu_kf = distrib_params["global_features"][self.mu_prefix+k]
                log_sigma_kf = distrib_params["global_features"][self.log_sigma_prefix+k]
                mu0_kf = self.mu_0["global_features"][k]
                log_sigma0_kf = self.log_sigma_0["global_features"][k]
                log_probs += self.feature_log_prob(action_kf,mu_kf, log_sigma_kf)

        return log_probs

    def feature_log_prob(self, action, mu, log_sigma):
        return np.nansum(- log_sigma - 0.5 * np.exp(-2 * log_sigma) * (action - mu)**2)

    def sample(self, params, observation: spaces.Space, seed=0, deterministic=False, n_action=1) -> Tuple[spaces.Space, float]:
        # n_action = 1, no list, n_action > 1 list
        """Sample an action and return it together with the corresponding log probability."""
        observation = self.normalizer(observation)
        distrib_params = self.nn.apply(params, observation)
        distrib_params = self.post_process_params(distrib_params)
        action = self.sample_from_params(
            distrib_params, np.random.default_rng(seed))  # TODO rng
        info = {"info": 0}
        return action, self.normal_log_prob(action, distrib_params), info

    def get_params(self, out_dict):
        mu = slice_with_prefix(out_dict, self.mu_prefix)
        sigma = slice_with_prefix(out_dict, self.log_sigma_prefix)
        return mu, sigma

    def sample_from_params(self, action_params: Dict, np_random) -> Dict:
        """Sample an action from the parameter of the continuous distribution."""
        mu, sigma = self.get_params(action_params)
        mu_flat, sigma_flat = spaces.flatten(
            self.action_space, mu), spaces.flatten(self.action_space, sigma)
        action_flat = mu_flat + \
            np_random.normal(size=sigma_flat.shape) * sigma_flat
        action = spaces.unflatten(self.action_space, action_flat)
        return action

    def post_process_params(self, action_params: Dict) -> Dict:
        """Apply post processing to the nn output."""
        return self.postprocessor(action_params)

    def build_normalizer(self, env, normalizer_args=None, data_dir=None):
        if isinstance(env, gymnasium.vector.VectorEnv):
            backend = env.get_attr("backend")[0]
            data_dir = env.get_attr("data_dir")[0]
        else:
            backend = env.backend
            data_dir = env.data_dir

        if normalizer_args is None:
            return Normalizer(backend=backend, data_dir=data_dir)
        else:
            return Normalizer(backend=backend, data_dir=data_dir, **normalizer_args)
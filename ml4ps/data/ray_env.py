import warnings
from numbers import Number
from typing import Any, Callable, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import ray
from gymnasium.vector.utils import concatenate, iterate
from jax.random import split

from ml4ps import H2MG, H2MGStructure
from ml4ps.backend import AbstractBackend
from ml4ps.h2mg import H2MGSpace


def apply_action(
    power_grid: Any, action: H2MG, *, backend: AbstractBackend, sim_args: Dict
) -> None | Any:
    backend.set_h2mg_into_power_grid(power_grid, action)
    return backend.run_power_grid(power_grid, **sim_args)


def default_pipeline(
    power_grid: Any,
    action: H2MG,
    *,
    backend: AbstractBackend,
    metric_fn: Callable[[H2MG], float],
    sim_args,
    save_folder=None,
    **run_kwargs,
) -> None | Any:
    apply_action(power_grid, action, backend=backend, sim_args=sim_args)
    if save_folder is not None:
        backend.save_power_grid(power_grid, path=save_folder)
    return metric_fn(power_grid, backend=backend, **run_kwargs)


def sample_power_grids(
    backend: AbstractBackend,
    data_dir: str,
    size: int,
    deterministic: bool = False,
    rng_key: jax.numpy.ndarray = None,
    valid_files: List = None,
    np_random: np.random.Generator = None,
) -> List[Any]:
    if valid_files is None:
        valid_files = backend.get_valid_files(data_dir)
    if deterministic:
        files = valid_files[:size]
    else:
        if rng_key is not None:
            idx = jax.random.choice(rng_key, len(valid_files), shape=(size,))
            files = [valid_files[_idx] for _idx in idx]
        else:
            if np_random is not None:
                files = np_random.choice(valid_files, size=size)
            else:
                files = np.random.choice(valid_files, size=size)
    power_grids = [backend.load_power_grid(file) for file in files]
    return power_grids


def sample_power_grids_and_h2mgs_jax(
    backend: AbstractBackend,
    data_dir: str,
    size: int,
    observation_structure: H2MGStructure,
    deterministic=False,
    rng_key=None,
    valid_files=None,
    np_random: np.random.Generator = None,
) -> Tuple[List[Any], List[H2MG]]:
    power_grids = sample_power_grids(
        backend=backend,
        data_dir=data_dir,
        size=size,
        deterministic=deterministic,
        rng_key=rng_key,
        valid_files=valid_files,
        np_random=np_random,
    )
    h2mgs_power_grids = [
        backend.get_h2mg_from_power_grid(power_grid, structure=observation_structure)
        for power_grid in power_grids
    ]
    return power_grids, h2mgs_power_grids


@ray.remote
class SEnv:
    backend: AbstractBackend
    data_dir: str
    valid_files: list

    def __init__(
        self,
        seed: int = 0,
        id=None,
        prefix="",
        backend: AbstractBackend = None,
        data_dir: str = None,
        observation_structure: H2MGStructure = None,
        sim_args: Dict = None,
        metric_fn=None,
        metric_fn_kwargs: Dict = None,
    ):
        self.data = 0
        self.id = id
        self.prefix = prefix
        self._np_random = np.random.default_rng(seed=seed)
        self.backend = backend
        self.valid_files = backend.get_valid_files(data_dir)
        self.data_dir = data_dir
        self.observation_structure = observation_structure
        self.sim_args = sim_args.copy()
        self.power_grid = None
        self.h2mg = None
        self.metric_fn = metric_fn
        self.metric_fn_kwargs = metric_fn_kwargs

    def __repr__(self) -> str:
        return f"SEnv_{self.prefix}{self.id:02}"

    def _reset(self, seed=None):
        self._np_random = np.random.default_rng(seed)
        return

    def _sample(
        self,
        rng_key: jax.random.PRNGKeyArray = None,
        deterministic: bool = False,
        file=None,
    ) -> H2MG:
        if file is None:
            power_grids, h2mgs = sample_power_grids_and_h2mgs_jax(
                self.backend,
                self.data_dir,
                size=1,
                observation_structure=self.observation_structure,
                deterministic=deterministic,
                rng_key=rng_key,
                valid_files=self.valid_files,
                np_random=self._np_random,
            )
        else:
            power_grid = self.backend.load_power_grid(file)
            h2mg = self.backend.get_h2mg_from_power_grid(
                power_grid, structure=self.observation_structure
            )
            power_grids = [power_grid]
            h2mgs = [h2mg]
        del self.power_grid
        self.power_grid = power_grids[0]
        self.h2mg = h2mgs[0]
        return self.h2mg

    def _apply(
        self,
        action: H2MG,
        metric_fn: Callable = None,
        metric_fn_kwargs: dict = None,
        clear_cache=False,
        save_folder=None,
        clip_action=True,
        clip_a_min=0.85,
        clip_a_max=1.15,
    ) -> Any:
        if clear_cache:
            jax.clear_caches()
        with jax.default_device(jax.devices("cpu")[0]):
            if clip_action:
                action.flat_array = np.clip(
                    action.flat_array, a_min=clip_a_min, a_max=clip_a_max
                )
            return default_pipeline(
                self.power_grid,
                action,
                backend=self.backend,
                metric_fn=self.metric_fn,
                sim_args=self.sim_args,
                save_folder=save_folder,
                **self.metric_fn_kwargs,
            )


class VEnv:
    envs: List[SEnv]
    num_envs: int
    multi_observation_space: H2MGSpace
    action_space: H2MGSpace
    metric_fn: Callable
    metric_fn_args: Dict

    def __init__(
        self,
        num_envs: int,
        multi_observation_space: H2MGSpace,
        action_space: H2MGSpace,
        metric_fn: Callable,
        metric_fn_args: Dict,
        seed: int = 0,
        prefix="",
        clear_cache=True,
        test_env=False,
        save_folder=None,
        *args,
        **kwargs,
    ):
        self.num_envs = num_envs
        self.envs = [
            SEnv.remote(
                *args,
                **kwargs,
                metric_fn=metric_fn,
                metric_fn_kwargs=metric_fn_args,
                seed=seed + idx,
                id=idx,
                prefix=prefix,
            )
            for idx in range(num_envs)
        ]
        self.multi_observation_space = multi_observation_space
        self.action_space = action_space
        self.metric_fn = metric_fn
        self.metric_fn_args = metric_fn_args
        self._default_obs = multi_observation_space.sample()
        self.apply_counter = 0
        self.clear_cache = clear_cache
        self.test_env = test_env
        self.save_folder = save_folder
        if test_env:
            self.backend = kwargs.get("backend", None)
            data_dir = kwargs.get("data_dir", None)
            self.valid_files = self.backend.get_valid_files(data_dir)
            self.cur_idx = 0

    def reset(self, seed=None):
        if self.test_env:
            self.cur_idx = 0
        ray.get(
            [env._reset.remote(seed=seed + idx) for idx, env in enumerate(self.envs)]
        )
        return

    def _concat(self, h2mgs: Sequence[H2MG]) -> H2MG:
        if len(h2mgs) != self.num_envs:
            warnings.warn(
                f"Number of h2mgs ({len(h2mgs)}) is not equal to number of envs ({self.num_envs})"
            )
        return concatenate(self.multi_observation_space, h2mgs, self._default_obs)

    def next_files(self):
        files = self.valid_files[self.cur_idx : self.cur_idx + self.num_envs]
        self.cur_idx += self.num_envs
        return files

    def sample(self, rng_key: jax.random.PRNGKeyArray = None) -> Tuple[H2MG, Dict]:
        if self.test_env:
            h2mgs = ray.get(
                [
                    env._sample.remote(file=file)
                    for env, file in zip(self.envs, self.next_files())
                ]
            )
            obs = self._concat(h2mgs)
            info = {}
            return obs, info
        if rng_key is None:
            h2mgs = ray.get([env._sample.remote() for env in self.envs])
        else:
            rng_keys = split(rng_key, self.num_envs)
            h2mgs = ray.get(
                [
                    env._sample.remote(rng_key)
                    for env, rng_key in zip(self.envs, rng_keys)
                ]
            )
        obs = self._concat(h2mgs)
        info = {}
        return obs, info

    def apply(self, action: Sequence[H2MG]) -> Tuple[Any, Dict]:
        self.apply_counter += 1
        actions = list(iterate(self.action_space, action))
        assert len(actions) == self.num_envs
        res = ray.get(
            [
                env._apply.remote(
                    a, clear_cache=self.clear_cache, save_folder=self.save_folder
                )
                for env, a in zip(self.envs, actions)
            ]
        )
        info = {}
        if (
            isinstance(res[0], Number)
            or isinstance(res[0], np.ndarray)
            or isinstance(res[0], jnp.ndarray)
        ):
            return np.array(res), info
        else:
            return res, info

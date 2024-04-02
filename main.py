import datetime
import json
import os
import pickle
import random
import time
import warnings
from functools import partial
from typing import Any, Callable, Dict, Sequence, Tuple

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray
from gymnasium.vector.utils import concatenate
from gymnasium.vector.utils.spaces import batch_space
from jax.random import PRNGKey, split
from omegaconf import OmegaConf
from tqdm import tqdm

from experiment_setup import exp_setup, policy_setup
from ml4ps.data.metrics import pp_cost, pp_cost_all
from ml4ps.data.ray_env import VEnv
from ml4ps.logger import get_logger
from ml4ps.reinforcement.policy import ContinuousPolicyNoEnv


def build_reinforce_loss(policy, multiple_actions: bool = False):
    def reinforce_loss(params, observations, actions, rewards, rngs, baseline=0):
        if multiple_actions:
            log_probs, _ = jax.vmap(
                policy.vmap_log_prob, in_axes=(None, None, 0), out_axes=0
            )(params, observations, actions)
        else:
            log_probs, _ = policy.vmap_log_prob(params, observations, action=actions)
        reinforce_loss = (log_probs * (rewards)).mean()
        loss = reinforce_loss
        return loss

    return reinforce_loss


def make_cst_lr(lr_value):
    def lr_fn(*args, **kwargs):
        return lr_value

    return lr_fn


def numpyfy(d, mean=False):
    if mean:
        d = {k: float(np.asarray(v).mean()) for k, v in d.items()}
    else:
        d = {k: np.asarray(v) for k, v in d.items()}
    return d


@partial(jax.jit, static_argnames=("batch_size", "loss_fn", "optimizer"))
def update_params(
    rng, params, obs, actions, metrics, batch_size, opt_state, loss_fn, optimizer
):
    rng, loss_rng = split(rng)
    value, grads = jax.value_and_grad(loss_fn)(
        params, obs, actions, jnp.array(metrics), split(loss_rng, batch_size)
    )
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), grads, updates, new_opt_state


def train_step(
    venv: VEnv,
    batch_size,
    params,
    rng,
    loss_fn,
    opt_state,
    optimizer,
    policy,
    det_sample,
    do_eval,
    n_action=1,
    normalize_signal=True,
    train_log=None,
):
    if train_log is None:
        train_log = do_eval
    # Sampling power grids
    # print("Sampling power grids")
    s0 = time.time()
    obs, _ = venv.sample()
    loading_time = time.time() - s0

    # Sample actions
    # print("Sample actions")
    s = time.time()
    rng, sample_rng = split(rng)
    actions, _, _ = policy.vmap_sample(
        params, obs, split(sample_rng, batch_size), det_sample, n_action=n_action
    )  # , action_space=venv.action_space) need for eval venv
    forward_time = time.time() - s

    # Apply action
    s = time.time()
    if n_action <= 1:
        metrics, _ = venv.apply(actions.as_numpy())
        if normalize_signal:
            metrics = metrics / (
                jnp.sqrt(jnp.nansum(metrics**2, axis=1, keepdims=True)) + 1e-8
            )
    else:
        metrics = []
        for action in actions:
            metric, _ = venv.apply(action.as_numpy())
            metrics.append(metric)
        metrics = np.stack(metrics, axis=0)
        metrics_raw = np.copy(metrics)
        if normalize_signal:
            metrics_mean = np.nanmean(metrics, axis=0, keepdims=True)
            metrics_std = np.nanstd(metrics, axis=0, keepdims=True) + 1e-8
            metrics = (metrics - metrics_mean) / metrics_std
        mutli_action_space = batch_space(venv.action_space, batch_size)
        mutli_action_space = batch_space(mutli_action_space, n_action)
        default_multi_action = mutli_action_space.sample()
        actions = concatenate(mutli_action_space, actions, default_multi_action)
    apply_time = time.time() - s

    # Update params
    s = time.time()
    new_params, grads, updates, new_opt_state = update_params(
        rng, params, obs, actions, metrics, batch_size, opt_state, loss_fn, optimizer
    )
    update_time = time.time() - s

    # Infos
    actions_flat = actions.flat_array
    if train_log:
        infos = {
            "grad_norm": optax.global_norm(grads),
            "update_norm": optax.global_norm(updates),
            "action_mean": np.nanmean(actions_flat),
            "action_max": np.nanmax(actions_flat),
            "action_min": np.nanmin(actions_flat),
            "action_mean_max": np.nanmax(actions_flat, axis=1).mean(),
            "action_mean_min": np.nanmin(actions_flat, axis=1).mean(),
            "loading_time": loading_time,
            "forward_time": forward_time,
            "apply_time": apply_time,
            "update_time": update_time,
            "train_step_time": time.time() - s0,
            "mean_metric": np.mean(np.asarray(metrics_raw)),
            "div_rate": np.mean(np.isclose(np.asarray(metrics_raw), 1)),
            "nan_rate": np.mean(np.isnan(np.asarray(metrics_raw))),
            "max_metric": np.max(np.asarray(metrics_raw)),
            "min_metric": np.min(np.asarray(metrics_raw)),
        }
    else:
        infos = {}

    return new_params, new_opt_state, infos


def do_eval_step(
    venv_eval, params, policy, do_eval, eval_batch_size, eval_size, run_dir
):
    if do_eval:
        if run_dir is not None:
            with open(os.path.join(run_dir, "last_params.pkl"), "wb") as f:
                pickle.dump(params, f)
            policy.save(os.path.join(run_dir, "last_policy.pkl"))
        eval_rng = PRNGKey(0)
        venv_eval.reset(seed=0)
        eval_info = eval_step_venv(
            venv_eval,
            eval_rng,
            policy,
            params,
            eval_batch_size,
            eval_size,
            deterministic=True,
            action_space=venv_eval.action_space,
        )
        eval_cost = eval_info["eval_cost"]
    else:
        eval_cost = None
        eval_info = {}
    return eval_info, eval_cost


def do_test_step(
    *,
    params,
    policy,
    eval_batch_size,
    eval_size,
    run_dir,
    test_data_dir,
    sim_args,
    metric_fn_args,
    save_results=True,
    env_name,
    seed=0,
):
    eval_rng = PRNGKey(0)
    path_parts = os.path.normpath(test_data_dir).split(os.path.sep)
    dataset_name = "_".join(path_parts[-2:])
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)
    if save_results:
        save_folder = os.path.join(run_dir, dataset_name)
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
    else:
        save_folder = None
    (
        backend,
        action_space,
        _,
        observation_space,
        observation_structure,
        _,
        _,
        _,
        _,
    ) = exp_setup(eval_batch_size, data_dir=test_data_dir, seed=seed, env_name=env_name)
    test_env = VEnv(
        eval_batch_size,
        batch_space(observation_space, n=eval_batch_size),
        action_space,
        pp_cost_all,
        metric_fn_args,
        clear_cache=False,
        seed=seed,
        backend=backend,
        data_dir=test_data_dir,
        observation_structure=observation_structure,
        sim_args=sim_args,
        test_env=True,
        save_folder=save_folder,
    )
    eval_info = eval_step_venv(
        test_env,
        eval_rng,
        policy,
        params,
        eval_batch_size,
        eval_size,
        deterministic=True,
        action_space=test_env.action_space,
    )
    with open(os.path.join(run_dir, f"info_{dataset_name}.json"), "w") as f:
        json.dump(numpyfy(eval_info, mean=True), f, indent=2)
    eval_cost = eval_info["eval_cost"]
    return eval_info, eval_cost


def eval_policy(
    folder,
    data_dir,
    num_envs,
    seed=0,
    cost_args=None,
    sim_args={},
    eval_size=256,
    cst_sigma=0.05,
    env_name="VoltageManagementPandapowerV1",
    save_results=False,
):
    policy = ContinuousPolicyNoEnv(
        file=os.path.join(folder, "best_policy.pkl"), cst_sigma=cst_sigma
    )
    with open(os.path.join(folder, "best_params.pkl"), "rb") as f:
        params = pickle.load(f)
    (
        backend,
        action_space,
        control_structure,
        observation_space,
        observation_structure,
        _,
        _,
        _,
        _,
    ) = exp_setup(num_envs, data_dir=data_dir, seed=seed, env_name=env_name)
    metric_fn_args = {"structure": control_structure, **cost_args}
    path_parts = os.path.normpath(data_dir).split(os.path.sep)
    dataset_name = "_".join(path_parts[-2:])
    if save_results:
        save_folder = os.path.join(folder, dataset_name)
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)
    else:
        save_folder = None
    eval_env = VEnv(
        num_envs,
        batch_space(observation_space, n=num_envs),
        action_space,
        pp_cost_all,
        metric_fn_args,
        clear_cache=False,
        seed=seed,
        backend=backend,
        data_dir=data_dir,
        observation_structure=observation_structure,
        sim_args=sim_args,
        test_env=True,
        save_folder=save_folder,
    )
    infos = eval_step_venv(
        eval_env,
        PRNGKey(seed),
        policy,
        params,
        num_envs,
        eval_size=eval_size,
        deterministic=True,
        action_space=action_space,
    )

    with open(os.path.join(folder, f"info_{dataset_name}.json"), "w") as f:
        json.dump(numpyfy(infos, mean=True), f, indent=2)
    return infos


def eval_step_venv(
    venv: VEnv,
    rng,
    policy,
    params,
    eval_batch_size,
    eval_size,
    deterministic=True,
    action_space=None,
):
    costs = []
    all_actions = []
    for i in tqdm(
        range(eval_size // eval_batch_size),
        leave=False,
        desc="Evaluation step",
        unit_scale=eval_batch_size,
    ):
        rng, sample_rng = split(rng)
        obs, _ = venv.sample()
        # Sample actions
        rng, sample_rng = split(rng)
        if action_space is not None:
            actions, _, _ = policy.vmap_sample(
                params,
                obs,
                split(sample_rng, eval_batch_size),
                deterministic,
                action_space=action_space,
            )
        else:
            actions, _, _ = policy.vmap_sample(
                params, obs, split(sample_rng, eval_batch_size), deterministic
            )

        cost, _ = venv.apply(actions.as_numpy())

        costs.extend(cost)
        all_actions.append(actions.flat_array)

    cost_item = costs[0]
    _cost = dict()
    if isinstance(cost_item, dict):
        for k in cost_item.keys():
            _cost["eval_" + k] = np.nanmean(
                np.array([single_cost_metric[k] for single_cost_metric in costs])
            )
            if "max" in k:
                _cost["eval_" + k + "_max"] = np.nanmax(
                    np.array([single_cost_metric[k] for single_cost_metric in costs])
                )
            if "min" in k:
                _cost["eval_" + k + "_min"] = np.nanmin(
                    np.array([single_cost_metric[k] for single_cost_metric in costs])
                )
    cost_info = _cost
    all_actions = np.vstack(all_actions)
    action_info = {
        "eval_action_mean": np.nanmean(all_actions),
        "eval_action_max": np.nanmax(all_actions),
        "eval_action_min": np.nanmin(all_actions),
        "eval_action_mean_max": np.nanmax(all_actions, axis=1).mean(),
        "eval_action_mean_min": np.nanmin(all_actions, axis=1).mean(),
    }
    return {**cost_info, **action_info}


def make_run_name(name=None):
    if name is None:
        return f'run_{random.randint(0,512):03d}_{datetime.datetime.now().strftime("%m%d%Y_%H%M%S")}'
    else:
        return name


def optim_setup(params, clip_norm=1000, cst_lr=3e-4):
    if isinstance(cst_lr, float):
        lr = make_cst_lr(cst_lr)
    else:
        lr = cst_lr
    optimizer = optax.adam(learning_rate=lr)
    optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optimizer)
    opt_state = optimizer.init(params)
    return optimizer, opt_state, lr


def loss_setup(method="reinforce_exact", multiple_actions=False, policy=None):
    if method == "reinforce_estimate":
        loss_fn = build_reinforce_loss(policy, multiple_actions=multiple_actions)
        metric_fn = pp_cost
        det_sample = False
    else:
        raise ValueError("Choose in ['reinforce_estimate']")

    return loss_fn, metric_fn, det_sample


def logger_setup(*, exp_name: str, run_name: str, run_dir: str | None = None):
    if run_dir is None:
        run_dir = os.path.join(exp_name, run_name)
        if not os.path.exists(exp_name):
            os.mkdir(exp_name)
        if not os.path.exists(run_dir):
            os.mkdir(run_dir)
    logger = get_logger(
        name="tensorboard",
        experiment_name=exp_name,
        run_name=run_name,
        res_dir=exp_name,
        run_dir=run_dir,
    )
    return logger, run_dir


def save_config(cfg):
    if not os.path.isdir(cfg.res_dir):
        os.mkdir(cfg.res_dir)
    if cfg.run_name is None:
        if cfg.hparams is None:
            run_name = f'algo_{random.randint(0,512):03d}_{datetime.datetime.now().strftime("%m%d%Y_%H%M%S")}'
        else:
            run_name = (
                build_run_name(cfg, cfg.hparams) + f"_{random.randint(0,512):03d}"
            )
    else:
        run_name = cfg.run_name
    run_dir = os.path.join(cfg.res_dir, run_name)
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    return run_dir, run_name


def get_hparam_value(cfg: OmegaConf, hparam_name: str):
    keys = hparam_name.split(".")
    value = cfg
    for k in keys:
        value = value[k]
    return value


def build_run_name(cfg, hparam_names):
    run_names = []
    for hparam_name in hparam_names:
        value = get_hparam_value(cfg, hparam_name)
        run_names.append(f"{hparam_name}={value}")
    return "_".join(run_names)


@hydra.main(version_base=None, config_path="config", config_name="pscc")
def main(cfg):
    ray.init(dashboard_host="0.0.0.0")
    print(ray.available_resources())
    run_dir, run_name = save_config(cfg)
    cfg.run_name = run_name
    return main_args(**cfg, run_dir=run_dir)


def main_args(
    exp_name: str = "default_exp",
    run_name: str = "debug",
    run_dir=None,
    method: str = "estimate",
    seed: int = 0,
    batch_size: int = 8,
    eval_batch_size: int = 4,
    data_dir="../data/case60nordic/vanilla_no_filter/train",
    eval_data_dir="../data/case60nordic/vanilla_no_filter/val_2000",
    cost_args: dict = {
        "eps_v": 0.05,
        "eps_i": 0.1,
        "eps_q": 0.5,
        "lmb_v": 200,
        "lmb_i": 200,
        "lmb_l": 1,
        "lmb_q": 0,
    },
    n_iter: int = 1000,
    n_action: int = 1,
    eval_size: int = 256,
    eval_every_n_iter: int = 20,
    train_log_every_n_iter: int = 20,
    clip_norm: int = 10,
    cst_lr: float = 3e-4,
    cst_sigma: float = 0.05,
    nn_args: dict | None = {
        "local_encoder_hidden_size": [32],
        "global_encoder_hidden_size": [32],
        "local_dynamics_hidden_size": [64],
        "global_dynamics_hidden_size": [64],
        "local_decoder_hidden_size": [32],
        "global_decoder_hidden_size": [32],
        "local_encoder_output": 32,
        "global_encoder_output": 32,
        "local_latent_dimension": 32,
        "global_latent_dimension": 32,
        "dt0": 0.01,
    },
    sim_args: dict | None = {
        "enforce_q_lims": False,
        "delta_q": 0.0,
        "recycle": {"bus_pq": False, "gen": True, "trafo": False},
        "init": "results",
    },
    normalize_signal: bool = True,
    clear_cache: bool = True,
    env_name: str = "VoltageManagementPandapowerV1",
    test_data_dir: str|None = None,
    test_size: int|None = None,
    **kwargs,
):

    # Logger
    logger, run_dir = logger_setup(
        exp_name=exp_name, run_name=run_name, run_dir=run_dir
    )

    # Variables
    rng_key = PRNGKey(seed)
    default_rng = rng_key
    multiple_actions = n_action > 1

    # General Setup
    (
        backend,
        action_space,
        control_structure,
        _,
        observation_structure,
        normalizer,
        init_h2mg,
        _,
        multi_observation_space,
    ) = exp_setup(batch_size, data_dir=data_dir, seed=seed, env_name=env_name)
    (
        _,
        eval_action_space,
        eval_control_structure,
        eval_observation_space,
        eval_observation_structure,
        _,
        _,
        _,
        _,
    ) = exp_setup(eval_batch_size, data_dir=eval_data_dir, seed=seed, env_name=env_name)
    metric_fn_args = {"structure": control_structure, **cost_args}
    eval_metric_fn_args = {"structure": eval_control_structure, **cost_args}

    # Policy Setup
    policy, params = policy_setup(
        action_space, cst_sigma, normalizer, default_rng, init_h2mg, nn_args=nn_args
    )

    # Metric and Loss
    loss_fn, metric_fn, det_sample = loss_setup(
        method=method, multiple_actions=multiple_actions, policy=policy
    )

    # Environment
    venv = VEnv(
        batch_size,
        multi_observation_space,
        action_space,
        metric_fn,
        metric_fn_args,
        seed=seed,
        backend=backend,
        data_dir=data_dir,
        observation_structure=observation_structure,
        sim_args=sim_args,
        prefix="train_",
        clear_cache=clear_cache,
    )
    venv_eval = VEnv(
        eval_batch_size,
        batch_space(eval_observation_space, n=eval_batch_size),
        eval_action_space,
        pp_cost_all,
        eval_metric_fn_args,
        seed=seed,
        backend=backend,
        data_dir=eval_data_dir,
        observation_structure=eval_observation_structure,
        sim_args=sim_args,
        prefix="eval_",
        clear_cache=clear_cache,
        test_env=True,
    )

    # Optimizer
    optimizer, opt_state, lr = optim_setup(params, clip_norm=clip_norm, cst_lr=cst_lr)

    # Training loop
    # tracker = SummaryTracker()
    best_cost = np.inf
    best_iter = 0
    for i in tqdm(range(n_iter)):
        default_rng, rng = split(default_rng, 2)
        # Train
        params, opt_state, infos = train_step(
            venv,
            batch_size,
            params,
            rng,
            loss_fn,
            opt_state,
            optimizer,
            policy,
            det_sample,
            do_eval=(i % eval_every_n_iter == 0),
            n_action=n_action,
            normalize_signal=normalize_signal,
            train_log=(i % train_log_every_n_iter == 0),
        )
        # Eval
        eval_info, cost = do_eval_step(
            venv_eval=venv_eval,
            params=params,
            policy=policy,
            do_eval=(i % eval_every_n_iter == 0),
            eval_batch_size=eval_batch_size,
            eval_size=eval_size,
            run_dir=run_dir,
        )
        infos = {**infos, **eval_info}

        # Log and Save
        if cost is not None and cost < best_cost:
            best_cost = cost
            best_iter = i
            with open(os.path.join(run_dir, "best_params.pkl"), "wb") as f:
                pickle.dump(params, f)
            policy.save(os.path.join(run_dir, "best_policy.pkl"))
            with open(os.path.join(run_dir, "best_info.json"), "w") as f:
                json.dump(
                    numpyfy({"iteration": i, "eval_cost": cost, **infos}, mean=True),
                    f,
                    indent=2,
                )
            with open(os.path.join(run_dir, "best_opt_state.pkl"), "wb") as f:
                pickle.dump(opt_state, f)
            with open(os.path.join(run_dir, "best_cost.json"), "w") as f:
                json.dump({"best_cost": best_cost}, f, indent=2)

        logger.log_metrics_dict(
            numpyfy({**infos, "lr": lr(i), "best_iter": best_iter}), step=i
        )

    # Test
    eval_info, cost = do_test_step(
        params=params,
        policy=policy,
        eval_batch_size=eval_batch_size,
        eval_size=test_size,
        run_dir=run_dir,
        test_data_dir=test_data_dir,
        sim_args=sim_args,
        metric_fn_args=eval_metric_fn_args,
        save_results=True,
        seed=0,
        env_name=env_name,
    )

    return


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
        warnings.simplefilter(action="ignore", category=FutureWarning)
        main()

import os
import warnings
from typing import Any, Callable, Dict, Sequence, Tuple

import jax
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import concatenate
from gymnasium.vector.utils.spaces import batch_space
from jax.random import PRNGKey, split

from ml4ps import (
    H2MG,
    H2MGStructure,
    HyperEdgesStructure,
    PaddingWrapper,
    PandaPowerBackend,
)
from ml4ps.backend import AbstractBackend, PaddingWrapper, PandaPowerBackend
from ml4ps.h2mg import H2MGNormalizer, H2MGSpace, HyperEdgesSpace
from ml4ps.reinforcement.policy import ContinuousPolicyNoEnv

warnings.simplefilter(action="ignore", category=FutureWarning)


def only_json(files):
    return [file for file in files if file.endswith(".json")]


def sample_power_grids(
    backend: AbstractBackend,
    data_dir,
    size,
    deterministic=False,
    rng_key=None,
    valid_files=None,
):
    if valid_files is None:
        valid_files = only_json(backend.get_valid_files(data_dir))
    if deterministic:
        files = valid_files[:size]
    else:
        if rng_key is not None:
            idx = jax.random.choice(rng_key, len(valid_files), shape=(size,))
            files = [valid_files[_idx] for _idx in idx]
        else:
            files = np.random.choice(valid_files, size=size)
    power_grids = [backend.load_power_grid(file) for file in files]
    return power_grids


def sample_power_grids_and_h2mgs(
    backend: AbstractBackend,
    data_dir,
    size,
    observation_structure,
    deterministic=False,
    valid_files=None,
):
    power_grids = sample_power_grids(
        backend=backend,
        data_dir=data_dir,
        size=size,
        deterministic=deterministic,
        valid_files=valid_files,
    )
    h2mgs_power_grids = [
        backend.get_h2mg_from_power_grid(power_grid, structure=observation_structure)
        for power_grid in power_grids
    ]
    return power_grids, h2mgs_power_grids


def sample_power_grids_and_h2mgs_jax(
    backend: AbstractBackend,
    data_dir,
    size,
    observation_structure,
    deterministic=False,
    rng_key=None,
    valid_files=None,
):
    power_grids = sample_power_grids(
        backend=backend,
        data_dir=data_dir,
        size=size,
        deterministic=deterministic,
        rng_key=rng_key,
        valid_files=valid_files,
    )
    h2mgs_power_grids = [
        backend.get_h2mg_from_power_grid(power_grid, structure=observation_structure)
        for power_grid in power_grids
    ]
    return power_grids, h2mgs_power_grids


def policy_setup(action_space, cst_sigma, normalizer, rng, h2mg, nn_args={}) -> Tuple:
    policy = ContinuousPolicyNoEnv(
        action_space, cst_sigma=cst_sigma, normalizer=normalizer, nn_args=nn_args
    )
    params = policy.init(rng, h2mg)
    return policy, params


def exp_setup(
    batch_size,
    data_dir=None,
    return_type="tuple",
    env_name="VoltageManagementPandapowerV1GenOnly",
    seed=0,
):

    OBSERVATION_STRUCTURE = get_observation_structure()

    CONTROL_STRUCTURE = get_control_structure(env_name)

    backend = PaddingWrapper(PandaPowerBackend())

    if not os.path.isdir(os.path.join(data_dir, env_name)):
        os.mkdir(os.path.join(data_dir, env_name))

    control_structure, action_space = build_action_struct_and_space(
        data_dir, env_name, CONTROL_STRUCTURE, backend
    )

    observation_structure, observation_space = build_obs_struct_and_space(
        data_dir, env_name, OBSERVATION_STRUCTURE, backend
    )

    normalizer = build_normalizer(backend, data_dir, observation_structure, env_name)

    default_rng = PRNGKey(seed)
    multi_action_space = batch_space(action_space, n=batch_size)
    multi_observation_space = batch_space(observation_space, n=batch_size)
    power_grids, h2mgs = sample_power_grids_and_h2mgs_jax(
        backend,
        data_dir,
        1,
        observation_structure,
        deterministic=False,
        rng_key=default_rng,
    )

    if return_type == "tuple":
        return (
            backend,
            action_space,
            control_structure,
            observation_space,
            observation_structure,
            normalizer,
            h2mgs[0],
            multi_action_space,
            multi_observation_space,
        )
    elif return_type == "dict":
        return {
            "action_space": action_space,
            "observation_space": observation_space,
            "observation_structure": observation_structure,
            "control_structure": control_structure,
            "backend": backend,
            "normalizer": normalizer,
            "multi_action_space": multi_action_space,
            "multi_observation_space": multi_observation_space,
        }
    else:
        raise ValueError("Choose return_type in 'tuple' or 'dict'")


def build_obs_struct_and_space(data_dir, env_name, OBSERVATION_STRUCTURE, backend):
    observation_structure_name = os.path.join(
        data_dir, env_name, "observation_structure.pkl"
    )

    observation_structure = backend.get_max_structure(
        observation_structure_name, data_dir, OBSERVATION_STRUCTURE
    )

    observation_space = H2MGSpace.from_structure(observation_structure)
    return observation_structure, observation_space


def build_action_struct_and_space(data_dir, env_name, CONTROL_STRUCTURE, backend):
    control_structure_name = os.path.join(data_dir, env_name, "control_structure.pkl")
    control_structure = backend.get_max_structure(
        control_structure_name, data_dir, CONTROL_STRUCTURE
    )
    action_space = build_action_space(control_structure, additive=False)
    return control_structure, action_space


def get_control_structure(env_name):
    CONTROL_STRUCTURE = H2MGStructure()
    CONTROL_STRUCTURE.add_local_hyper_edges_structure(
        "gen", HyperEdgesStructure(features=["vm_pu"])
    )
    if env_name == "VoltageManagementPandapowerV1":
        CONTROL_STRUCTURE.add_local_hyper_edges_structure(
            "ext_grid", HyperEdgesStructure(features=["vm_pu"])
        )

    return CONTROL_STRUCTURE


def build_action_space(control_structure, additive=True):

    if additive:
        offset = 0.0
    else:
        offset = 1.0
    scale = 0.2
    gen_vm_pu_space = spaces.Box(
        low=offset - scale,
        high=offset + scale,
        shape=(control_structure["gen"].features["vm_pu"],),
    )

    action_space = H2MGSpace()
    action_space._add_hyper_edges_space(
        "gen", HyperEdgesSpace(features=spaces.Dict({"vm_pu": gen_vm_pu_space}))
    )
    if "ext_grid" in control_structure:
        ext_grid_vm_pu_space = spaces.Box(
            low=offset - scale,
            high=offset + scale,
            shape=(control_structure["ext_grid"].features["vm_pu"],),
        )
        action_space._add_hyper_edges_space(
            "ext_grid",
            HyperEdgesSpace(features=spaces.Dict({"vm_pu": ext_grid_vm_pu_space})),
        )

    return action_space


def build_normalizer(
    backend, data_dir, observation_structure, env_name, normalizer_args=None
):
    normalizer_dir = os.path.join(data_dir, env_name)
    normalize_path = os.path.join(normalizer_dir, "normalizer.pkl")
    if os.path.exists(normalize_path):
        return H2MGNormalizer(filename=normalize_path)
    if normalizer_args is None:
        normalizer = H2MGNormalizer(
            backend=backend, structure=observation_structure, data_dir=data_dir
        )
    else:
        normalizer = H2MGNormalizer(
            backend=backend,
            structure=observation_structure,
            data_dir=data_dir,
            **normalizer_args,
        )

    if not os.path.exists(normalize_path):
        if not os.path.exists(normalizer_dir):
            os.mkdir(normalizer_dir)
        normalizer.save(normalize_path)
    return normalizer


def get_observation_structure():
    OBSERVATION_STRUCTURE = H2MGStructure()

    bus_structure = HyperEdgesStructure(
        addresses=["id"], features=["in_service", "max_vm_pu", "min_vm_pu", "vn_kv"]
    )
    OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("bus", bus_structure)

    load_structure = HyperEdgesStructure(
        addresses=["bus_id"],
        features=[
            "const_i_percent",
            "const_z_percent",
            "controllable",
            "in_service",
            "p_mw",
            "q_mvar",
            "scaling",
            "sn_mva",
        ],
    )
    OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("load", load_structure)

    sgen_structure = HyperEdgesStructure(
        addresses=["bus_id"],
        features=[
            "in_service",
            "p_mw",
            "q_mvar",
            "scaling",
            "sn_mva",
            "current_source",
        ],
    )
    OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("sgen", sgen_structure)

    gen_structure = HyperEdgesStructure(
        addresses=["bus_id"],
        features=[
            "controllable",
            "in_service",
            "p_mw",
            "scaling",
            "sn_mva",
            "vm_pu",
            "slack",
            "max_p_mw",
            "min_p_mw",
            "max_q_mvar",
            "min_q_mvar",
            "slack_weight",
        ],
    )
    OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("gen", gen_structure)

    shunt_structure = HyperEdgesStructure(
        addresses=["bus_id"],
        features=["q_mvar", "p_mw", "vn_kv", "step", "max_step", "in_service"],
    )
    OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("shunt", shunt_structure)

    ext_grid_structure = HyperEdgesStructure(
        addresses=["bus_id"],
        features=[
            "in_service",
            "va_degree",
            "vm_pu",
            "max_p_mw",
            "min_p_mw",
            "max_q_mvar",
            "min_q_mvar",
            "slack_weight",
        ],
    )
    OBSERVATION_STRUCTURE.add_local_hyper_edges_structure(
        "ext_grid", ext_grid_structure
    )

    line_structure = HyperEdgesStructure(
        addresses=["from_bus_id", "to_bus_id"],
        features=[
            "c_nf_per_km",
            "df",
            "g_us_per_km",
            "in_service",
            "length_km",
            "max_i_ka",
            "max_loading_percent",
            "parallel",
            "r_ohm_per_km",
            "x_ohm_per_km",
        ],
    )
    OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("line", line_structure)

    trafo_structure = HyperEdgesStructure(
        addresses=["hv_bus_id", "lv_bus_id"],
        features=[
            "df",
            "i0_percent",
            "in_service",
            "max_loading_percent",
            "parallel",
            "pfe_kw",
            "shift_degree",
            "sn_mva",
            "tap_max",
            "tap_neutral",
            "tap_min",
            "tap_phase_shifter",
            "tap_pos",
            "tap_side",
            "tap_step_degree",
            "tap_step_percent",
            "vn_hv_kv",
            "vn_lv_kv",
            "vk_percent",
            "vkr_percent",
        ],
    )
    OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("trafo", trafo_structure)
    return OBSERVATION_STRUCTURE

from numbers import Number
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np


def null_metric(power_grid, *, backend, **kwargs):
    return


def make_cst_metric(cst_value):
    def cst_metric_fn(power_grid, *, backend, **kwargs):
        return cst_value

    return cst_metric_fn


def pp_cost(power_grid, *, backend, structure=None, **kwargs):
    return cost(power_grid, **kwargs)


def pp_cost_all(power_grid, *, backend, structure=None, **kwargs):
    return cost(power_grid, detailled=True, **kwargs)


def _normalized_cost_np(
    value, min_value, max_value, eps_min_threshold, eps_max_threshold
) -> Number:
    v = (value - min_value) / (max_value - min_value)
    return np.nanmean(
        np.greater(v, 1 - eps_max_threshold) * np.power(v - (1 - eps_max_threshold), 2)
        + np.greater(eps_min_threshold, v) * np.power(eps_min_threshold - v, 2)
    )


def c_v(bus_voltage, min_bus_voltage=0.9, max_bus_voltage=1.1, eps_v=0.05) -> Number:
    return _normalized_cost_np(
        bus_voltage, min_bus_voltage, max_bus_voltage, eps_v, eps_v
    )


def c_i(loading_percent, eps_i=0.1) -> Number:
    return _normalized_cost_np((loading_percent / 100) ** 2, -1, 1, 0, eps_i)


def c_l(total_losses, total_load) -> Number:
    return total_losses / total_load


def c_q(q_mvar, min_q_mvar, max_q_mvar, eps_q=0.5) -> Number:
    return _normalized_cost_np(q_mvar, min_q_mvar, max_q_mvar, eps_q, eps_q)


def c(
    bus_voltage,
    loading_percent,
    total_losses,
    q_mvar,
    total_load,
    min_q_mvar,
    max_q_mvar,
    min_bus_voltage=0.9,
    max_bus_voltage=1.1,
    eps_v=0.05,
    eps_i=0.1,
    eps_q=0.5,
    lmb_v=1,
    lmb_i=1,
    lmb_l=1,
    lmb_q=1,
    detailled=False,
    diverged=None,
    c_div=1,
) -> Number | Dict:
    if diverged:
        cv = np.nan
        ci = np.nan
        cl = np.nan
        cq = np.nan
    else:
        cv = c_v(
            bus_voltage,
            min_bus_voltage=min_bus_voltage,
            max_bus_voltage=max_bus_voltage,
            eps_v=eps_v,
        )
        ci = c_i(loading_percent, eps_i=eps_i)
        cl = c_l(total_losses, total_load)
        cq = c_q(q_mvar, min_q_mvar, max_q_mvar, eps_q=eps_q)
    cost_value = lmb_v * cv + lmb_i * ci + lmb_l * cl + lmb_q * cq
    if detailled:
        if diverged:
            return {
                "cost": c_div,
                "lv_cv": np.nan,
                "li_ci": np.nan,
                "ll_cl": np.nan,
                "lq_cq": np.nan,
                "cv": np.nan,
                "ci": np.nan,
                "cl": np.nan,
                "cq": np.nan,
                "over_v": np.nan,
                "v_violated": np.nan,
                "is_v_violated": np.nan,
                "i_violated": np.nan,
                "is_i_violated": np.nan,
                "q_violated": np.nan,
                "is_q_violated": np.nan,
                "is_violated": np.nan,
                "under_v": np.nan,
                "max_v": np.nan,
                "min_v": np.nan,
                "diverged": diverged,
            }
        v_violated = np.logical_or(
            bus_voltage > max_bus_voltage, bus_voltage < min_bus_voltage
        )
        is_v_violated = np.any(v_violated)
        i_violated = loading_percent > 100
        is_i_violated = np.any(i_violated)
        q_violated = np.logical_or(q_mvar < min_q_mvar, q_mvar > max_q_mvar)
        is_q_violated = np.any(q_violated)
        return {
            "cost": cost_value,
            "lv_cv": lmb_v * cv,
            "li_ci": lmb_i * ci,
            "ll_cl": lmb_l * cl,
            "lq_cq": lmb_q * cq,
            "cv": cv,
            "ci": ci,
            "cl": cl,
            "cq": cq,
            "over_v": np.mean(bus_voltage > max_bus_voltage),
            "v_violated": v_violated.mean(),
            "is_v_violated": is_v_violated,
            "i_violated": i_violated.mean(),
            "is_i_violated": is_i_violated,
            "q_violated": q_violated.mean(),
            "is_q_violated": is_q_violated,
            "is_violated": np.logical_or(
                np.logical_or(is_v_violated, is_i_violated), is_q_violated
            ),
            "under_v": np.mean(bus_voltage < min_bus_voltage),
            "max_v": np.max(bus_voltage),
            "min_v": np.min(bus_voltage),
            "diverged": diverged,
        }
    if diverged:
        return c_div
    return cost_value


def cost(
    net,
    min_bus_voltage=0.9,
    max_bus_voltage=1.1,
    eps_v=0.05,
    eps_i=0.1,
    eps_q=0.5,
    lmb_v=1,
    lmb_i=1,
    lmb_l=1,
    lmb_q=1,
    detailled=False,
    c_div=1,
) -> Number | Dict:
    bus_vm_pu = net.res_bus.vm_pu

    gen_mvar = np.array(net.res_gen.q_mvar)
    gen_min_mvar = np.array(net.gen.min_q_mvar)
    gen_max_mvar = np.array(net.gen.max_q_mvar)

    line_loading_percent = np.array(net.res_line.loading_percent)
    transfo_loading_percent = np.array(net.res_trafo.loading_percent)
    loading_percent = np.concatenate(
        [line_loading_percent, transfo_loading_percent], axis=-1
    )

    bus_p_mw = np.array(net.res_bus.p_mw)
    total_losses = -np.sum(
        bus_p_mw
    )  # jnp.sum(ext_grid_p_mw) +  + jnp.sum(gen_p_mw) + jnp.sum(loads)# jnp.sum(bus_p_mw)
    total_load = np.array(net.load.p_mw).sum()

    if hasattr(net, "converged"):
        diverged = not net.converged
    else:
        diverged = np.nan
    return c(
        bus_vm_pu,
        loading_percent,
        total_losses,
        gen_mvar,
        total_load,
        gen_min_mvar,
        gen_max_mvar,
        min_bus_voltage=min_bus_voltage,
        max_bus_voltage=max_bus_voltage,
        eps_v=eps_v,
        eps_i=eps_i,
        eps_q=eps_q,
        lmb_v=lmb_v,
        lmb_i=lmb_i,
        lmb_l=lmb_l,
        lmb_q=lmb_q,
        detailled=detailled,
        diverged=diverged,
        c_div=c_div,
    )

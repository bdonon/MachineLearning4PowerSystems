import jax.numpy as jnp
import jax.nn as jnn
import diffrax
import jax
import flax

from flax import linen as nn
from typing import Sequence

from ml4ps.h2mg import H2MG, H2MGStructure, HyperEdges

LOCAL_KEY = "local"
GLOBAL_KEY = "global"
LOCAL_ENCODER_KEY = "local_encoder"
GLOBAL_ENCODER_KEY = "global_encoder"
LOCAL_DYNAMICS_KEY = "local_dynamics"
GLOBAL_DYNAMICS_KEY = "global_dynamics"
LOCAL_DECODER_KEY = "local_decoder"
GLOBAL_DECODER_KEY = "global_decoder"

MAX_INTEGER = 2147483647

def smooth_activation(x, alpha=0.1):
    return alpha * x + (1-alpha) * (jnp.log(1+jnp.exp(x)) - jnp.log(2))

def identity_fn(x):
    return x

def get_activation(name, **kwargs):
    if name == "tanh":
        return jnn.tanh
    elif name == "leaky_relu":
        return jnn.leaky_relu
    elif name == "relu":
        return jnn.relu
    elif name == "smooth_leaky_relu":
        return smooth_activation
    elif name == "identity" or name == "none":
        return identity_fn
    else:
        raise ValueError("Unknown Activation function")

def nan_mean_at(h, a):
    clean_a = jnp.nan_to_num(a, nan=MAX_INTEGER).astype(int)
    sum = jnp.nansum(h.at[clean_a].get(mode='drop', fill_value=0.), axis=0, keepdims=True)
    count = jnp.nansum((1+0*h).at[clean_a].get(mode='drop', fill_value=0.), axis=0, keepdims=True)
    return sum / count

def nan_sum_at(h, a):
    clean_a = jnp.nan_to_num(a, nan=MAX_INTEGER).astype(int)
    sum = jnp.nansum(h.at[clean_a].get(mode='drop', fill_value=0.), axis=0, keepdims=True)
    count = jnp.nansum((1+0*h).at[clean_a].get(mode='drop', fill_value=0.), axis=0, keepdims=True)
    return jnn.tanh(sum)

def nan_max_at(h, a):
    clean_a = jnp.nan_to_num(a, nan=MAX_INTEGER).astype(int)
    max = jnp.nanmax(h.at[clean_a].get(mode='drop', fill_value=0.), axis=0, keepdims=True)
    count = jnp.nansum((1+0*h).at[clean_a].get(mode='drop', fill_value=0.), axis=0, keepdims=True)
    return max


class MLP(nn.Module):
    """Multi-Layer Perceptron. Building block of the broader H2MGNODE architecture.

    Attributes:
        hidden_size (:obj:`typing.Sequence` of :obj:`int`): List of sizes of the MLP.
        out_size (:obj:`int`): Output size of the MLP.
        sigma (:obj:`nn.activation`): Non-linearity
    """

    hidden_size: Sequence[int]
    out_size: int
    sigma: nn.activation
    use_bias: bool=True

    @nn.compact
    def __call__(self, x):
        for i, d in enumerate(self.hidden_size):
            x = nn.Dense(d)(x)
            x = self.sigma(x)
        return nn.Dense(self.out_size, use_bias=self.use_bias)(x)


class LocalEncoder(nn.Module):
    """Encodes local hyper-edges variables."""
    hidden_size: Sequence[int]
    out_size: int
    activation: str

    @nn.compact
    def __call__(self, h2mg_in: H2MG):
        r = {}
        for k, hyper_edges in h2mg_in.local_hyper_edges.items():
            mlp = MLP(self.hidden_size, self.out_size, get_activation(self.activation), name="{}".format(k))
            r[k] = mlp(jnp.nan_to_num(hyper_edges.array, nan=0.))
            r[k] = jnp.where(jnp.isnan(r[k]), jnp.nan, jnp.nan_to_num(r[k], nan=0.))
        return r


class GlobalEncoder(nn.Module):
    """Encodes global variables."""
    hidden_size: Sequence[int]
    out_size: int
    activation: str

    @nn.compact
    def __call__(self, h2mg_in: H2MG):
        if h2mg_in.global_hyper_edges is not None:
            mlp = MLP(self.hidden_size, self.out_size, get_activation(self.activation), name="global")
            r = mlp(jnp.nan_to_num(h2mg_in.global_hyper_edges.array, nan=0.))
            r = jnp.where(jnp.isnan(r), jnp.nan, jnp.nan_to_num(r, nan=0.))
            return r
        else:
            return jnp.array([[]])


class LocalDynamics(nn.Module):
    r"""Computes the second term of the Neural Ordinary Differential Equation that concerns local latent variables.

    .. math::
            \forall i \in \{1, \dots, n\}, \frac{dh_i}{dt} = \tanh(\sum_{(c,e,o) \in \mathcal{N}(i)} \tanh
            \Phi_\theta^{c,o} (x^g, x^c_e, h^g, (h_{o(e)})_{o\in \mathcal{O}^c}, t)

    Attributes:
        hidden_size (:obj:`typing.Sequence` of :obj:`int`): List of sizes of hidden layers of the MLPs
            :math:`(\Phi_\theta^{c,o})_{c \in \mathcal{C}, o\in \mathcal{O}^c}`.
        out_size (:obj:`int`): Output size of the MLPs
            :math:`(\Phi_\theta^{c,o})_{c \in \mathcal{C}, o\in \mathcal{O}^c}`.
    """
    hidden_size: Sequence[int]
    out_size: int
    activation: str
    final_activation: str
    global_msg: bool


    @nn.compact
    def __call__(self, h2mg_in: H2MG, h2mg_encoded: dict, h: dict, t: float):
        delta_sum = 0 * h[LOCAL_KEY]
        count = 0
        for hyper_edge_name, hyper_edges in h2mg_in.local_hyper_edges.items():
            if hyper_edges.addresses is not None:

                ones = jnp.ones_like(h2mg_encoded[LOCAL_KEY][hyper_edge_name])[:, :1]
                if self.global_msg:
                    global_mask = jnp.ones_like(h2mg_encoded[LOCAL_KEY][hyper_edge_name])[:, :1]
                else:
                    global_mask = jnp.zeros_like(h2mg_encoded[LOCAL_KEY][hyper_edge_name])[:, :1]
                
                nn_input_list = [
                    h[GLOBAL_KEY] * global_mask, # GLOBAL_MSG
                    h2mg_encoded[LOCAL_KEY][hyper_edge_name],
                    h2mg_encoded[GLOBAL_KEY] * global_mask, # GLOBAL_MSG
                    t * ones
                ]
                for address_name, address_values in hyper_edges.addresses.items():
                    clean_address_values = jnp.nan_to_num(address_values, nan=MAX_INTEGER).astype(int)
                    nn_input_list.append(h[LOCAL_KEY].at[clean_address_values].get(mode='drop', fill_value=0.))

                nn_input = jnp.concatenate(nn_input_list, axis=1)

                for address_name, address_values in hyper_edges.addresses.items():
                    clean_address_values = jnp.nan_to_num(address_values, nan=MAX_INTEGER).astype(int)
                    mlp = MLP(self.hidden_size, self.out_size, get_activation(self.activation), name="{}-{}".format(hyper_edge_name, address_name)) # leaky or tanh
                    r = get_activation(self.activation)(mlp(nn_input)) # leaky_relu or tanh
                    r = jnp.where(jnp.isnan(r), jnp.nan, jnp.nan_to_num(r, nan=0.))
                    delta_sum = delta_sum.at[clean_address_values].add(r, mode='drop')
                    count += 1
        return get_activation(self.final_activation)(delta_sum) # leaky_relu or tanh


class LocalDecoder(nn.Module):
    r"""Decodes the solution of the NODE into local predictions located at hyper-edges.

    .. math::
            \forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c, \hat{y}^c_e =
            \Psi_\theta^{c} (x^g, x^c_e, h^g(t=1), (h_{o(e)}(t=1))_{o\in \mathcal{O}^c}, t)

    Attributes:
        hidden_size (:obj:`typing.Sequence` of :obj:`int`): List of sizes of hidden layers of the local decoder MLPs
            :math:`(\Psi_\theta^{c})_{c \in \mathcal{C}}`.
        out_size (:obj:`int`): Output size of the local decoder MLPs
            :math:`(\Phi_\theta^{c})_{c \in \mathcal{C}}`.
        local_output_feature_names (:obj:`dict` of :obj:`float`): Provides for every object class a list of
            features for which a prediction should be output.
    """
    hidden_size: Sequence[int]
    local_output_features_dict: dict
    activation: str
    global_msg: bool

    @nn.compact
    def __call__(self, h2mg_in: H2MG, h2mg_encoded: dict, h: dict):

        r = {}
        for hyper_edges_name, feature_list in self.local_output_features_dict.items():
            isnan_mask = jnp.isnan(h2mg_in[hyper_edges_name].array[:, 0])
            latent_variable_input_list = []
            for address_name, address_values in h2mg_in[hyper_edges_name].addresses.items():
                clean_address_values = jnp.nan_to_num(address_values, nan=MAX_INTEGER).astype(int)
                latent_variable_input_list.append(h[LOCAL_KEY].at[clean_address_values].get(mode='drop', fill_value=0.))
            ones = jnp.ones_like(h2mg_encoded[LOCAL_KEY][hyper_edges_name])[:, :1]
            if self.global_msg:
                global_mask = jnp.ones_like(h2mg_encoded[LOCAL_KEY][hyper_edges_name])[:, :1]
            else:
                global_mask = jnp.zeros_like(h2mg_encoded[LOCAL_KEY][hyper_edges_name])[:, :1]
            nn_input = jnp.concatenate([
                *latent_variable_input_list,
                h[GLOBAL_KEY] * global_mask, # GLOBAL_MSG
                h2mg_encoded[LOCAL_KEY][hyper_edges_name],
                h2mg_encoded[GLOBAL_KEY] * global_mask], axis=1) # GLOBAL_MSG
            features_dict = {}
            for feature_name in feature_list:
                mlp = MLP(self.hidden_size, 1, get_activation(self.activation), name="{}-{}".format(hyper_edges_name, feature_name))
                features_dict[feature_name] = jnp.where(isnan_mask, jnp.nan, mlp(nn_input)[:, 0])
            r[hyper_edges_name] = HyperEdges(features=features_dict)
        return r


class GlobalDynamics(nn.Module):
    r"""Computes the second term of the Neural Ordinary Differential Equation that concerns global latent variables.

    .. math::
            \frac{dh^g}{dt} = \Phi^g_\theta (x^g, h^g, \frac{1}{N}\sum_{i=1}^N h_i, t)

    Attributes:
        hidden_size (:obj:`typing.Sequence` of :obj:`int`): Sequence of sizes of hidden layers of the global dynamics
            MLPs :math:`\Phi^g_\theta`.
        out_size (:obj:`int`): Output size of the global dynamics MLPs :math:`\Phi^g_\theta`.
    """
    hidden_size: Sequence[int]
    out_size: int
    activation: str

    @nn.compact
    def __call__(self, h2mg_in: H2MG, h2mg_encoded: dict, h: dict, t: float):
        nn_input = jnp.concatenate([
            nan_sum_at(h[LOCAL_KEY], h2mg_in.all_addresses_array),
            h[GLOBAL_KEY],
            h2mg_encoded[GLOBAL_KEY],
            t * jnp.ones([1, 1])
        ], axis=1)
        r = MLP(self.hidden_size, self.out_size, get_activation(self.activation))(nn_input) # leaky_relu or tanh
        r = jnp.where(jnp.isnan(r), jnp.nan, jnp.nan_to_num(r, nan=0.))
        return r


class GlobalDecoder(nn.Module):
    r"""Decodes the solution of the NODE into a global prediction shared across the input graph.

    .. math::
        \hat{y}^g = \Psi^g_\theta (x^g, h^g(t=1), \frac{1}{N}\sum_{i=1}^N h_i(t=1), t)
    """
    hidden_size: Sequence[int]
    global_output_features_list: list
    activation: str

    @nn.compact
    def __call__(self, h2mg_in, h2mg_encoded, h):
        features_dict = {}
        nn_input = jnp.concatenate(
            [nan_mean_at(h[LOCAL_KEY], h2mg_in.all_addresses_array), h[GLOBAL_KEY], h2mg_encoded[GLOBAL_KEY]], axis=1)

        for k in self.global_output_features_list:
            mlp = MLP(self.hidden_size, 1, get_activation(self.activation), name="{}".format(k), use_bias=True)
            features_dict[k] = mlp(nn_input)[:, 0]
        return HyperEdges(features=features_dict)


class H2MGNODE(flax.struct.PyTreeNode):
    r"""Hyper Heterogeneous Multi Graph Neural Ordinary Differential Equation.

    Solves the following differential system defined over a H2MG (Hyper Heterogeneous Multi Graph):

    .. math::
        \forall i \in \{1, \dots, n\}, \frac{dh_i}{dt} = \tanh(\sum_{(c,e,o) \in \mathcal{N}(i)} \tanh
            \Phi_\theta^{c,o} (x^g, x^c_e, h^g, (h_{o(e)})_{o\in \mathcal{O}^c}, t) \\
        \frac{dh^g}{dt} = \Phi^g_\theta (x^g, h^g, \frac{1}{N}\sum_{i=1}^N h_i, t) \\
        \forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c, \hat{y}^c_e =
            \Psi_\theta^{c} (x^g, x^c_e, h^g(t=1), (h_{o(e)}(t=1))_{o\in \mathcal{O}^c}, t) \\
        \hat{y}^g = \Psi^g_\theta (x^g, h^g(t=1), \frac{1}{N}\sum_{i=1}^N h_i(t=1), t)

    Attributes:
        local_dynamics (:obj:`nn.Module`): Set of neural networks that define the dynamics of local hidden variables.
        global_dynamics (:obj:`nn.Module`): Neural network that define the dynamics of global hidden variables.
        local_decoder (:obj:`nn.Module`): Set of neural networks that decode hidden variables into local predictions.
        global_decoder (:obj:`nn.Module`): Set of neural networks that decode hidden variables into global predictions.
        local_latent_dimension (:obj:`int`): Dimension of local latent variables.
        global_latent_dimension (:obj:`int`): Dimension of global latent variables.
        solver (:obj:`diffrax.solver`): Neural Ordinary Differential Equation solver.
        dt0 (:obj:`int`): Neural Ordinary Differential Equation initial time step.
        stepsize_controller (:obj:`diffrax.step_size_controller`): Neural Ordinary Differential Equation step size
            controller.
        adjoint (:obj:`diffrax.adjoint`): Neural Ordinary Differential Equation adjoint method.
        max_steps (:obj:`int`): Neural Ordinary Differential Equation max amount of steps.
    """
    local_encoder: nn.Module = None
    global_encoder: nn.Module = None
    local_dynamics: nn.Module = None
    global_dynamics: nn.Module = None
    local_decoder: nn.Module = None
    global_decoder: nn.Module = None
    local_latent_dimension: int = 4
    global_latent_dimension: int = 4
    solver: diffrax.solver = None
    dt0: float = 0.1
    stepsize_controller: diffrax.step_size_controller = None
    adjoint: diffrax.adjoint = None
    max_steps: int = 4096

    @classmethod
    def make(cls,
             output_structure: H2MGStructure,
             local_encoder_hidden_size: Sequence[int] = None,
             global_encoder_hidden_size: Sequence[int] = None,
             local_dynamics_hidden_size: Sequence[int] = None,
             global_dynamics_hidden_size: Sequence[int] = None,
             local_decoder_hidden_size: Sequence[int] = None,
             global_decoder_hidden_size: Sequence[int] = None,
             local_encoder_output: int = 16,
             global_encoder_output: int = 16,
             local_latent_dimension: int = 16,
             global_latent_dimension: int = 16,
             solver_name: str = "Euler",
             dt0: float = 0.1,
             stepsize_controller_name: str = "ConstantStepSize",
             stepsize_controller_kwargs: dict = None,
             adjoint_name: str = "RecursiveCheckpointAdjoint",
             max_steps: int = 4096,
             enc_dec_activation: str = "tanh",
             dyn_activation: str = "tanh",
             dyn_final_activation: str = "tanh",
             global_msg: bool = True):

        if local_encoder_hidden_size is None:
            local_encoder_hidden_size = [32, 32]

        if global_encoder_hidden_size is None:
            global_encoder_hidden_size = [32, 32]

        if local_dynamics_hidden_size is None:
            local_dynamics_hidden_size = []

        if global_dynamics_hidden_size is None:
            global_dynamics_hidden_size = []

        if local_decoder_hidden_size is None:
            local_decoder_hidden_size = [32, 32]

        if global_decoder_hidden_size is None:
            global_decoder_hidden_size = [32, 32]

        if stepsize_controller_kwargs is None:
            stepsize_controller_kwargs = dict()

        local_encoder = LocalEncoder(local_encoder_hidden_size, local_encoder_output, activation=enc_dec_activation)
        global_encoder = GlobalEncoder(global_encoder_hidden_size, global_encoder_output, activation=enc_dec_activation)
        local_dynamics = LocalDynamics(local_dynamics_hidden_size, local_latent_dimension, activation=dyn_activation, final_activation=dyn_final_activation, global_msg=global_msg)
        global_dynamics = GlobalDynamics(global_dynamics_hidden_size, global_latent_dimension, activation=dyn_activation)
        if any([(he_struct.features is not None) for he_struct in output_structure.local_hyper_edges_structure.values()]):
            local_output_features_dict = {k: list(he_struct.features.keys())
                for k, he_struct in output_structure.local_hyper_edges_structure.items()
                if he_struct.features is not None}
            local_decoder = LocalDecoder(local_decoder_hidden_size, local_output_features_dict, activation=enc_dec_activation, global_msg=global_msg)

        else:
            local_decoder = None
        if output_structure.global_hyper_edges_structure is not None:
            if output_structure.global_hyper_edges_structure.features is not None:
                global_decoder = GlobalDecoder(global_decoder_hidden_size,
                                               global_output_features_list=list(output_structure.global_hyper_edges_structure.features.keys()), activation=enc_dec_activation)
        else:
            global_decoder = None
        solver = eval("diffrax.{}()".format(solver_name))
        stepsize_controller = eval("diffrax.{}".format(stepsize_controller_name))(**stepsize_controller_kwargs)
        adjoint = eval("diffrax.{}()".format(adjoint_name))

        return cls(local_encoder, global_encoder, local_dynamics, global_dynamics, local_decoder, global_decoder,
                   local_latent_dimension, global_latent_dimension, solver, dt0, stepsize_controller, adjoint,
                   max_steps)

    def init(self, rng, h2mg_in: H2MG):
        params = {}

        rng1, rng2, rng3, rng4, rng5, rng6 = jax.random.split(rng, 6)
        h = self._initialize_latent_variables(h2mg_in)

        h2mg_local_encoded, params[LOCAL_ENCODER_KEY] = self.local_encoder.init_with_output(rng1, h2mg_in)
        h2mg_global_encoded, params[GLOBAL_ENCODER_KEY] = self.global_encoder.init_with_output(rng2, h2mg_in)
        h2mg_encoded = {LOCAL_KEY: h2mg_local_encoded, GLOBAL_KEY: h2mg_global_encoded}

        params[LOCAL_DYNAMICS_KEY] = self.local_dynamics.init(rng3, h2mg_in, h2mg_encoded, h, 0.)
        params[GLOBAL_DYNAMICS_KEY] = self.global_dynamics.init(rng4, h2mg_in, h2mg_encoded, h, 0.)
        if self.local_decoder is not None:
            params[LOCAL_DECODER_KEY] = self.local_decoder.init(rng5, h2mg_in, h2mg_encoded, h)
        if self.global_decoder is not None:
            params[GLOBAL_DECODER_KEY] = self.global_decoder.init(rng6, h2mg_in, h2mg_encoded, h)

        return params

    def vmap_apply(self, params, h2mg_in, **kwargs):
        return jax.vmap(self.apply, in_axes=(None, 0))(params, h2mg_in, **kwargs)
    
    def encode(self, params, h2mg_in, **kwargs):
        h2mg_local_encoded = self.local_encoder.apply(params[LOCAL_ENCODER_KEY], h2mg_in)
        h2mg_global_encoded = self.global_encoder.apply(params[GLOBAL_ENCODER_KEY], h2mg_in)
        h2mg_encoded = {LOCAL_KEY: h2mg_local_encoded, GLOBAL_KEY: h2mg_global_encoded}
        return h2mg_encoded
    
    def vmap_encode(self, params, h2mg_in, **kwargs):
        return jax.vmap(self.encode, in_axes=(None, 0))(params, h2mg_in, **kwargs)

    def apply(self, params, h2mg_in, **kwargs):

        h2mg_local_encoded = self.local_encoder.apply(params[LOCAL_ENCODER_KEY], h2mg_in)
        h2mg_global_encoded = self.global_encoder.apply(params[GLOBAL_ENCODER_KEY], h2mg_in)
        h2mg_encoded = {LOCAL_KEY: h2mg_local_encoded, GLOBAL_KEY: h2mg_global_encoded}

        def f(t, y, args):
            h2mg_in, h2mg_encoded = args
            local_delta = self.local_dynamics.apply(params[LOCAL_DYNAMICS_KEY], h2mg_in, h2mg_encoded, y, t)
            global_delta = self.global_dynamics.apply(params[GLOBAL_DYNAMICS_KEY], h2mg_in, h2mg_encoded, y, t)
            return {GLOBAL_KEY: global_delta, LOCAL_KEY: local_delta}

        term = diffrax.ODETerm(f)
        h0 = self._initialize_latent_variables(h2mg_in)
        solution = diffrax.diffeqsolve(term,
                                       kwargs.get("solver", self.solver),
                                       t0=0,
                                       t1=1,
                                       dt0=kwargs.get("dt0", self.dt0),
                                       y0=h0,
                                       args=(h2mg_in, h2mg_encoded),
                                       stepsize_controller=kwargs.get("stepsize_controller", self.stepsize_controller),
                                       adjoint=kwargs.get("adjoint", self.adjoint),
                                       max_steps=kwargs.get("max_steps", self.max_steps))
        h1 = {GLOBAL_KEY: solution.ys[GLOBAL_KEY][0], LOCAL_KEY: solution.ys[LOCAL_KEY][0]}

        h2mg_out = H2MG()
        if self.global_decoder is not None:
            global_hyper_edges = self.global_decoder.apply(params[GLOBAL_DECODER_KEY], h2mg_in, h2mg_encoded, h1)
            h2mg_out.add_global_hyper_edges(global_hyper_edges)
        if self.local_decoder is not None:
            local_hyper_edges = self.local_decoder.apply(params[LOCAL_DECODER_KEY], h2mg_in, h2mg_encoded, h1)
            for k, hyper_edges in local_hyper_edges.items():
                h2mg_out.add_local_hyper_edges(k, hyper_edges)

        return h2mg_out

    def _initialize_latent_variables(self, h2mg_in):
        return {GLOBAL_KEY: jnp.zeros([1, self.global_latent_dimension]),
            LOCAL_KEY: jnp.zeros([jnp.shape(h2mg_in.all_addresses_array)[0], self.local_latent_dimension])}

    
    def apply_h1(self, params, h2mg_in, **kwargs):

        h2mg_local_encoded = self.local_encoder.apply(params[LOCAL_ENCODER_KEY], h2mg_in)
        h2mg_global_encoded = self.global_encoder.apply(params[GLOBAL_ENCODER_KEY], h2mg_in)
        h2mg_encoded = {LOCAL_KEY: h2mg_local_encoded, GLOBAL_KEY: h2mg_global_encoded}

        def f(t, y, args):
            h2mg_in, h2mg_encoded = args
            local_delta = self.local_dynamics.apply(params[LOCAL_DYNAMICS_KEY], h2mg_in, h2mg_encoded, y, t)
            global_delta = self.global_dynamics.apply(params[GLOBAL_DYNAMICS_KEY], h2mg_in, h2mg_encoded, y, t)
            return {GLOBAL_KEY: global_delta, LOCAL_KEY: local_delta}

        term = diffrax.ODETerm(f)
        h0 = self._initialize_latent_variables(h2mg_in)
        solution = diffrax.diffeqsolve(term,
                                       kwargs.get("solver", self.solver),
                                       t0=0,
                                       t1=1,
                                       dt0=kwargs.get("dt0", self.dt0),
                                       y0=h0,
                                       args=(h2mg_in, h2mg_encoded),
                                       stepsize_controller=kwargs.get("stepsize_controller", self.stepsize_controller),
                                       adjoint=kwargs.get("adjoint", self.adjoint),
                                       max_steps=kwargs.get("max_steps", self.max_steps))
        h1 = {GLOBAL_KEY: solution.ys[GLOBAL_KEY][0], LOCAL_KEY: solution.ys[LOCAL_KEY][0]}

        nn_input = jnp.concatenate(
            [nan_sum_at(h1[LOCAL_KEY], h2mg_in.all_addresses_array), h1[GLOBAL_KEY], h2mg_encoded[GLOBAL_KEY]], axis=1)

        return nn_input

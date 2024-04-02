from gymnasium import spaces
import gymnasium as gym
import numpy as np
from ml4ps.h2mg import H2MG, H2MGStructure, HyperEdges, HyperEdgesStructure
from copy import deepcopy
from collections import OrderedDict
from gymnasium.vector.utils.spaces import iterate


ADDRESSES = "addresses"
FEATURES = "features"


class H2MGSpace(spaces.Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    @classmethod
    def from_structure(cls, structure: H2MGStructure) -> 'H2MGSpace':
        """Builds a H2MGSpace from a H2MG structure."""
        data = cls()
        for k, he_structure in structure.items():
            data._add_hyper_edges_space(k, HyperEdgesSpace.from_structure(he_structure))
        return data

    @property
    def structure(self) -> H2MGStructure:
        """Structure of the H2MGSpace."""
        structure = H2MGStructure()
        for k, hyper_edges_space in self.spaces.items():
            structure._add_hyper_edges_structure(k, hyper_edges_space.structure)
        return structure

    def add_suffix(self, suffix: str) -> 'H2MGSpace':
        """Returns an identical H2MGSpace with `suffix` appended at the end of every feature/address name."""
        h2mg_space = H2MGSpace()
        for k, he in self.spaces.items():
            h2mg_space._add_hyper_edges_space(k, he.add_suffix(suffix))
        return h2mg_space

    def _add_hyper_edges_space(self, name: str, hyper_edges_space: 'HyperEdgesSpace'):
        """Adds a HyperEdgesSpace to a H2MGSpace."""
        self.spaces[name] = hyper_edges_space

    @property
    def high(self) -> H2MG:
        """H2MG that corresponds to the 'high' bound of the H2MGSpace."""
        h2mg = H2MG()
        for k, he in self.spaces.items():
            h2mg._add_hyper_edges(k, he.high)
        return h2mg

    @property
    def low(self) -> H2MG:
        """H2MG that corresponds to the 'low' bound of the H2MGSpace."""
        h2mg = H2MG()
        for k, he in self.spaces.items():
            h2mg._add_hyper_edges(k, he.low)
        return h2mg

    @property
    def continuous(self) -> 'H2MGSpace':
        """Returns only the continuous (i.e. Box) part of the H2MGSpace."""
        h2mg_space = H2MGSpace()
        for k, hyper_edges_space in self.spaces.items():
            # TODO: check if adding this condition is correct
            if hyper_edges_space.continuous:
                h2mg_space._add_hyper_edges_space(k, hyper_edges_space.continuous)
        return h2mg_space

    @property
    def multi_discrete(self) -> 'H2MGSpace':
        """Returns only the multi discrete part of the H2MGSpace."""
        h2mg_space = H2MGSpace()
        for k, hyper_edges_space in self.spaces.items():
            # TODO: check if adding this condition is correct
            if hyper_edges_space.multi_discrete:
                h2mg_space._add_hyper_edges_space(k, hyper_edges_space.multi_discrete)
        return h2mg_space

    @property
    def multi_binary(self) -> 'H2MGSpace':
        """Returns only the multi binary part of the H2MGSpace."""
        h2mg_space = H2MGSpace()
        for k, hyper_edges_space in self.spaces.items():
            # TODO: check if adding this condition is correct, fails if it yields an empty h2mg_space
            if hyper_edges_space.multi_binary:
                h2mg_space._add_hyper_edges_space(k, hyper_edges_space.multi_binary)
        return h2mg_space

    def combine(self, other: 'H2MGSpace') -> 'H2MGSpace':
        """Returns an H2MGSpace that is the combination of `self` and `other`.

        The output contains all HyperEdgesSpace and addresses and features contained in either one of them.
        """
        h2mg_space = H2MGSpace()
        for k in (self.spaces | other.spaces):
            if (k in self.spaces) and (k in other.spaces):
                h2mg_space._add_hyper_edges_space(k, self.spaces[k].combine(other.spaces[k]))
            elif k in self.spaces:
                h2mg_space._add_hyper_edges_space(k, self.spaces[k])
            else:
                h2mg_space._add_hyper_edges_space(k, other.spaces[k])
        return h2mg_space

    def sample(self) -> H2MG:
        data = super().sample()
        dict_ = {k: HyperEdges(**{field: {f: data[k][field][f] for f in data[k][field]} for field in data[k]}) for k in data}
        res = H2MG()
        res.update(dict_)
        return res


@gym.vector.utils.spaces.batch_space.register(H2MGSpace)
def _my_batch(space, n=1):
    return H2MGSpace(OrderedDict(
        [(key, gym.vector.utils.spaces.batch_space(subspace, n=n)) for (key, subspace) in space.spaces.items()]),
        seed=deepcopy(space.np_random), )


@gym.vector.utils.shared_memory.create_shared_memory.register(H2MGSpace)
def _create_dict_shared_memory(space, n=1, ctx=None):
    h2mg = H2MG()
    for key, subspace in space.spaces.items():
        h2mg._add_hyper_edges(key, gym.vector.utils.shared_memory.create_shared_memory(subspace, n=n, ctx=ctx))
    return h2mg


@gym.vector.utils.shared_memory.read_from_shared_memory.register(H2MGSpace)
def _my_read(space, shared_memory, n: int = 1):
    h2mg = H2MG()
    for (key, subspace) in space.spaces.items():
        h2mg._add_hyper_edges(key, gym.vector.utils.shared_memory.read_from_shared_memory(subspace,
                                                                                          shared_memory[key], n=n))
    return h2mg

@gym.vector.utils.shared_memory.write_to_shared_memory.register(H2MGSpace)
def _my_write(space: H2MGSpace, index: int, values: H2MG, shared_memory: H2MG):
    for (key, subspace) in space.spaces.items():
        gym.vector.utils.shared_memory.write_to_shared_memory(space[key], index, values[key], shared_memory[key])

@gym.vector.utils.spaces.iterate.register(H2MGSpace)
def _iterate_h2mg_space(space, items):
    keys, values = zip(*[(key, iterate(subspace, items[key])) for key, subspace in space.spaces.items()])
    for item in zip(*values):
        out = H2MG()
        for key, value in zip(keys, item):
            out._add_hyper_edges(key, value)
        yield out


class HyperEdgesSpace(spaces.Dict):
    def __init__(self, *args, features: spaces.Dict = None, addresses: spaces.Dict = None, **kwargs):
        data = {}
        if (features is not None) and features:
            data[FEATURES] = features
        if (addresses is not None) and addresses:
            data[ADDRESSES] = addresses
        super().__init__(spaces.Dict(data), *args, **kwargs)
    
    def __hash__(self):
        return hash(tuple(sorted(list(self.keys()))+[hash(self.structure)])) # TODO

    @classmethod
    def from_structure(cls, structure: HyperEdgesStructure) -> 'HyperEdgesSpace':
        """Builds a H2MGSpace from a H2MG structure."""
        if structure.addresses is not None:
            addresses_dict = spaces.Dict({f: spaces.Box(-np.inf, np.inf, shape=[v], dtype=np.float64) for f, v in structure.addresses.items()})
        else:
            addresses_dict = None
        if structure.features is not None:
            features_dict = spaces.Dict({f: spaces.Box(-np.inf, np.inf, shape=[v], dtype=np.float64) for f, v in structure.features.items()})
        else:
            features_dict = None
        return cls(addresses=addresses_dict, features=features_dict)

    @property
    def structure(self) -> H2MGStructure:
        """Structure of the HyperEdgesStructure."""
        if self.addresses is not None:
            addresses_dict = {f: v.shape[-1] for f, v in self.addresses.items()}
        else:
            addresses_dict = None
        if self.features is not None:
            features_dict = {f: v.shape[-1] for f, v in self.features.items()}
        else:
            features_dict = None
        return HyperEdgesStructure(addresses=addresses_dict, features=features_dict)

    @property
    def features(self) -> spaces.Dict:
        """Dictionary of feature spaces."""
        return self.spaces.get(FEATURES, None)

    @property
    def addresses(self) -> spaces.Dict:
        """Dictionary of address spaces."""
        return self.spaces.get(ADDRESSES, None)

    def add_suffix(self, suffix: str) -> 'HyperEdgesSpace':
        """Returns an identical HyperEdgesSpace with `suffix` appended at the end of every feature/address name."""
        if self.addresses is not None:
            addresses_dict = spaces.Dict({f + suffix: v for f, v in self.addresses.items()})
        else:
            addresses_dict = None
        if self.features is not None:
            features_dict = spaces.Dict({f + suffix: v for f, v in self.features.items()})
        else:
            features_dict = None
        return HyperEdgesSpace(addresses=addresses_dict, features=features_dict)

    def combine(self, other: 'HyperEdgesSpace') -> 'HyperEdgesSpace':
        """Returns an HyperEdgesSpace that is the combination of `self` and `other`.

        The output contains all addresses and features contained in either one of them.
        """
        addresses_dict = {}
        if (self.addresses is not None) and (other.addresses is not None):
            for k, feature_space in (self.addresses | other.addresses).items():
                if k in other.addresses:
                    addresses_dict[k] = other.addresses[k]
                else:
                    addresses_dict[k] = self.addresses[k]
        if not addresses_dict:
            addresses_dict = None

        features_dict = {}
        if (self.features is not None) and (other.features is not None):
            for k, feature_space in (self.features | other.features).items():
                if k in other.features:
                    features_dict[k] = other.features[k]
                else:
                    features_dict[k] = self.features[k]
        if not features_dict:
            features_dict = None

        return HyperEdgesSpace(addresses=spaces.Dict(addresses_dict), features=spaces.Dict(features_dict))

    @property
    def high(self) -> HyperEdges:
        """HyperEdges that corresponds to the 'high' bound of the HyperEdgesSpace."""
        if self.addresses is not None:
            addresses_dict = {f: v.high * np.ones([v.shape[-1], ]) for f, v in self.addresses.items()}
        else:
            addresses_dict = None
        if self.features is not None:
            features_dict = {f: v.high * np.ones([v.shape[-1], ]) for f, v in self.features.items()}
        else:
            features_dict = None
        return HyperEdges(addresses=addresses_dict, features=features_dict)

    @property
    def low(self) -> HyperEdges:
        """HyperEdges that corresponds to the 'low' bound of the HyperEdgesSpace."""
        if self.addresses is not None:
            addresses_dict = {f: v.low * np.ones([v.shape[-1], ]) for f, v in self.addresses.items()}
        else:
            addresses_dict = None
        if self.features is not None:
            features_dict = {f: v.low * np.ones([v.shape[-1], ]) for f, v in self.features.items()}
        else:
            features_dict = None
        return HyperEdges(addresses=addresses_dict, features=features_dict)

    @property
    def continuous(self) -> 'HyperEdgesSpace':
        """Returns only the continuous (i.e. Box) part of the HyperEdgesSpace."""
        if self.addresses is not None:
            addresses_dict = spaces.Dict({f: v for f, v in self.addresses.items() if isinstance(v, spaces.Box)})
            if not addresses_dict:
                addresses_dict = None
        else:
            addresses_dict = None
        if self.features is not None:
            features_dict = spaces.Dict({f: v for f, v in self.features.items() if isinstance(v, spaces.Box)})
            if not features_dict:
                features_dict = None
        else:
            features_dict = None
        return HyperEdgesSpace(addresses=addresses_dict, features=features_dict)

    @property
    def multi_discrete(self) -> 'HyperEdgesSpace':
        """Returns only the multi discrete part of the HyperEdgesSpace."""
        if self.addresses is not None:
            addresses_dict = spaces.Dict({f: v for f, v in self.addresses.items() if isinstance(v, spaces.MultiDiscrete)})
            if not addresses_dict:
                addresses_dict = None
        else:
            addresses_dict = None
        if self.features is not None:
            features_dict = spaces.Dict({f: v for f, v in self.features.items() if isinstance(v, spaces.MultiDiscrete)})
            if not features_dict:
                features_dict = None
        else:
            features_dict = None
        return HyperEdgesSpace(addresses=addresses_dict, features=features_dict)

    @property
    def multi_binary(self) -> 'HyperEdgesSpace':
        """Returns only the multi binary part of the HyperEdgesSpace."""
        if self.addresses is not None:
            addresses_dict = spaces.Dict({f: v for f, v in self.addresses.items() if isinstance(v, spaces.MultiBinary)})
            if not addresses_dict:
                addresses_dict = None
        else:
            addresses_dict = None
        if self.features is not None:
            features_dict = spaces.Dict({f: v for f, v in self.features.items() if isinstance(v, spaces.MultiBinary)})
            if not features_dict:
                features_dict = None
        else:
            features_dict = None
        return HyperEdgesSpace(addresses=addresses_dict, features=features_dict)


@gym.vector.utils.spaces.batch_space.register(HyperEdgesSpace)
def _my_batch(space, n=1):
    if space.addresses is not None:
        addresses_dict = gym.vector.utils.spaces.batch_space(spaces.Dict(space.addresses), n=n)
    else:
        addresses_dict = None
    if space.features is not None:
        features_dict = gym.vector.utils.spaces.batch_space(spaces.Dict(space.features), n=n)
    else:
        features_dict = None
    return HyperEdgesSpace(addresses=addresses_dict, features=features_dict)


@gym.vector.utils.shared_memory.create_shared_memory.register(HyperEdgesSpace)
def _create_dict_shared_memory(space: HyperEdgesSpace, n=1, ctx=None):
    if space.addresses is not None:
        addresses_dict = gym.vector.utils.shared_memory.create_shared_memory(space.addresses, n=n, ctx=ctx)
    else:
        addresses_dict = None
    if space.features is not None:
        features_dict = gym.vector.utils.shared_memory.create_shared_memory(space.features, n=n, ctx=ctx)
    else:
        features_dict = None
    return HyperEdges(addresses=addresses_dict, features=features_dict)


@gym.vector.utils.shared_memory.read_from_shared_memory.register(HyperEdgesSpace)
def _my_read(space: HyperEdgesSpace, shared_memory: HyperEdges, n: int = 1):
    if shared_memory.addresses is not None:
        addresses_dict = gym.vector.utils.shared_memory.read_from_shared_memory(space.addresses, shared_memory.addresses, n=n)
    else:
        addresses_dict = None
    if shared_memory.features is not None:
        features_dict = gym.vector.utils.shared_memory.read_from_shared_memory(space.features, shared_memory.features, n=n)
    else:
        features_dict = None
    return HyperEdges(addresses=addresses_dict, features=features_dict)


@gym.vector.utils.spaces.iterate.register(HyperEdgesSpace)
def _iterate_hyper_edges_space(space, items):
    keys, values = zip(*[(key, iterate(subspace, items[key])) for key, subspace in space.spaces.items()])
    for item in zip(*values):
        dict_ = OrderedDict([(key, value) for (key, value) in zip(keys, item)])
        features_dict = None
        if FEATURES in dict_.keys():
            features_dict = dict_[FEATURES]
        addresses_dict = None
        if ADDRESSES in dict_.keys():
            addresses_dict = dict_[ADDRESSES]
        yield HyperEdges(addresses=addresses_dict, features=features_dict)


@gym.vector.utils.shared_memory.write_to_shared_memory.register(HyperEdgesSpace)
def _my_write(space: HyperEdgesSpace, index: int, values: HyperEdges, shared_memory: HyperEdges):
    if values.addresses is not None:
        gym.vector.utils.shared_memory.write_to_shared_memory(space.addresses, index, values.addresses, shared_memory.addresses)
    if values.features is not None:
        gym.vector.utils.shared_memory.write_to_shared_memory(space.features, index, values.features, shared_memory.features)


# TODO
@gym.vector.utils.concatenate.register(H2MGSpace)
def _concatenate_dict(space, items, out):
    return H2MG(
        [
            (key, gym.vector.utils.concatenate(subspace, [item[key] for item in items], out[key]))
            for (key, subspace) in space.spaces.items()
        ]
    )

# TODO
@gym.vector.utils.concatenate.register(HyperEdgesSpace)
def _concatenate_dict(space, items, out):
    _d= OrderedDict(
        [
            (key, gym.vector.utils.concatenate(subspace, [item[key] for item in items], out[key]))
            for (key, subspace) in space.spaces.items()
        ]
    )
    return HyperEdges(features=_d.get(FEATURES, None), addresses=_d.get(ADDRESSES, None))

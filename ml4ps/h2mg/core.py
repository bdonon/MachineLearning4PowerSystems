import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class
from typing import Dict, Any

from ml4ps.h2mg.hyper_edges import HyperEdges, collate_hyper_edges, separate_hyper_edges, stack_hyper_edges
from typing import Callable


GLOBAL = "global"
ALL_ADDRESSES = "all_addresses"


@register_pytree_node_class
class H2MG(dict):
    """Hyper Heterogeneous Multi Graph (H2MG).

    It contains a series of local hyper-edges (local quantities, e.g. buses, generators, lines, etc.), a single global
    hyper-edge (global quantities, e.g. power frequency), and a 'all_addresses' hyper-edges which contains all the
    addresses contained in the said H2MG (this carries no additional information compared to the other hyper-edges,
    but are necessary for the functioning of the H2MGNODE architecture).

    By default, when building an H2MG, the addresses are represented by `str`. Calling the method `convert_str_to_int`
    allows to convert addresses into integers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)

    def __str__(self):
        def build_str(h2mg):
            r = ""
            for k, hyper_edges in h2mg.items():
                r += k + "\n" + str(hyper_edges) + "\n" + "\n"
            return r
        return build_str(self)

    def tree_flatten(self):
        """Flattens a PyTree, required for JAX compatibility."""
        children = self.values()
        aux = self.keys()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflattens a PyTree, required for JAX compatibility."""
        h2mg = H2MG()
        for k, v in zip(aux_data, children):
            h2mg._add_hyper_edges(k, v)
        return h2mg

    @classmethod
    def from_structure(cls, structure: 'H2MGStructure', value: float = 0.):
        """Creates a H2MG filled with `value` that has the structure specified in `structure`."""
        h2mg = cls()
        for k, v in structure.items():
            h2mg._add_hyper_edges(k, HyperEdges.from_structure(v, value=value))
        return h2mg

    def add_local_hyper_edges(self, name: str, hyper_edges: HyperEdges):
        """Adds a set of local hyper-edges to a H2MG under the `name` key."""
        assert name not in {GLOBAL, ALL_ADDRESSES}
        assert name not in self.keys()
        self[name] = hyper_edges

    def add_global_hyper_edges(self, hyper_edges: HyperEdges):
        """Adds global features defined as a set of hyper-edges."""
        self[GLOBAL] = hyper_edges

    def _add_hyper_edges(self, name: str, hyper_edges: HyperEdges):
        """Adds a set of hyper-edges, either local, global or all_addresses."""
        self[name] = hyper_edges

    @property
    def hyper_edges(self) -> dict:
        return self

    @property
    def local_hyper_edges(self) -> dict[str, HyperEdges]:
        """Dictionary of local hyper-edges."""
        return {k: self[k] for k in self if k not in {GLOBAL, ALL_ADDRESSES}}

    @property
    def global_hyper_edges(self) -> HyperEdges:
        """Hyper-edges containing all the global features of the H2MG."""
        return self.get(GLOBAL, None)

    @property
    def all_addresses_hyper_edges(self) -> HyperEdges:
        """Hyper-edges of the H2MG containing all the addresses used in the said H2MG."""
        return self.get(ALL_ADDRESSES, None)

    @property
    def all_addresses_array(self) -> np.array:
        """Array containing all the addresses used in an H2MG."""
        if ALL_ADDRESSES in self:
            return self[ALL_ADDRESSES].addresses["id"]
        else:
            return None

    @property
    def flat_array(self) -> np.array:
        """Array made of the concatenation of all numerical features contained in an H2MG."""
        return jnp.concatenate([self[k].flat_array for k in sorted(self) if self[k].flat_array is not None], axis=-1)

    @flat_array.setter
    def flat_array(self, value: np.array):
        lengths = [self[k].flat_array.shape[-1] for k in sorted(self) if self[k].flat_array is not None]
        split_indices = [sum(lengths[:i]) for i in range(1, len(lengths))]
        new_values = jnp.split(value, split_indices, axis=-1)
        for k_, v in zip([k for k in sorted(self) if self[k].flat_array is not None], new_values):
            self[k_].flat_array = v

    @property
    def structure(self) -> 'H2MGStructure':
        """Dictionary containing the structure of each Hyper-Edges."""
        _structure = H2MGStructure()
        for name, he in self.local_hyper_edges.items():
            _structure.add_local_hyper_edges_structure(name, he.structure)
        if self.global_hyper_edges is not None:
            _structure.add_global_hyper_edges_structure(self.global_hyper_edges.structure)
        if self.all_addresses_hyper_edges is not None:
            _structure.add_all_addresses_structure(self.all_addresses_hyper_edges.structure)
        return _structure

    def convert_str_to_int(self) -> dict:
        """Converts the addresses of hyper-edges contained in an H2MG into integer. Also returns int_to_str dict."""
        all_addresses = []
        for hyper_edges_name, hyper_edges in self.local_hyper_edges.items():
            if hyper_edges.addresses is not None:
                for address_name, values in hyper_edges.addresses.items():
                    all_addresses.append(values)
        if all_addresses:
            all_addresses = list(np.unique(np.concatenate(all_addresses)))
            str_to_int = {address: np.array(i) for i, address in enumerate(all_addresses)}
            converter = np.vectorize(str_to_int.get)

            for hyper_edges_name in self.local_hyper_edges:
                self[hyper_edges_name].addresses = {address_name: np.array(converter(self[hyper_edges_name].addresses[address_name])) for address_name in self.local_hyper_edges[hyper_edges_name].addresses}
            self[ALL_ADDRESSES] = HyperEdges(addresses={"id": np.array(converter(all_addresses))})
        return

    def add_suffix(self, suffix: str) -> None:
        """Modifies a H2MG by adding a suffix to all features."""
        for hyper_edges in self.values():
            hyper_edges.add_suffix(suffix)
    
    def apply(self, fn: Callable):
        for hyper_edges in self.values():
            hyper_edges.apply(fn)
        return self
    
    def as_numpy(self):
        data = dict()
        for k, hyper_edges in self.items():
            data[k] = hyper_edges.as_numpy()
        return H2MG(data)


    def combine(self, other: 'H2MG') -> 'H2MG':
        """Returns an H2MG that is the combination of `self` and `other`.

        The output contains all hyper-edges and addresses and features contained in either one of them.
        """
        h2mg = H2MG()
        for k in (self | other):
            if (k in self) and (k in other):
                h2mg._add_hyper_edges(k, self[k].combine(other[k]))
            elif k in self:
                h2mg._add_hyper_edges(k, self[k])
            else:
                h2mg._add_hyper_edges(k, other[k])
        return h2mg

    def extract_from_structure(self, structure: 'H2MGStructure') -> 'H2MG':
        """Keeps only the parts of the H2MG that are compliant with the provided `structure`."""
        h2mg = H2MG()
        for k, hyper_edge_structure in structure.items():
            if k not in self:
                raise ValueError("Object class {} not available in current H2MG".format(k))
            else:
                h2mg._add_hyper_edges(k, self[k].extract_from_structure(hyper_edge_structure))
        return h2mg

    def pad_with_nans(self, structure: 'H2MGStructure') -> 'H2MG':
        """Appends a series of fictitious objects (represented by NaNs) to match `structure`."""
        h2mg = H2MG()
        for k in structure.hyper_edges_structure:
            if k in self.hyper_edges:
                h2mg._add_hyper_edges(k, self[k].pad_with_nans(structure[k]))
            else:
                h2mg._add_hyper_edges(k, HyperEdges.from_structure(structure[k], value=np.nan))
        return h2mg

    def unpad_nans(self) -> 'H2MG':
        """Deletes fictitious objects in each Hyper-Edges."""
        h2mg = H2MG()
        for k in self.hyper_edges:
            hyper_edges = self[k].unpad_nans()
            if not hyper_edges.is_empty():
                h2mg._add_hyper_edges(k, hyper_edges)
        return h2mg

    def plot(self):
        """Plots an H2MG."""
        raise NotImplementedError


@register_pytree_node_class
class H2MGStructure(dict):
    """Hyper Heterogeneous Multi Graph Structure.

    Represents the structure of an H2MG, by only representing the keys and the features/addresses sizes.
    """

    def __init__(self):
        super().__init__({})
    
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def tree_flatten(self):
        """Flattens a PyTree, required for JAX compatibility."""
        children = self.values()
        aux = self.keys()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflattens a PyTree, required for JAX compatibility."""
        h2mg_structure = H2MGStructure()
        for k, v in zip(aux_data, children):
            h2mg_structure._add_hyper_edges_structure(k, v)
        return h2mg_structure

    def _add_hyper_edges_structure(self, name, structure):
        """Adds a hyper-edges structure under the key `name`."""
        self[name] = structure

    def add_local_hyper_edges_structure(self, name, structure):
        """Adds a local hyper-edges structure under the key `name`."""
        assert name not in {GLOBAL, ALL_ADDRESSES}
        self[name] = structure

    def add_global_hyper_edges_structure(self, structure):
        """Adds a global hyper-edges structure."""
        self[GLOBAL] = structure

    def add_all_addresses_structure(self, structure):
        """Adds a all addresses hyper-edges structure."""
        self[ALL_ADDRESSES] = structure

    @property
    def hyper_edges_structure(self):
        """Dictionary containing all hyper-edges structures and their associated keys."""
        return self

    @property
    def local_hyper_edges_structure(self):
        """Dictionary of local hyper-edges structure and their associated keys."""
        return {k: self[k] for k in self.keys() if k not in {GLOBAL, ALL_ADDRESSES}}

    @property
    def global_hyper_edges_structure(self):
        """Global hyper-edges structure."""
        return self.get(GLOBAL, None)

    @property
    def all_addresses_structure(self):
        """All addresses hyper-edges structure."""
        return self.get(ALL_ADDRESSES, None)

    def max(self, other: 'H2MGStructure') -> 'H2MGStructure':
        """Compares a H2MGStructure with another and returns the largest structure with the largest leaf values."""
        max_structure = H2MGStructure()
        for k in self | other:
            if (k in self) and (k in other):
                max_structure._add_hyper_edges_structure(k, self[k].max(other[k]))
            elif k in self:
                max_structure._add_hyper_edges_structure(k, self[k])
            else:
                max_structure._add_hyper_edges_structure(k, other[k])
        return max_structure

    def combine(self, other: 'H2MGStructure') -> 'H2MGStructure':
        """Returns an H2MGStructure that is the combination of `self` and `other`.

        The output contains all hyper-edges and addresses and features contained in either one of them.
        """
        h2mg_structure = H2MGStructure()
        for k in (self | other):
            if (k in self) and (k in other):
                h2mg_structure._add_hyper_edges_structure(k, self[k].combine(other[k]))
            elif k in self:
                h2mg_structure._add_hyper_edges_structure(k, self[k])
            else:
                h2mg_structure._add_hyper_edges_structure(k, other[k])
        return h2mg_structure


def collate_h2mgs(h2mgs_list):
    """Collates together a list of H2MGs by collating Hyper-Edges together."""
    r = H2MG()
    for k, hyper_edges in h2mgs_list[0].items():
        r[k] = collate_hyper_edges([h2mg[k] for h2mg in h2mgs_list])
    return r

def stack_h2mgs(h2mgs_list):
    """Collates together a list of H2MGs by collating Hyper-Edges together."""
    r = H2MG()
    for k, hyper_edges in h2mgs_list[0].items():
        r[k] = stack_hyper_edges([h2mg[k] for h2mg in h2mgs_list])
    return r


def separate_h2mgs(h2mg_batch):
    """Separates a batch of collated H2MGs into a list of H2MGs."""
    remaining, i, r = True, 0, []
    while remaining:
        try:
            h2mg = H2MG()
            for k in h2mg_batch:
                h2mg[k] = separate_hyper_edges(h2mg_batch[k])[i]
            r.append(h2mg)
            i += 1
        except:
            remaining = False
    return r

def shallow_repr(h2mg: H2MG) -> Dict[str, Any]:
    results = {}
    for obj_name, hyper_edge in h2mg.hyper_edges.items():
        for feature_name, feature_value in hyper_edge.features.items():
            results[obj_name + "_" + feature_name] = feature_value
    return results
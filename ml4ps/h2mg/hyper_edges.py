import numpy as np
import jax.numpy as jnp
import pandas as pd
from jax.tree_util import register_pytree_node_class
from typing import Callable


FEATURES = "features"
ADDRESSES = "addresses"


@register_pytree_node_class
class HyperEdges(dict):
    """Hyper Edges, represents a group of multiple hyper edges of the same class stacked together.

    Hyper-edges can have addresses and/or features.
    """

    def __init__(self, features: dict = None, addresses: dict = None):
        data = {}
        if (features is not None) and features:
            data[FEATURES] = features
        if (addresses is not None) and addresses:
            data[ADDRESSES] = addresses
        super().__init__(data)

    def __str__(self):
        if self.addresses is None and self.features is not None:
            return pd.DataFrame(self.features, index=range(len(list(self.features.values())[0]))).__str__()
        elif self.features is None:
            return pd.DataFrame(self.addresses).__str__()
        else:
            return pd.DataFrame(self.addresses | self.features).__str__()

    def tree_flatten(self):
        """Flattens a PyTree, required for JAX compatibility."""
        children = self.values()
        aux = self.keys()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflattens a PyTree, required for JAX compatibility."""
        return HyperEdges(**{k: f for k, f in zip(aux_data, children)})
    
    def apply(self, fn: Callable):
        self.features = {k: fn(v) for k, v in self.features.items()}
    
    def as_numpy(self):
        features = self.features
        if features is not None:
            features = {k: np.asarray(v) for k,v in features.items()}
        addresses = self.addresses
        if addresses is not None:
            addresses = {k: np.asarray(v) for k,v in addresses.items()}
        return HyperEdges(features=features, addresses=addresses)

    @classmethod
    def from_structure(cls, structure: 'HyperEdgesStructure', value: float = 0.) -> 'HyperEdges':
        """Builds a set of hyper-edges filled with `value`, following the provided `structure`."""
        if structure.features is not None:
            features = {f: value * jnp.ones(v) for f, v in structure.features.items()}
        else:
            features = None

        if structure.addresses is not None:
            addresses = {f: value * jnp.ones(v) for f, v in structure.addresses.items()}
        else:
            addresses = None
        return cls(features=features, addresses=addresses)

    def is_empty(self) -> bool:
        """Checks whether a set of hyper-edges is empty."""
        return not bool(self)

    @property
    def features(self) -> dict:
        """Dictionary of numerical features and their associated keys."""
        return self.get(FEATURES, None)

    @features.setter
    def features(self, value: dict):
        self[FEATURES] = value

    @property
    def addresses(self) -> dict:
        """Dictionary of addresses and their associated keys."""
        return self.get(ADDRESSES, None)

    @addresses.setter
    def addresses(self, value: dict):
        self[ADDRESSES] = value

    @property
    def structure(self) -> 'HyperEdgesStructure':
        """Dictionary of shapes for all features and addresses."""
        if self.addresses is not None:
            addresses_structure = {k: v.shape[-1] for k, v in self.addresses.items()}
        else:
            addresses_structure = None

        if self.features is not None:
            features_structure = {k: v.shape[-1] for k, v in self.features.items()}
        else:
            features_structure = None

        return HyperEdgesStructure(addresses=addresses_structure, features=features_structure)

    @property
    def array(self) -> np.array:
        """Returns an array by stacking features together along the last dimension."""
        if self.features:
            return jnp.stack([self.features[k] for k in sorted(self.features)], axis=-1)
        else:
            return None

    @array.setter
    def array(self, value: np.array):
        if self.features:
            i = 0
            for key in sorted(self.features.keys()):
                self.features[key] = value[..., i]  # Slice over last axis
                i += 1
        else:
            pass
        

    @property
    def flat_array(self) -> np.array:
        """Returns a flat array by concatenating all features together."""
        if self.features:
            return jnp.concatenate([self.features[k] for k in sorted(self.features)], axis=-1)
        else:
            return None

    @flat_array.setter
    def flat_array(self, value: np.array):
        if self.features:
            new_values = jnp.split(value, len(self.features), axis=-1)
            self.features = {k: v for k, v in zip(sorted(self.features), new_values)}
        else:
            pass

    def add_suffix(self, suffix: str):
        """Modifies a HyperEdges by appending a suffix at the end of every feature and address key."""
        feature_keys = list(self.features.keys())
        for k in feature_keys:
            self.features[k+suffix] = self.features.pop(k)
        address_keys = list(self.addresses.keys()) if self.addresses else []
        for k in address_keys:
            self.addresses[k+suffix] = self.addresses.pop(k)

    def combine(self, other: 'HyperEdges') -> 'HyperEdges':
        """Returns an HyperEdges that is the combination of `self` and `other`.

        The output contains all addresses and features contained in either one of them.
        """
        addresses_dict = {}
        if (self.addresses is not None) and (other.addresses is not None):
            for k in (self.addresses | other.addresses):
                if k in other.addresses:
                    addresses_dict[k] = other.addresses[k]
                else:
                    addresses_dict[k] = self.addresses[k]
        if not addresses_dict:
            addresses_dict = None

        features_dict = {}
        if (self.features is not None) and (other.features is not None):
            for k in (self.features | other.features):
                if k in other.features:
                    features_dict[k] = other.features[k]
                else:
                    features_dict[k] = self.features[k]
        elif self.features is not None:
            for k in self.features:
                features_dict[k] = self.features[k]
        elif other.features is not None:
            for k in other.features:
                features_dict[k] = other.features[k]

        if not features_dict:
            features_dict = None

        return HyperEdges(addresses=addresses_dict, features=features_dict)

    def extract_from_structure(self, structure: 'HyperEdgesStructure') -> 'HyperEdges':
        """Keeps only the parts of the hyper-edges that are compliant with the provided `structure`."""
        if structure.features is not None:
            assert self.features is not None
            features_dict = {}
            for k, hyper_edge_structure in structure[FEATURES].items():
                if k not in self.features:
                    raise ValueError("Feature {} not available in current hyper-edge.".format(k))
                else:
                    features_dict[k] = self.features[k]
        else:
            features_dict = None 

        if structure.addresses is not None:
            assert self.addresses is not None
            addresses_dict = {}
            for k, hyper_edge_structure in structure.addresses.items():
                if k not in self.addresses:
                    raise ValueError("Address {} not available in current hyper-edge.".format(k))
                else:
                    addresses_dict[k] = self.addresses[k]
        else:
            addresses_dict = None
        return HyperEdges(features=features_dict, addresses=addresses_dict)

    def pad_with_nans(self, structure: 'HyperEdgesStructure'):
        """Appends a series of fictitious objects (represented by NaNs) in order to respect `structure`.

        Warning : Does not keep features and addresses that are not specified in `structure`.
        """
        addresses_dict = None
        if structure.addresses is not None:
            addresses_dict = {}
            for name, n in structure.addresses.items():
                if (self.addresses is not None) and (name in self.addresses):
                    old_array = self.addresses.get(name)
                    new_array = np.concatenate([old_array, np.full((n - old_array.shape[0],), np.nan)])
                else:
                    new_array = np.full([n], np.nan)
                addresses_dict[name] = new_array

        features_dict = None
        if structure.features is not None:
            features_dict = {}
            for name, n in structure.features.items():
                if (self.features is not None) and (name in self.features):
                    old_array = self.features.get(name)
                    new_array = np.concatenate([old_array, np.full((n - old_array.shape[0],), np.nan)])
                else:
                    new_array = np.full([n], np.nan)
                features_dict[name] = new_array

        return HyperEdges(addresses=addresses_dict, features=features_dict)

    def unpad_nans(self) -> 'HyperEdges':
        """Deletes fictitious objects represented by NaNs."""
        addresses_dict = None
        if self.addresses is not None:
            addresses_dict = {}
            for name, old_array in self.addresses.items():
                new_array = old_array[~np.isnan(old_array)]
                if new_array.size > 0:
                    addresses_dict[name] = new_array
            if not addresses_dict:
                addresses_dict = None

        features_dict = None
        if self.features is not None:
            features_dict = {}
            for name, old_array in self.features.items():
                new_array = old_array[~np.isnan(old_array)]
                if new_array.size > 0:
                    features_dict[name] = new_array
            if not features_dict:
                features_dict = None

        return HyperEdges(features=features_dict, addresses=addresses_dict)


@register_pytree_node_class
class HyperEdgesStructure(dict):
    """Hyper Edges Structure.

    Details the address and feature structures of an HyperEdges object.
    """

    def __init__(self, addresses: dict | list = None, features: dict | list = None):
        data = {}
        if isinstance(addresses, list):
            data[ADDRESSES] = {k: None for k in addresses}
        elif isinstance(addresses, dict):
            data[ADDRESSES] = addresses

        if isinstance(features, list):
            data[FEATURES] = {k: None for k in features}
        if isinstance(features, dict):
            data[FEATURES] = features

        super().__init__(data)
    
    def __hash__(self):
        addresses = self.addresses or {}
        features = self.features or {}
        return hash(tuple(sorted(list(addresses.items())+list(features.items()))))

    def tree_flatten(self):
        """Flattens a PyTree, required for JAX compatibility."""
        children = self.values()
        aux = self.keys()
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflattens a PyTree, required for JAX compatibility."""
        dict_ = {k: f for k, f in zip(aux_data, children)}
        if ADDRESSES in dict_:
            addresses_dict = dict_[ADDRESSES]
        else:
            addresses_dict = None

        if FEATURES in dict_:
            features_dict = dict_[FEATURES]
        else:
            features_dict = None

        return HyperEdgesStructure(addresses=addresses_dict, features=features_dict)

    @property
    def addresses(self) -> dict:
        """Dictionary of addresses shapes and their associated keys."""
        return self.get(ADDRESSES, None)

    @property
    def features(self) -> dict:
        """Dictionary of features shapes and their associated keys."""
        return self.get(FEATURES, None)

    def max(self, other: 'HyperEdgesStructure') -> 'HyperEdgesStructure':
        """Compares a HyperEdgesStructure with another, returns the largest structure with the largest leaf values."""
        if (self.addresses is not None) or (other.addresses is not None):
            addresses_dict = {k: max(self.addresses.get(k, 0), other.addresses.get(k, 0))
                for k in self.addresses | other.addresses}
        elif self.addresses is not None:
            addresses_dict = self.addresses
        elif other.addresses is not None:
            addresses_dict = other.addresses
        else:
            addresses_dict = None

        if (self.features is not None) or (other.features is not None):
            features_dict = {k: max(self.features.get(k, 0), other.features.get(k, 0))
                for k in self.features | other.features}
        elif self.features is not None:
            features_dict = self.features
        elif other.features is not None:
            features_dict = other.features
        else:
            features_dict = None

        return HyperEdgesStructure(addresses=addresses_dict, features=features_dict)

    def combine(self, other: 'HyperEdgesStructure') -> 'HyperEdgesStructure':
        """Returns an HyperEdgesStructure that is the combination of `self` and `other`.

        The output contains all addresses and features contained in either one of them.
        """
        addresses_dict = {}
        if (self.addresses is not None) and (other.addresses is not None):
            for k in (self.addresses | other.addresses):
                if k in other.addresses:
                    addresses_dict[k] = other.addresses[k]
                else:
                    addresses_dict[k] = self.addresses[k]
        if not addresses_dict:
            addresses_dict = None

        features_dict = {}
        if (self.features is not None) and (other.features is not None):
            for k in (self.features | other.features):
                if k in other.features:
                    features_dict[k] = other.features[k]
                else:
                    features_dict[k] = self.features[k]
        elif self.features is not None:
            for k in self.features:
                features_dict[k] = self.features[k]
        elif other.features is not None:
            for k in other.features:
                features_dict[k] = other.features[k]
        if not features_dict:
            features_dict = None

        return HyperEdgesStructure(addresses=addresses_dict, features=features_dict)


def collate_hyper_edges(hyper_edges_list: list) -> HyperEdges:
    """Collates together a list of Hyper Edges by batching addresses and features along the 0-th axis."""
    features_dict = None
    if hyper_edges_list[0].features is not None:
        features_dict = {}
        for k in sorted(hyper_edges_list[0].features):
            features_dict[k] = np.stack([he.features[k] for he in hyper_edges_list], axis=0)

    addresses_dict = None
    if hyper_edges_list[0].addresses is not None:
        addresses_dict = {}
        for k in sorted(hyper_edges_list[0].addresses):
            addresses_dict[k] = np.stack([he.addresses[k] for he in hyper_edges_list], axis=0)

    return HyperEdges(features=features_dict, addresses=addresses_dict)



def stack_hyper_edges(hyper_edges_list: list) -> HyperEdges:
    """Stack together a list of Hyper Edges by batching addresses and features along the last axis."""
    features_dict = None
    if hyper_edges_list[0].features is not None:
        features_dict = {}
        for k in sorted(hyper_edges_list[0].features):
            features_dict[k] = np.stack([he.features[k] for he in hyper_edges_list], axis=1)

    addresses_dict = None
    if hyper_edges_list[0].addresses is not None:
        addresses_dict = {}
        for k in sorted(hyper_edges_list[0].addresses):
            addresses_dict[k] = np.stack([he.addresses[k] for he in hyper_edges_list], axis=1)

    return HyperEdges(features=features_dict, addresses=addresses_dict)


def separate_hyper_edges(hyper_edges_batch: HyperEdges) -> list:
    """Separates a batch of collated Hyper Edges into a list of Hyper Edges."""
    remaining = True
    i = 0
    r = []
    while remaining:
        try:
            features_dict = None
            if hyper_edges_batch.features is not None:
                features_dict = {k: hyper_edges_batch.features[k][i] for k in sorted(hyper_edges_batch.features)}

            addresses_dict = None
            if hyper_edges_batch.addresses is not None:
                addresses_dict = {k: hyper_edges_batch.addresses[k][i] for k in sorted(hyper_edges_batch.addresses)}

            r.append(HyperEdges(addresses=addresses_dict, features=features_dict))

            i += 1
        except:
            remaining = False
    return r

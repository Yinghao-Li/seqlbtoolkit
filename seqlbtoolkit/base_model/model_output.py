import copy
import torch
import numpy as np
from functools import cached_property
from typing import List, Optional, Union, Any


class ModelOutput:
    """
    Sequence labeling metrics

    This class is designed to facilitate easy inheritance.
    You can conveniently add new metrics into the class by reloading the `__init__` function.
    Other member functions should work fine with new metrics.

    all class members should be torch.Tensor
    """
    def __init__(self):
        pass

    def keys(self, idx: Optional[int] = None) -> Union[List[str], str]:
        if idx is None:
            return list(self.__dict__.keys())
        elif isinstance(idx, int):
            return list(self.__dict__.keys())[idx]
        else:
            raise TypeError(f'Unsupported index type: {type(idx)}!')

    def items(self, idx: Optional[int] = None):
        if idx is None:
            for k in self.keys():
                yield k, self[k]
        elif isinstance(idx, int):
            for k in self.keys():
                yield k, self[k][idx]
        else:
            raise TypeError(f'Unsupported index type: {type(idx)}!')

    def values(self, idx: Optional[int] = None):
        if idx is None:
            for k in self.keys():
                yield self[k]
        elif isinstance(idx, int):
            for k in self.keys():
                yield self[k][idx]
        else:
            raise TypeError(f'Unsupported index type: {type(idx)}!')

    @cached_property
    def array_attribs(self):
        """
        Get all tensor members
        """
        ks = self.keys()
        array_ks = list()
        for k in ks:
            if isinstance(getattr(self, k), (torch.Tensor, np.ndarray)):
                array_ks.append(k)
        return array_ks

    @property
    def is_empty(self):
        """
        Check if the instance has any tensor member
        """
        return False if self.array_attribs else True

    def unhook(self):
        """
        Detach the results from training graph and move in to cpu
        """
        for attrib in self.array_attribs:
            detached = getattr(self, attrib).detach().clone().cpu()
            setattr(self, attrib, detached)
        return self

    def numpy(self):
        """
        Convert the Tensor results to numpy.array

        Returns
        -------
        numpy array
        """
        for attrib in self.array_attribs:
            try:
                detached = getattr(self, attrib).numpy()
                setattr(self, attrib, detached)
            except:
                pass
        return self

    def to_dict(self):
        """
        Convert class to dictionary
        """
        return {attrib: getattr(self, attrib) for attrib in self.array_attribs}

    def __str__(self):
        return str(self.to_dict())

    def __repr__(self):
        return str(self.to_dict())

    def __len__(self):
        return len(self.array_attribs)

    def __iter__(self):
        for k in self.array_attribs:
            yield self[k]

    def __getitem__(self, item: Union[int, str]):
        if isinstance(item, str):
            return getattr(self, item)
        elif isinstance(item, int):
            return getattr(self, self.keys(item))
        else:
            raise KeyError(f'Unsupported index type: {type(item)}!')

    def __setitem__(self, key: Union[int, str], value: Any):
        if isinstance(key, str):
            delattr(self, "tensor_attribs")
            return setattr(self, key, value)
        elif isinstance(key, int):
            delattr(self, "tensor_attribs")
            return getattr(self, self.keys(key), value)
        else:
            raise KeyError(f'Unsupported index type: {type(key)}!')

    def __eq__(self, other: Any):
        if isinstance(other, type(self)):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    def __add__(self, other: "ModelOutput"):
        if self.is_empty:
            return other
        new_inst = copy.deepcopy(self)
        for attrib in self.array_attribs:
            nts = torch.cat((getattr(new_inst, attrib), getattr(other, attrib)), dim=0)
            setattr(new_inst, attrib, nts)
        return new_inst

    def __iadd__(self, other: "ModelOutput"):
        if self.is_empty:
            return other
        for attrib in self.array_attribs:
            nts = torch.cat((getattr(self, attrib), getattr(other, attrib)), dim=0)
            setattr(self, attrib, nts)
        return self

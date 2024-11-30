import torch
from typing import Optional


class Batch(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tensor_keys = list()
        for k, v in self.items():
            self.register_tensor_members(k, v)

    def register_tensor_members(self, k, v):
        if isinstance(v, torch.Tensor) or callable(getattr(v, "to", None)):
            self._tensor_keys.append(k)

    def to(self, device):
        for k in self._tensor_keys:
            self[k] = self[k].to(device)
        return self

    @property
    def tensors(self):
        return {k: self[k] for k in self._tensor_keys}

    def __len__(self):
        return len(self[self._tensor_keys[0]]) if self._tensor_keys else 0

    def __getattr__(self, name):
        if name in self:
            return self[name]
        elif name.startswith("_"):
            return super().__getattribute__(name)
        else:
            raise AttributeError(f"'Batch' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value
            self.register_tensor_members(name, value)


def pack_instances(**kwargs) -> list[dict]:
    """
    Convert attribute lists to a list of data instances, each is a dict with attribute names as keys
    and one datapoint attribute values as values
    """

    instance_list = list()
    keys = tuple(kwargs.keys())

    for inst_attrs in zip(*tuple(kwargs.values())):
        inst = dict(zip(keys, inst_attrs))
        instance_list.append(inst)

    return instance_list


def unpack_instances(instance_list: list[dict], attr_names: Optional[list[str]] = None):
    """
    Convert a list of dict-type instances to a list of value lists,
    each contains all values within a batch of each attribute

    Parameters
    ----------
    instance_list: a list of attributes
    attr_names: the name of the needed attributes. Notice that this variable should be specified
        for Python versions that does not natively support ordered dict
    """
    if not attr_names:
        attr_names = list(instance_list[0].keys())
    attribute_tuple = [[inst[name] for inst in instance_list] for name in attr_names]

    return attribute_tuple

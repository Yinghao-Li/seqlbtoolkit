from seqeval import metrics
from seqeval.scheme import IOB2
from typing import Optional, Union, Any

from ..utils import all_equal


class Metric:
    """
    Sequence labeling metrics

    This class is designed to facilitate easy inheritance.
    You can conveniently add new metrics into the class by reloading the `__init__` function.
    Other member functions should work fine with new metrics.

    Notice that one metric value should not be a `list` instance, otherwise the `append` function
    will not work as intended.
    """
    def __init__(self, precision=None, recall=None, f1=None):
        self.precision = precision
        self.recall = recall
        self.f1 = f1

    def remove_attrs(self, item: Optional[str] = '<default>'):
        """
        Remove one or all default attributes (precision, recall, f1).
        Call this function in your `__init__` if you do not want those metrics

        Returns
        -------
        self
        """
        if item == '<default>':
            delattr(self, 'precision')
            delattr(self, 'recall')
            delattr(self, 'f1')
        else:
            delattr(self, item)
        return self

    def keys(self, idx: Optional[int] = None):
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

    def __len__(self):
        values = self[0]
        if values is None:
            return 0
        elif isinstance(values, (int, float)):
            return 1
        else:
            return len(values)

    def _value_lens(self):
        try:
            return [len(v) for v in self.values()]
        except Exception:
            return None

    def check_equal_length(self):
        lens = self._value_lens()
        if lens is None:
            return False
        return all_equal(lens)

    def __getitem__(self, item: Union[int, str]):
        if isinstance(item, str):
            return getattr(self, item)
        elif isinstance(item, int):
            return getattr(self, self.keys(item))
        else:
            raise KeyError(f'Unsupported index type: {type(item)}!')

    def __setitem__(self, key: Union[int, str], value: Any):
        if isinstance(key, str):
            return setattr(self, key, value)
        elif isinstance(key, int):
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

    def append(self, m):
        if m is None:
            return self
        self_attr_set = set(self.keys())
        m_attr_set = set(m.keys())
        if m_attr_set - self_attr_set:
            raise ValueError(f'Input instance has unknown attributes: {list(m_attr_set-self_attr_set)}')
        for k in m_attr_set:
            if m[k] is None:
                continue
            if self[k] is None:
                setattr(self, k, m[k])
            else:
                if not isinstance(self[k], list):
                    setattr(self, k, [self[k]])
                if isinstance(m[k], list):
                    self[k] += m[k]
                else:
                    self[k].append(m[k])
        return self

    def pop_attr(self, k: str):
        """
        Pop a given attribute

        Parameters
        ----------
        k: attribute name, string

        Returns
        -------
        popped attribute
        """
        attr = getattr(self, k)
        setattr(self, k, None)
        return attr


# noinspection PyTypeChecker
def get_ner_metrics(true_lbs,
                    pred_lbs,
                    mode: Optional[str] = 'strict',
                    scheme: Optional = IOB2,
                    detailed: Optional[bool] = False):
    """
    Get NER metrics including precision, recall and f1

    Parameters
    ----------
    true_lbs: true labels
    pred_lbs: predicted labels
    mode:
    scheme: NER label scheme (IOB-2 as default, [O, B-, I-] )
    detailed: Whether get detailed result report instead of micro-averaged one

    Returns
    -------
    Metrics if not detailed else Dict[str, Metrics]
    """
    if not detailed:
        p = metrics.precision_score(true_lbs, pred_lbs, mode=mode, zero_division=0, scheme=scheme)
        r = metrics.recall_score(true_lbs, pred_lbs, mode=mode, zero_division=0, scheme=scheme)
        f = metrics.f1_score(true_lbs, pred_lbs, mode=mode, zero_division=0, scheme=scheme)
        return Metric(p, r, f)

    else:
        metric_dict = dict()
        report = metrics.classification_report(
            true_lbs, pred_lbs, output_dict=True, mode=mode, zero_division=0, scheme=scheme
        )
        for tp, results in report.items():
            metric_dict[tp] = Metric(results['precision'], results['recall'], results['f1-score'])
        return metric_dict

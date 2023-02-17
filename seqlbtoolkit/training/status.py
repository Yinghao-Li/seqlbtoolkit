import copy
import regex
import torch
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union

logger = logging.getLogger(__name__)


class ModelBuffer:
    def __init__(self, size: Optional[int] = 1, smaller_is_better: Optional[bool] = False):
        """
        Parameters
        ----------
        size: buffer size, i.e., how many model will the buffer hold
        smaller_is_better: decides whether the metrics are in descend order
        """
        self._size = size
        self._smaller_is_better = smaller_is_better
        self._model_state_dicts = list()
        self._metrics = list()

    @property
    def size(self):
        return self._size

    @property
    def smaller_is_better(self):
        return self._smaller_is_better

    @size.setter
    def size(self, s):
        self._size = s

    @smaller_is_better.setter
    def smaller_is_better(self, sib):
        self._smaller_is_better = sib

    @property
    def model_state_dicts(self) -> list:
        self.sort()
        try:
            return self._model_state_dicts.tolist()
        except AttributeError:
            return self._model_state_dicts

    @property
    def best_state_dict(self):
        self.sort()
        return self._model_state_dicts[-1]

    @property
    def metrics(self) -> list:
        self.sort()
        try:
            return self._metrics.tolist()
        except AttributeError:
            return self._metrics

    @property
    def best_metrics(self):
        self.sort()
        return self._metrics[-1]

    def sort(self):
        sorted_ids = np.argsort(self._metrics)
        # in reverse order
        if not self.smaller_is_better:
            sorted_ids = sorted_ids[::-1]
        self._metrics = np.array(self._metrics)[sorted_ids]
        self._model_state_dicts = np.array(self._model_state_dicts)[sorted_ids]

    def check_and_update(self, metric: Union[int, float], model) -> bool:
        """
        Check whether the new model performs better than the buffered models.
        If so, replace the worst model in the buffer by the new model

        Parameters
        ----------
        metric: metric to compare the model performance
        model: the models

        Returns
        -------
        bool, whether there's any change to the buffer
        """

        model_cp = copy.deepcopy(model)
        try:
            model_cp.to('cpu')
        except AttributeError:
            pass

        if len(self._metrics) < self.size:
            self._metrics.append(metric)
            self._model_state_dicts.append(model_cp.state_dict())

            return True
        else:
            self.sort()

            if (self._metrics[-1] >= metric and self.smaller_is_better) \
                    or (self._metrics[-1] <= metric and not self.smaller_is_better):
                self._metrics[-1] = metric
                self._model_state_dicts[-1] = model_cp.state_dict()

                return True

        return False

    def save(self, model_dir: str):
        """
        Parameters
        ----------
        model_dir: which directory to save the model

        Returns
        -------
        None
        """
        out_dict = dict()
        for attr, value in self.__dict__.items():
            if regex.match(f"^_[a-z]", attr):
                out_dict[attr] = value

        torch.save(out_dict, model_dir)
        return None

    def load(self, model_dir: str):
        """
        Parameters
        ----------
        model_dir: from which directory to load the model

        Returns
        -------
        self
        """
        model_dict = torch.load(model_dir)

        for attr, value in model_dict.items():
            if attr not in self.__dict__:
                logger.warning(f"Attribute {attr} is not natively defined in model buffer!")
            setattr(self, attr, value)

        return self


class Status:

    def __init__(self, buffer_size=1, metric_smaller_is_better=False):
        self.eval_step: int = 0
        self.model_buffer: ModelBuffer = ModelBuffer(buffer_size, metric_smaller_is_better)

import os
import regex
import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        # Whether the text and lbs sequences are separated according to maximum BERT input lengths
        self.instances = None

    def __len__(self):
        return len(self.instances) if self.instances else 0

    def __getitem__(self, idx):
        return self.instances[idx]

    def prepare(self, config, partition: str):
        """
        Load data from disk

        Parameters
        ----------
        config: configurations
        partition: dataset partition; in [train, valid, test]

        Returns
        -------
        self
        """
        raise NotImplementedError

    def prepare_debug(self, n_inst: Optional[int] = 100):
        for attr in self.__dict__.keys():
            if regex.match(f"^_[a-z]", attr):
                try:
                    setattr(self, attr, getattr(self, attr)[:n_inst])
                except TypeError:
                    pass

        return self

    def save(self, file_path: str):
        """
        Save the entire dataset for future usage
        Parameters
        ----------
        file_path: path to the saved file
        Returns
        -------
        self
        """
        attr_dict = dict()
        for attr, value in self.__dict__.items():
            if regex.match(f"^_[a-z]", attr):
                attr_dict[attr] = value

        os.makedirs(os.path.dirname(os.path.normpath(file_path)), exist_ok=True)
        torch.save(attr_dict, file_path)

        return self

    def load(self, file_path: str):
        """
        Load the entire dataset from disk
        Parameters
        ----------
        file_path: path to the saved file
        Returns
        -------
        self
        """
        attr_dict = torch.load(file_path)

        for attr, value in attr_dict.items():
            if attr not in self.__dict__:
                logger.warning(f"Attribute {attr} is not natively defined in dataset!")

            setattr(self, attr, value)

        return self

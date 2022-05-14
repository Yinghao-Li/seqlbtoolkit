import os
import json
import logging
from typing import Optional, List
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig:
    """
    model & trainer base_model configuration
    """

    def from_args(self, args):
        """
        Initialize configuration from arguments

        Parameters
        ----------
        args: arguments (parent class)

        Returns
        -------
        self (type: BertConfig)
        """
        logger.info(f'Setting {type(self)} from {type(args)}.')
        arg_elements = {attr: getattr(args, attr) for attr in dir(args) if not callable(getattr(args, attr))
                        and not attr.startswith("__") and not attr.startswith("_")}
        logger.info(f'The following attributes will be changed: {arg_elements.keys()}')
        for attr, value in arg_elements.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self

    def from_config(self, config):
        """
        update configuration results from other config

        Parameters
        ----------
        config: other configurations

        Returns
        -------
        self (BertConfig)
        """
        logger.info(f'Setting {type(self)} from {type(config)}.')
        config_elements = {attr: getattr(config, attr) for attr in dir(self) if not callable(getattr(config, attr))
                           and not attr.startswith("__") and not attr.startswith("_")
                           and getattr(self, attr) is None}
        logger.info(f'The following attributes will be changed: {config_elements.keys()}')
        for attr, value in config_elements.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self

    def save(self, file_dir: str, file_name: Optional[str] = 'config'):
        """
        Save configuration to file

        Parameters
        ----------
        file_dir: file directory
        file_name: file name (suffix free)

        Returns
        -------
        self
        """
        if os.path.isdir(file_dir):
            file_path = os.path.join(file_dir, f'{file_name}.json')
        elif os.path.isdir(os.path.split(file_dir)[0]):
            file_path = file_dir
        else:
            raise FileNotFoundError(f"{file_dir} does not exist!")

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(self), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception(f"Cannot save config file to {file_path}; "
                             f"encountered Error {e}")
            raise e
        return self

    def load(self, file_dir: str, file_name: Optional[str] = 'config'):
        """
        Load configuration from stored file

        Parameters
        ----------
        file_dir: file directory
        file_name: file name (suffix free)

        Returns
        -------
        self
        """
        if os.path.isdir(file_dir):
            file_path = os.path.join(file_dir, f'{file_name}.json')
            assert os.path.isfile(file_path), FileNotFoundError(f"{file_path} does not exist!")
        elif os.path.isfile(file_dir):
            file_path = file_dir
        else:
            raise FileNotFoundError(f"{file_dir} does not exist!")

        logger.info(f'Setting {type(self)} parameters from {file_path}.')

        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        for attr, value in config.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass
        return self


@dataclass
class BaseNERConfig(BaseConfig):

    entity_types: Optional[List[str]] = None
    bio_label_types: Optional[List[str]] = None

    @property
    def n_lbs(self) -> "int":
        """
        Returns
        -------
        The number of BIO labels
        """
        return len(self.bio_label_types) if self.bio_label_types is not None else 0

    @property
    def n_ent(self) -> "int":
        """
        Returns
        -------
        The number of entity types
        """
        return len(self.entity_types) if self.entity_types is not None else 0

    @property
    def lb2idx(self):
        if self.bio_label_types is not None:
            return {lb: idx for idx, lb in enumerate(self.bio_label_types)}
        else:
            return {}

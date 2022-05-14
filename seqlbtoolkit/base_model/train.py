import os
import logging
import numpy as np
from typing import Optional, List, Any, Dict

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from .model_output import ModelOutput

logger = logging.getLogger(__name__)


def default_collate_fn(features: List[Any]) -> Dict[str, Any]:
    """
    default collate function for pytorch. Copied from `transformers`

    Parameters
    ----------
    features: input features

    Returns
    -------
    Dictionary of feature name: feature value
    """
    if not isinstance(features[0], dict):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor(np.array([f["label"] for f in features]), dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor(np.array([f["label_ids"] for f in features]), dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        dtype = torch.long if isinstance(v, int) else torch.float
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features]).to(dtype)
            else:
                batch[k] = torch.tensor(np.array([f[k] for f in features])).to(dtype)

    return batch


class BaseTrainer:
    def __init__(self,
                 config,
                 training_dataset=None,
                 valid_dataset=None,
                 test_dataset=None,
                 collate_fn=default_collate_fn):

        self._config = config
        self._training_dataset = training_dataset
        self._valid_dataset = valid_dataset
        self._test_dataset = test_dataset
        self._collate_fn = collate_fn
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._device = getattr(config, "device", "cpu")

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, x):
        logger.warning("Updating DirCHMMTrainer.config")
        self._config = x

    @property
    def model(self):
        return self._model

    def initialize(self):
        self.initialize_model()
        self.initialize_optimizer()
        self.initialize_scheduler()
        return self

    def initialize_model(self):
        logger.warning("Model is not defined by initialization function!")
        return self

    def initialize_optimizer(self):
        logger.warning("Optimizer is not defined by initialization function!")
        return self

    def initialize_scheduler(self):
        logger.warning("Scheduler is not defined by initialization function!")
        return self

    def get_dataloader(self,
                       dataset,
                       shuffle: Optional[bool] = False,
                       batch_size: Optional[int] = 0):
        try:
            dataloader = DataLoader(
                dataset=dataset,
                collate_fn=self._collate_fn,
                batch_size=batch_size if batch_size else self._config.batch_size,
                num_workers=getattr(self._config, "num_workers", 0),
                pin_memory=getattr(self._config, "pin_memory", False),
                shuffle=shuffle,
                drop_last=False
            )
        except Exception as e:
            logger.exception(e)
            raise e

        return dataloader

    def run(self, *args, **kwargs):
        """
        The main process of the trainer class
        """
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def training_step(self, dataloader):
        raise NotImplementedError

    def predict(self, dataset):
        """
        Predict labels/values given input dataset

        Parameters
        ----------
        dataset: input dataset

        Returns
        -------
        model output
        """
        data_loader = self.get_dataloader(dataset)

        self._model.to(self._device)
        self._model.eval()

        preds = ModelOutput()

        with torch.no_grad():
            for batch in data_loader:
                # get data and move to device
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self._device)

                # get model output
                preds_batch = self._model(**batch)
                preds += preds_batch

        return preds.unhook()

    def evaluate(self, dataset):
        raise NotImplementedError

    def valid(self):
        metrics = self.evaluate(self._valid_dataset)
        return metrics

    def test(self):
        metrics = self.evaluate(self._test_dataset)
        return metrics

    def save(self,
             output_dir: Optional[str] = None,
             save_optimizer: Optional[bool] = False,
             save_scheduler: Optional[bool] = False,
             model_name: Optional[str] = 'model',
             optimizer_name: Optional[str] = 'optimizer',
             scheduler_name: Optional[str] = 'scheduler'):
        """
        Save model parameters as well as trainer parameters

        Parameters
        ----------
        output_dir: model directory
        save_optimizer: whether to save optimizer
        save_scheduler: whether to save scheduler
        model_name: model name (suffix free)
        optimizer_name: optimizer name (suffix free)
        scheduler_name: scheduler name (suffix free)

        Returns
        -------
        None
        """
        output_dir = output_dir if output_dir is not None else getattr(self._config, 'output_dir', 'output')

        model_state_dict = self._model.state_dict()
        torch.save(model_state_dict, os.path.join(output_dir, f'{model_name}.bin'))

        self._config.save(output_dir)

        if save_optimizer:
            torch.save(self._optimizer.state_dict(), os.path.join(output_dir, f"{optimizer_name}.bin"))
        if save_scheduler and self._scheduler is not None:
            torch.save(self._scheduler.state_dict(), os.path.join(output_dir, f"{scheduler_name}.bin"))

        return None

    def load(self,
             input_dir: Optional[str] = None,
             load_optimizer: Optional[bool] = False,
             load_scheduler: Optional[bool] = False,
             model_name: Optional[str] = 'model',
             optimizer_name: Optional[str] = 'optimizer',
             scheduler_name: Optional[str] = 'scheduler'):
        """
        Load model parameters.

        Parameters
        ----------
        input_dir: model directory
        load_optimizer: whether load other trainer parameters
        load_scheduler: whether load scheduler
        model_name: model name (suffix free)
        optimizer_name: optimizer name (suffix free)
        scheduler_name: scheduler name

        Returns
        -------
        self
        """
        input_dir = input_dir if input_dir is not None else getattr(self._config, 'output_dir', 'output')

        logger.info(f"Loading model from {input_dir}")

        self.initialize_model()
        self._model.load_state_dict(torch.load(os.path.join(input_dir, f'{model_name}.bin')))
        self._model.to(self._device)

        if load_optimizer:
            logger.info("Loading optimizer")

            if self._optimizer is None:
                self.initialize_optimizer()

            if os.path.isfile(os.path.join(input_dir, f"{optimizer_name}.bin")):
                self._optimizer.load_state_dict(
                    torch.load(os.path.join(input_dir, f"{optimizer_name}.bin"), map_location=self._device)
                )
            else:
                logger.warning("Optimizer file does not exist!")

        if load_scheduler:
            logger.info("Loading scheduler")

            if self._scheduler is None:
                self.initialize_scheduler()

            if os.path.isfile(os.path.join(input_dir, f"{scheduler_name}.bin")):
                self._optimizer.load_state_dict(
                    torch.load(os.path.join(input_dir, f"{scheduler_name}.bin"), map_location=self._device)
                )
            else:
                logger.warning("Scheduler file does not exist!")
        return self

    def save_fig(self, fig, epoch: Optional[int] = None, directory="Plots"):
        save_dir = os.path.join(self._config.output_dir, directory)
        os.makedirs(save_dir, exist_ok=True)

        f_name = f'epoch-{epoch}.png' if epoch is not None else 'test'

        fig.savefig(os.path.join(save_dir, f_name), bbox_inches='tight')
        plt.close(plt.gcf())

    @staticmethod
    def log_tensorboard_fig(tensorboard_writer, fig, epoch: Optional[int] = 0, directory="Plots"):
        # Draw figure on canvas
        fig.canvas.draw()

        # Convert the figure to numpy array, read the pixel values and reshape the array
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # Normalize into 0-1 range for TensorBoard(X).
        # Swap axes for newer versions where API expects colors in first dim
        img = img / 255.0
        img = np.rot90(np.flipud(img), k=-1)
        img = np.swapaxes(img, 0, 2)  # if your TensorFlow + TensorBoard version are >= 1.8

        # Add figure in numpy "image" to TensorBoard writer
        tensorboard_writer.add_image(directory, img, epoch)
        plt.close(fig)

import logging
from dataclasses import dataclass
from typing import Optional

from ..base_model.config import BaseNERConfig

logger = logging.getLogger(__name__)


@dataclass
class BertBaseConfig(BaseNERConfig):
    """
    Conditional HMM training configuration
    """
    def save(self, file_dir: str, file_name: Optional[str] = 'bert-config') -> "BertBaseConfig":
        BaseNERConfig.save(self, file_dir, file_name)
        return self

    def load(self, file_dir: str, file_name: Optional[str] = 'bert-config') -> "BertBaseConfig":
        BaseNERConfig.load(self, file_dir, file_name)
        return self

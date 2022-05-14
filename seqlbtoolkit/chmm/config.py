import logging
from typing import Optional, List
from dataclasses import dataclass

from ..base_model.config import BaseNERConfig

logger = logging.getLogger(__name__)


@dataclass
class CHMMBaseConfig(BaseNERConfig):
    """
    Conditional HMM training configuration
    """
    sources: Optional[List[str]] = None
    src_priors: Optional[dict] = None
    d_emb: Optional[int] = None

    @property
    def d_hidden(self) -> "int":
        """
        Returns the HMM hidden dimension, AKA, the number of bio labels
        """
        return self.n_lbs

    @property
    def d_obs(self) -> "int":
        """
        Returns
        -------
        The observation dimension, equals to the number of bio labels
        """
        return self.n_lbs

    @property
    def n_src(self) -> "int":
        """
        Returns
        -------
        The number of sources
        """
        return len(self.sources) if self.sources is not None else 0

    def save(self, file_dir: str, file_name: Optional[str] = 'chmm-config') -> "CHMMBaseConfig":
        BaseNERConfig.save(self, file_dir, file_name)
        return self

    def load(self, file_dir: str, file_name: Optional[str] = 'chmm-config') -> "CHMMBaseConfig":
        BaseNERConfig.load(self, file_dir, file_name)
        return self


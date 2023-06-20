import os
import copy
import regex
import logging
import itertools
import operator
import torch
import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer

from seqlbtoolkit.text import (
    split_overlength_bert_input_sequence,
    substitute_unknown_tokens
)
from .base_dataset import BaseDataset


logger = logging.getLogger(__name__)


class NERDataset(BaseDataset):
    def __init__(self,
                 text: Optional[List[List[str]]] = None,
                 lbs: Optional[List[List[str]]] = None):
        super().__init__()
        self._text = text
        self._lbs = lbs
        self._sent_lens = None
        # Whether the text and lbs sequences are separated according to maximum BERT input lengths
        self._is_separated = False

    @property
    def n_insts(self):
        return len(self._text)

    @property
    def text(self):
        return self._text if self._text else list()

    @property
    def lbs(self):
        return self._lbs if self._lbs else list()

    @text.setter
    def text(self, value):
        self._text = value

    @lbs.setter
    def lbs(self, value):
        self._lbs = value

    def __len__(self):
        return self.n_insts

    def __add__(self, other):

        return NERDataset(
            text=copy.deepcopy(self.text + other.text),
            lbs=copy.deepcopy(self.lbs + other.lbs),
        )

    def __iadd__(self, other):

        self.text = copy.deepcopy(self.text + other.text)
        self.lbs = copy.deepcopy(self.lbs + other.lbs)
        return self

    def substitute_unknown_tokens(self, tokenizer_or_name):
        """
        Substitute the tokens in the sequences that cannot be recognized by the tokenizer
        This will not change sequence lengths

        Parameters
        ----------
        tokenizer_or_name: bert tokenizer

        Returns
        -------
        self
        """

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_name, add_prefix_space=True) \
            if isinstance(tokenizer_or_name, str) else tokenizer_or_name

        self._text = [substitute_unknown_tokens(tk_seq, tokenizer) for tk_seq in self._text]
        return self

    def separate_sequence(self, tokenizer_or_name, max_seq_length):
        """
        Separate the overlength sequences and separate the labels accordingly

        Parameters
        ----------
        tokenizer_or_name: bert tokenizer
        max_seq_length: maximum bert sequence length

        Returns
        -------
        self
        """
        if self._is_separated:
            logger.warning("The sequences are already separated!")
            return self

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_name, add_prefix_space=True) \
            if isinstance(tokenizer_or_name, str) else tokenizer_or_name

        if (np.array([len(tk_ids) for tk_ids in tokenizer(
                self._text, add_special_tokens=True, is_split_into_words=True
        ).input_ids]) <= max_seq_length).all():
            self._is_separated = True
            return self

        assert self._sent_lens, AttributeError("To separate sequences, attribute `_sent_lens` cannot be empty!")

        new_text_list = list()
        new_lbs_list = list()

        for sent_lens_inst, text_inst, lbs_inst in zip(self._sent_lens, self._text, self._lbs):

            split_tk_seqs = split_overlength_bert_input_sequence(text_inst, tokenizer, max_seq_length, sent_lens_inst)
            split_sq_lens = [len(tk_seq) for tk_seq in split_tk_seqs]

            seq_ends = list(itertools.accumulate(split_sq_lens, operator.add))
            seq_starts = [0] + seq_ends[:-1]

            split_lb_seqs = [lbs_inst[s:e] for s, e in zip(seq_starts, seq_ends)]

            new_text_list += split_tk_seqs
            new_lbs_list += split_lb_seqs

        self._text = new_text_list
        self._lbs = new_lbs_list

        self._is_separated = True

        return self

    def downsample_training_set(self, ids: List[int]):
        for attr in self.__dict__.keys():
            if regex.match(f"^_[a-z]", attr):
                try:
                    values = getattr(self, attr)
                    sampled_values = [values[idx] for idx in ids]
                    setattr(self, attr, sampled_values)
                except TypeError:
                    pass

        return self

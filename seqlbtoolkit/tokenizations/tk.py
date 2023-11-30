"""
# Author: Yinghao Li
# Modified: November 29th, 2023
# ---------------------------------------
# Description: token alignment functions
# Reference: https://github.com/explosion/tokenizations
"""

import unicodedata
from typing import List, Optional

from .diff import diff


def normalize(text: str) -> str:
    """Normalize text by converting to lowercase and applying Unicode NFKD normalization."""
    return unicodedata.normalize("NFKD", text.lower())


def get_char2token(tokens: List[str]) -> List[int]:
    """
    Map each character in the tokens to its corresponding token index.
    """
    token_lengths = [len(token) for token in tokens]
    character_to_token_map = []
    for token_index, length in enumerate(token_lengths):
        character_to_token_map.extend([token_index] * length)
    return character_to_token_map


def get_alignment(
    num_tokens: int,
    token_alignment: List[Optional[int]],
    source_char_to_token: List[int],
    target_char_to_token: List[int],
):
    """
    Create an alignment mapping between two tokenized sequences.
    """
    token_alignment_map = [[] for _ in range(num_tokens)]
    for source_token_index, aligned_token_index in zip(source_char_to_token, token_alignment):
        if aligned_token_index is not None:
            target_token_index = target_char_to_token[aligned_token_index]
            if (
                token_alignment_map[source_token_index]
                and token_alignment_map[source_token_index][-1] == target_token_index
            ):
                continue
            token_alignment_map[source_token_index].append(target_token_index)
    return token_alignment_map


def get_alignments(source_tokens: List[str], target_tokens: List[str]):
    """
    Get alignments between two lists of tokens.
    """
    normalized_source = [normalize(token) for token in source_tokens]
    normalized_target = [normalize(token) for token in target_tokens]

    source_char_to_token_map = get_char2token(normalized_source)
    target_char_to_token_map = get_char2token(normalized_target)

    source_to_target_alignment, target_to_source_alignment = diff(
        "".join(normalized_source), "".join(normalized_target)
    )

    source_to_target_token_alignment = get_alignment(
        len(normalized_source), source_to_target_alignment, source_char_to_token_map, target_char_to_token_map
    )
    target_to_source_token_alignment = get_alignment(
        len(normalized_target), target_to_source_alignment, target_char_to_token_map, source_char_to_token_map
    )

    return source_to_target_token_alignment, target_to_source_token_alignment


def get_charmap(source_str: str, target_str: str):
    """
    Map characters between two strings to understand their alignment.
    """
    source_characters = list(source_str)
    target_characters = list(target_str)
    return get_alignments(source_characters, target_characters)

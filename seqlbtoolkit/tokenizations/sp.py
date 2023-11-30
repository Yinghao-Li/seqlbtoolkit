"""
# Author: Yinghao Li
# Modified: November 29th, 2023
# ---------------------------------------
# Description: Align spans between tokenized text and original text
# Reference: https://github.com/tamuhey/textspan
"""


from typing import List, Tuple, Optional
from .tk import get_charmap

Span = Tuple[int, int]


def get_span_indices(tokens: List[str]) -> List[Span]:
    """Calculate span indices for each token in a list of tokens."""
    spans = []
    start_index = 0
    for token in tokens:
        end_index = start_index + len(token)
        spans.append((start_index, end_index))
        start_index = end_index
    return spans


def get_original_spans(tokens: List[str], original_text: str) -> List[List[Span]]:
    """Align spans from tokenized text with the original text."""
    token_spans = get_span_indices(tokens)
    joined_tokens = "".join(tokens)
    aligned_spans = align_spans(token_spans, joined_tokens, original_text)
    return aligned_spans


def align_spans(token_spans: List[Span], tokenized_text: str, original_text: str) -> List[List[Span]]:
    """Align spans between tokenized text and original text based on character mapping."""
    char_to_token_mapping, _ = get_charmap(tokenized_text, original_text)
    return align_spans_by_mapping(token_spans, char_to_token_mapping)


def align_spans_by_mapping(token_spans: List[Span], char_to_token_mapping: List[List[int]]) -> List[List[Span]]:
    """Align spans based on a character to token mapping."""
    aligned_spans = []
    for start, end in token_spans:
        span_start = None
        span_end = None
        previous_char_index = None
        current_span_list = []
        for char_indices in char_to_token_mapping[start:end]:
            for char_index in char_indices:
                if previous_char_index is not None and previous_char_index + 1 < char_index:
                    if span_start is not None:
                        current_span_list.append((span_start, span_end))
                    span_start = None
                span_end = char_index + 1
                if span_start is None:
                    span_start = char_index
                previous_char_index = char_index
        if span_start is not None:
            current_span_list.append((span_start, span_end))
        aligned_spans.append(current_span_list)
    return aligned_spans


def remove_span_overlaps(spans: List[Span]) -> List[Span]:
    """Remove overlapping spans, prioritizing longer spans."""
    sorted_spans = sorted(spans, key=lambda x: (x[0], -x[1]))
    non_overlapping_spans = []
    current_end = 0
    for start, end in sorted_spans:
        if start >= current_end:
            non_overlapping_spans.append((start, end))
            current_end = end
    return non_overlapping_spans


def remove_span_overlaps_idx(spans: List[Span]) -> List[int]:
    """Remove overlapping spans and return their indices, prioritizing longer spans."""
    sorted_indices = sorted(range(len(spans)), key=lambda i: (spans[i][0], -spans[i][1]))
    non_overlapping_indices = []
    current_end = 0
    for index in sorted_indices:
        start, end = spans[index]
        if start >= current_end:
            non_overlapping_indices.append(index)
            current_end = end
    return non_overlapping_indices


def lift_span_index(span: Span, target_spans: List[Span]) -> Tuple[Optional[int], Optional[int]]:
    """Find the start and end indices of a span within a list of target spans."""
    start, end = span
    start_index = next((index for index, (span_start, _) in enumerate(target_spans) if span_start == start), None)
    end_index = next((index for index, (_, span_end) in enumerate(target_spans) if span_end == end), None)
    return start_index, end_index


def lift_spans_index(spans: List[Span], target_spans: List[Span]) -> List[Tuple[Optional[int], Optional[int]]]:
    """Lift a list of spans to their respective indices in a list of target spans."""
    lifted_indices = []
    for span in spans:
        lifted_indices.append(lift_span_index(span, target_spans))
    return lifted_indices

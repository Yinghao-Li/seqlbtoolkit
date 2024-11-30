import copy
import regex
import itertools
import operator

import numpy as np


from .data import merge_list_of_lists


def format_text(text, remove_ref: bool = False, remove_emoj: bool = False):
    """
    Normalize text and transform some unicode characters into ascii

    Parameters
    ----------
    text: input text string
    remove_ref: whether remove reference tokens such as [1].
    remove_emoj: whether remove emojis from the text

    Returns
    -------
    normalized text string
    """

    # deal with interpuncts
    interpunct = (
        r"[\u00B7\u02D1\u0387\u05BC\u16EB\u2022\u2027\u2218\u2219\u22C5\u23FA"
        r"\u25CF\u25E6\u26AB\u2981\u2E30\u2E31\u2E33\u30FB\uA78F\uFF65]"
    )
    text = regex.sub(interpunct, " ", text)

    # deal with invisible characters
    text = regex.sub(r"[\u2060-\u2061|\u200b]", "", text)

    # deal with bullets
    bullets = r"[\u2022\u2023\u2043\u204C\u204D\u2219\u25CB\u25D8\u25E6\u2619\u2765\u2767\u29BE\u29BF]"
    text = regex.sub(bullets, " ", text)

    # deal with quotation marks
    s_quotation = r"[\u2018-\u201B\u2039\u203A]"
    d_quotation = r"[\u00AB\u00BB\u201C\u201D\u201E]"
    text = regex.sub(s_quotation, "'", text)
    text = regex.sub(d_quotation, '"', text)

    # deal with overlay tilde
    tilde = r"[\u0303\u223C\u224B\u02DC\u02F7\u223D\u0360\u0334\u0330\u033E\u1DEC\uFE29\uFE2A\uFE22\uFE23]"
    text = regex.sub(tilde, "~", text)

    # deal with overlay not tilde
    not_tilde = r"[\u034A]"
    text = regex.sub(not_tilde, "≁", text)

    text = remove_combining_marks(text)

    # deal with "/"
    text = regex.sub(r"[ ]?/[ ]?", "/", text)

    # deal with invisible Soft Hyphen
    text = regex.sub(r"[\u00ad]", " ", text)
    # deal with dash/hyphen
    text = regex.sub(r"[\p{Pd}]", "-", text)
    text = regex.sub(r"[\u2212\uf8ff\uf8fe\ue5f8]", "-", text)

    # deal with `=`
    text = regex.sub(r"[﹦＝]", "=", text)
    text = regex.sub(r"([^ ~≈=≅≤≥⩽⩾<>])([~≈=≅≤≥⩽⩾<>]+)([^ ~≈=≅≤≥⩽⩾<>])", r"\g<1> \g<2> \g<3>", text)

    # convert user-defined characters to dash
    text = regex.sub(r"[\uE000-\uF8FF]", "-", text)

    # deal with repeated comma and period
    text = regex.sub(r"[.]+( *[,.])+", r"..", text)

    # deal with white spaces and \n
    text = regex.sub(r"[\p{Z}]", " ", text)
    text = regex.sub(r"([ \t]+)?[\r\n]([ \t]+)?", " ", text)
    text = regex.sub(r"\n+", " ", text)
    text = regex.sub(r"([ ]{2,})", " ", text)

    if remove_ref:
        text = remove_reference(text)

    if remove_emoj:
        text = remove_emojis(text)

    text = text.strip()
    return text


def remove_emojis(text):
    """
    Remove emojis in the text
    """
    emoj = regex.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030]+",
        regex.UNICODE,
    )
    return regex.sub(emoj, "", text)


def remove_reference(text: str):
    """
    Remove reference tokens such as [1] from text
    """
    # deal with in-sentence references
    text = regex.sub(r" *\[[0-9-, ]+][,-]*", r"", text)
    # deal with after-sentence references
    text = regex.sub(
        r'([A-Za-zα-ωΑ-Ω\p{Pe}\'"\u2018-\u201d]+[\d-]*)([.])( |[\d]+)([-, ]*[\d]*)*'
        r'([ ]+[A-Z\dα-ωΑ-Ω\p{Ps}\'"\u2018-\u201d]|$)',
        r"\g<1>\g<2>\g<5>",
        text,
    )

    return text


def remove_combining_marks(text: str):
    """
    Remove combining marks (with unicode 0300-036f)

    Parameters
    ----------
    text: input string text

    Returns
    -------
    string text
    """
    # deal with interpuncts
    diacritics = r"[\u0300-\u036F]"
    text = regex.sub(diacritics, "", text)

    return text


def substring_mapping(text: str, mapping_dict: dict):
    """
    Map substrings in the input string according to the dict

    Parameters
    ----------
    text: input string
    mapping_dict: the mapping dictionary

    Returns
    -------
    string text
    """
    rep = dict((regex.escape(k), v) for k, v in mapping_dict.items())
    pattern = regex.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[regex.escape(m.group(0))], text)
    return text


def split_overlength_bert_input_sequence(
    sequence: str | list[str],
    tokenizer,
    max_seq_length: int = 512,
    sent_lens: list[int] = None,
) -> list[list[str]]:
    """
    Break the sentences that exceeds the maximum BERT length

    Parameters
    ----------
    sequence: The original text sequence, could be a string, a list tokens,
        or a list of sentences split into tokens (list of lists of string, preferred).
        If the input sequence is in the first two format, it will be split into sentences and tokens using
        `nltk.sent_tokenizer` and `nltk.word_tokenizer`.
    tokenizer: BERT tokenizer used to check input length shen encoded using BERT's vocabulary
    max_seq_length: maximum BERT sequence length
    sent_lens: The length of each sentence.

    Returns
    -------
    a list of token collections (list containing tokens from multiple sentences)
    """

    if isinstance(sequence, str):
        from nltk import word_tokenize, sent_tokenize

        tks_seq_list = [word_tokenize(sent) for sent in sent_tokenize(sequence)]
    elif isinstance(sequence[0], str):
        if sent_lens:
            ends = list(itertools.accumulate(sent_lens, operator.add))
            starts = [0] + ends[:-1]
            tks_seq_list = [sequence[s:e] for s, e in zip(starts, ends)]
        else:
            from nltk import sent_tokenize

            tks_seq_list = [sent.split(" ") for sent in sent_tokenize(" ".join(sequence))]
    else:
        raise TypeError("Input parameter `sequence` has Unknown type.")

    tks = merge_list_of_lists(tks_seq_list)
    if len(tokenizer.tokenize(tks, add_special_tokens=True, is_split_into_words=True)) <= max_seq_length:
        return [sequence]

    if isinstance(sequence, list):
        assert len(tks) == len(sequence), (
            "Sentence tokenization changed the original tokens! "
            "Consider assigning values to `sent_lens` to disable "
            "automatic sentence tokenization!"
        )

    seq_bert_len_list = [
        len(tokenizer.tokenize(tks_seq, add_special_tokens=True, is_split_into_words=True)) for tks_seq in tks_seq_list
    ]

    assert (np.asarray(seq_bert_len_list) <= max_seq_length).all(), ValueError(
        "One or more sentences in the input sequence are longer than the designated maximum length."
    )

    split_points = [0, len(tks_seq_list)]
    split_bert_lens = [
        sum(seq_bert_len_list[split_points[i] : split_points[i + 1]]) for i in range(len(split_points) - 1)
    ]

    while (np.asarray(split_bert_lens) > max_seq_length).any():
        new_split_points = list()
        for idx, bert_len in enumerate(split_bert_lens):
            if bert_len > max_seq_length:
                seq_bert_len_sub_list = seq_bert_len_list[split_points[idx] : split_points[idx + 1]]
                seq_bert_len_sub_accu_list = list(itertools.accumulate(seq_bert_len_sub_list, operator.add))
                # try to separate sentences as evenly as possible
                split_offset = np.argmin((np.array(seq_bert_len_sub_accu_list) - bert_len / 2) ** 2)
                new_split_points.append(split_offset + split_points[idx] + 1)

        split_points += new_split_points
        split_points.sort()

        split_bert_lens = [
            sum(seq_bert_len_list[split_points[i] : split_points[i + 1]]) for i in range(len(split_points) - 1)
        ]

    split_tks_seq_list = [
        merge_list_of_lists(tks_seq_list[split_points[i] : split_points[i + 1]]) for i in range(len(split_points) - 1)
    ]

    return split_tks_seq_list


def substitute_unknown_tokens(tk_seq: list[str], tokenizer) -> list[str]:
    """
    Substitute the tokens in tk_seq unknown to the tokenizer by `unk_tag`

    Parameters
    ----------
    tk_seq: a list (sequence) of tokens
    tokenizer: a loaded BERT tokenizer

    Returns
    -------
    token sequences with unknown tokens substituted
    """

    tks = np.asarray(copy.deepcopy(tk_seq))

    word_ids = tokenizer(tk_seq, is_split_into_words=True).word_ids(batch_index=0)
    ori_tk_ids = np.arange(len(tk_seq))

    word_ids_shifted_left = np.asarray([-100] + word_ids[:-1])
    word_ids = np.asarray(word_ids)

    is_first_wordpiece = (word_ids_shifted_left != word_ids) & (word_ids != None)
    word_ids[~is_first_wordpiece] = -100  # could be anything less than 0

    tks[np.setdiff1d(ori_tk_ids, word_ids)] = tokenizer.unk_token
    return tks.tolist()


def remove_invalid_parenthesis(sent: str) -> str:
    remove_ind = set()
    stack_ind = []
    for ind, char in enumerate(sent):
        if char not in "()":
            continue
        if char == "(":
            stack_ind.append(ind)
        elif not stack_ind:
            remove_ind.add(ind)
        else:
            stack_ind.pop()
    remove_ind = remove_ind.union(set(stack_ind))
    output = ""
    for ind, char in enumerate(sent):
        if ind in remove_ind:
            continue
        output += char
    return output

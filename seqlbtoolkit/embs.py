import logging
import numpy as np
from tqdm import tqdm
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModel

from .text import split_overlength_bert_input_sequence

logger = logging.getLogger(__name__)


def build_bert_token_embeddings(tk_seq_list: List[List[str]],
                                model_or_name,
                                tokenizer_or_name,
                                max_seq_length: Optional[int] = 512,
                                device: Optional = 'cpu',
                                prepend_cls_embs: Optional[bool] = False) -> List[torch.Tensor]:
    """
    Build the BERT token embeddings of the input sentences

    Parameters
    ----------
    tk_seq_list: a list of token sequences
    model_or_name: a loaded Huggingface BERT model or its name
    tokenizer_or_name: a loaded Huggingface tokenizer or its name
    max_seq_length: maximum input sequence length that the BERT model allow
    device: device
    prepend_cls_embs: whether add the embeddings corresponding the CLS token back at the beginning of each sequence

    Returns
    -------
    List[torch.Tensor]
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_name) if isinstance(tokenizer_or_name, str) \
        else tokenizer_or_name
    model = AutoModel.from_pretrained(model_or_name) if isinstance(model_or_name, str) else model_or_name
    model.to(device)

    split_tk_seq_list = list()
    ori2split_ids_map = list()
    n = 0

    # update input sentences so that every sentence has BERT length < 510
    logger.info(f'Checking lengths. Paragraphs longer than {max_seq_length} tokens will be separated.')
    for tk_seq in tk_seq_list:
        tk_seqs, _, _ = split_overlength_bert_input_sequence(tk_seq, tokenizer, max_seq_length)
        n_splits = len(tk_seqs)
        split_tk_seq_list += tk_seqs

        ori2split_ids_map.append(list(range(n, n + n_splits)))
        n += n_splits

    logger.info('Building embeddings...')
    sent_emb_list = build_emb_helper(split_tk_seq_list, tokenizer, model, device, prepend_cls_embs)

    # Combine embeddings so that the embedding lengths equal to the lengths of the original sentences
    logger.info('Combining results...')
    tk_emb_seq_list = list()

    for sep_ids in ori2split_ids_map:
        cat_emb = None

        for sep_idx in sep_ids:
            if cat_emb is None:
                cat_emb = sent_emb_list[sep_idx]
            else:
                cat_emb = torch.cat([cat_emb, sent_emb_list[sep_idx][1:]], dim=0)

        assert cat_emb is not None, ValueError('Empty embedding!')
        tk_emb_seq_list.append(cat_emb)

    # The embeddings of [CLS] + original tokens
    for emb, tk_seq in zip(tk_emb_seq_list, tk_seq_list):
        if prepend_cls_embs:
            assert len(emb) == len(tk_seq) + 1  # check the length of embeddings equals to the length of sequence
        else:
            assert len(emb) == len(tk_seq)

    return tk_emb_seq_list


def build_emb_helper(tk_seq_list: List[List[str]],
                     tokenizer,
                     model,
                     device: Optional = 'cpu',
                     prepend_cls_embs: Optional[bool] = False):

    tk_emb_seq_list = list()

    for tk_seq in tqdm(tk_seq_list):

        encs = tokenizer(tk_seq, is_split_into_words=True, add_special_tokens=True, return_offsets_mapping=True)
        input_ids = torch.tensor([encs.input_ids], device=device)
        offsets_mapping = np.array(encs.offset_mapping)

        # calculate BERT last layer embeddings
        with torch.no_grad():
            # get the last hidden state from the BERT model
            last_hidden_states = model(input_ids)[0].squeeze(0).to('cpu')
            # remove the token embeddings regarding the [CLS] and [SEP]
            trunc_hidden_states = last_hidden_states[1:-1, :]

        ori2bert_tk_ids = list()
        idx = 0
        for tk_start in (offsets_mapping[1:-1, 0] == 0):
            if tk_start:
                ori2bert_tk_ids.append([idx])
            else:
                ori2bert_tk_ids[-1].append(idx)
            idx += 1

        emb_list = list()
        for ids in ori2bert_tk_ids:
            embeddings = trunc_hidden_states[ids, :]  # first dim could be 1 or n
            emb_list.append(embeddings.mean(dim=0))

        if prepend_cls_embs:
            # add back the embedding of [CLS] as the sentence embedding
            emb_list = [last_hidden_states[0, :]] + emb_list

        bert_emb = torch.stack(emb_list)
        assert not bert_emb.isnan().any(), ValueError('NaN Embeddings!')
        tk_emb_seq_list.append(bert_emb.detach().cpu())

    return tk_emb_seq_list

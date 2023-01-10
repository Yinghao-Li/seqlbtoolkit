from transformers import AutoTokenizer, AutoModel

from seqlbtoolkit.text import substitute_unknown_tokens
from seqlbtoolkit.embs import build_emb_helper


tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModel.from_pretrained('distilbert-base-uncased')

tk_seq = ["q", "x", "0.64Å", "-1", "represents", "the", "second", "-", "order", "(", "200", ")",
          "", "reection", "of", "lamellae", "(", "d", "200", "spacing", "x", "9.83Å", ").", "[UNK]"]

tks = substitute_unknown_tokens(tk_seq, tokenizer)
emb_list = build_emb_helper([tks], tokenizer, model)

assert len(emb_list[0]) == len(tk_seq)

print("Test Success!")

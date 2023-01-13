from transformers import AutoTokenizer

from seqlbtoolkit.text import substitute_unknown_tokens


tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)

tk_seq = ["q", "x", "0.64Å", "-1", "represents", "the", "second", "-", "order", "(", "200", ")",
          "", "reection", "of", "lamellae", "(", "d", "200", "spacing", "x", "9.83Å", ").", "[UNK]"]

tks = substitute_unknown_tokens(tk_seq, tokenizer)

print("Test Success!")

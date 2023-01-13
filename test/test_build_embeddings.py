from transformers import AutoTokenizer, AutoModel

from seqlbtoolkit.text import substitute_unknown_tokens
from seqlbtoolkit.embs import build_emb_helper

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', add_prefix_space=True)
model = AutoModel.from_pretrained('distilbert-base-uncased')

tk_seq = ["q", "x", "0.64Å", "-1", "represents", "the", "second", "-", "order", "(", "200", ")",
          "", "reection", "of", "lamellae", "(", "d", "200", "spacing", "x", "9.83Å", ").", "[UNK]"]


# tk_seq = ['The', 'HOMO-LUMO', '(', 'HL', ')', 'gap', 'of', 'each', 'structure', 'calculated', 'using', 'the',
#           'hybrid', 'B3LYP', 'functional', 'is', 'notably', 'smaller', ',', 'by', '∼', '0.4', 'eV', ',', 'than',
#           'that', 'using', 'the', 'meta-hybrid', 'M06', 'functional', ',', 'however', ',', 'the', 'calculated',
#           'optical', 'gaps', 'are', 'only', 'marginally', 'smaller', ',', 'with', 'a', 'difference', 'of', '∼',
#           '0.1', 'eV', '.', 'In', 'Table', '2', ',', 'we', 'also', 'provide', 'the', 'character', 'of', 'the',
#           'first', 'allowed', 'excitations', 'only', 'for', 'contributions', 'larger', 'than', '4', '%', '.',
#           'The', 'first', 'excitation', ',', 'as', 'calculated', 'by', 'each', 'of', 'the', 'functional', 'for',
#           'all', 'three', 'structures', ',', 'clearly', 'exhibits', 'a', 'single', '-', 'configuration',
#           'character', '.', 'In', 'Fig.', '7', ',', 'we', 'have', 'plotted', 'the', 'isosurfaces', '(',
#           'isovalue', '=', '0.02', ')', 'of', 'the', 'HOMO', 'and', 'LUMO', 'for', 'both', 'structures', '.',
#           'In', 'both', 'cases', 'the', 'HOMO', 'extends', 'evenly', 'over', 'the', 'main', 'body', '.', 'For',
#           'the', 'LUMO', 'of', 'each', 'structure', 'the', 'delocalizations', 'are', 'also', 'similar', '.',
#           'The', 'LUMO', 'of', 'P1', 'and', 'P2', 'extends', 'over', 'the', 'main', 'structure', 'but',
#           'considerably', 'more', 'over', 'the', 'triazole', 'group', 'than', 'in', 'the', 'case', 'of', 'the',
#           'respective', 'HOMOs', '.', 'To', 'quantify', 'the', 'contributions', 'of', 'the', 'moieties', 'to',
#           'the', 'frontier', 'orbitals', 'we', 'have', 'calculated', 'the', 'total', 'and', 'partial', 'density',
#           'of', 'states', '(', 'PDOS', ').', 'The', 'PDOSs', 'for', 'P1', 'and', 'P2', 'are', 'shown', 'in',
#           'Fig.', 'S6', '(', 'ESI', '†', ').', 'We', 'partition', 'all', 'of', 'the', 'structures', 'into',
#           'the', 'silolodithiophene', '(', 'SDT', ')', 'and', 'fluorobenzotriazole', '(', 'FBT', ')', 'moieties',
#           '.', 'As', 'expected', ',', 'structures', 'P1', 'and', 'P2', 'have', 'significant', 'similarities',
#           'on', 'the', 'delocalization', 'of', 'the', 'frontier', 'orbitals', '.']

tks = substitute_unknown_tokens(tk_seq, tokenizer)
embs = build_emb_helper([tks], tokenizer, model)



print("Test Success!")

import json
import wandb
from seqlbtoolkit.base_model.eval import get_ner_metrics

y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]

r = get_ner_metrics(y_true, y_pred, detailed=True)

wandb.init(
    project='test',
    name='test'
)
columns = ['ent', 'precision', 'recall', 'f1']
tb = wandb.Table(columns)

for ent, metrics in r.items():
    row = [ent]
    for value in metrics.values():
        row.append(value)
    tb.add_data(*row)
wandb.run.log({'tb': tb})


print('Finished')

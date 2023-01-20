from seqlbtoolkit.io import save_json

d = {'a':{'b':{'c':'d'}}}

save_json(d, './t.json', 3)

import re
import json
from seqlbtoolkit.io import dumps_json


a = {
    "\\data": [
        {"a": 1, "b": 2, "c": [1, 2, 3, {"d": 4, "e": 5, "f": [6, 7, "'\"8"]}]},
        "\\\\x",
        "y",
        "z",
        (1, 2, 3, 4, 5),
    ]
}


dumps = dumps_json(a, expand_to_level=2, indent=2)
print(dumps)


def restore_escape(m):
    s = m.group(0)
    c_idx = s.find(":")
    b_idx = s.find("{")
    if c_idx < b_idx or b_idx == -1 and c_idx != -1:
        s = re.sub(r": \"", r": ", s, count=1)
    else:
        s = re.sub(r"^( +)\"", r"\g<1>", s)
    s = re.sub(r"\"(,{0,1})$", r"\g<1>", s)
    s = re.sub(r'\\"', r'"', s)
    s = re.sub(r"\\\\", r"\\", s)
    return s


dp = dumps
dp = re.sub(r"^( {%d,}.*)$" % (3 * 2), restore_escape, dp, flags=re.MULTILINE)
print(dp)

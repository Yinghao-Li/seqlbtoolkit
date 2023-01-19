# SeqLbToolkit

This repo realizes Sequence Labeling Toolkits, a toolkit box containing useful functions for accelerating implementing sequences labeling deep learning models such as BiLSTM-CRF or BERT for Token Classification.

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple)](https://www.python.org/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Yinghao-Li/seqlbtoolkit/)
[![PyPI version](https://badge.fury.io/py/SeqLbToolkit.svg)](https://badge.fury.io/py/SeqLbToolkit)
![GitHub stars](https://img.shields.io/github/stars/Yinghao-Li/seqlbtoolkit.svg?color=gold)
![GitHub forks](https://img.shields.io/github/forks/Yinghao-Li/seqlbtoolkit?color=9cf)

## 1. Installation

- Install from PyPI:
```bash
pip install -U SeqLbLoolkit
```

- Install from wheel:
```bash
wget https://github.com/Yinghao-Li/seqlbtoolkit/releases/download/latest/SeqLbToolkit-latest-py3-none-any.whl
pip install SeqLbToolkit-latest-py3-none-any.whl
```

## 2. Documentation

### 2.1. IO
This module defines frequently-used IO control functions.
```python
from seqlbtoolkit.io import (set_logging, save_json)

# setup logging informatoin format as
# "%(asctime)s - %(levelname)s - %(name)s -   %(message)s"; For example:
# "01/06/2023 17:28:31 - INFO - seqlbtoolkit.io - <logging information>"
set_logging(your_logging_dir)


```

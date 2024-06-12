import os
import os.path as osp
import re
import json
import yaml
import shutil
import logging
import textwrap
from pathlib import Path
from typing import Optional
from dataclasses import asdict
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from .utils import deprecated

logger = logging.getLogger(__name__)


__all__ = [
    "progress_bar",
    "set_logging",
    "log_args",
    "remove_dir",
    "init_dir",
    "save_json",
    "dump_json",
    "dumps_json",
    "save_yaml",
    "dumps_yaml",
]


"""
Define custom progress bar.
Usage:

```python
with progress_bar as p:
    for i in p.track(range(1000)):
        # Do something here
        pass
```
"""
progress_bar = Progress(
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)


def set_logging(log_path: Optional[str] = None, level: str = "NOTSET"):
    """Sets up logging format and file handler.

    Args:
        log_path (Optional[str]): Path to save the logging file. If None, no log file is saved.
    """
    rh = RichHandler()
    rh.setFormatter(logging.Formatter("%(message)s", datefmt="[%m/%d %X]"))

    if log_path:
        log_path = osp.abspath(log_path)
        if not osp.isdir(osp.split(log_path)[0]):
            os.makedirs(osp.abspath(osp.normpath(osp.split(log_path)[0])))
        if osp.isfile(log_path):
            os.remove(log_path)

        file_handler = logging.FileHandler(filename=log_path)
        file_handler.setLevel(logging.DEBUG)

        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)-8s %(message)-80s     @ %(pathname)-s:%(lineno)d",
            datefmt="[%m/%d %X]",
            handlers=[file_handler, rh],
        )

    else:
        logging.basicConfig(
            datefmt="[%m/%d %X]",
            level=level,
            handlers=[rh],
        )

    return None


def log_args(args):
    """
    Logging model arguments

    Parameters
    ----------
    args: arguments

    Returns
    -------
    None
    """
    arg_elements = {
        attr: getattr(args, attr)
        for attr in asdict(args)
        if not callable(getattr(args, attr)) and not attr.startswith("_")
    }
    arg_string = yaml.dump(arg_elements, default_flow_style=False, sort_keys=False)
    arg_string = textwrap.indent(arg_string, "  ")
    logger.info(f"Configurations ({type(args).__name__}):\n{arg_string}")

    return None


@deprecated("Use `log_args` instead.")
def logging_args(args):
    """
    Logging model arguments

    Parameters
    ----------
    args: arguments

    Returns
    -------
    None
    """
    log_args(args)
    return None


def remove_dir(directory: str):
    """
    Remove a directory and its subtree folders/files
    """
    dirpath = Path(directory)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    return None


def init_dir(directory: str, clear_original_content: Optional[bool] = True):
    """
    Create the target directory. If the directory exists, remove all subtree folders/files in it.
    """

    if clear_original_content:
        remove_dir(directory)
    os.makedirs(os.path.normpath(directory), exist_ok=True)
    return None


@deprecated
def replace_pattern_with_list(input_string, pattern, replacements):
    # Find all occurrences of the pattern
    occurrences = re.findall(pattern, input_string)

    # Check if the number of occurrences matches the length of the replacements list
    if len(occurrences) != len(replacements):
        print(len(occurrences), len(replacements))
        raise ValueError("The number of replacements does not match the number of occurrences.")

    # Replace each occurrence with an element from the replacements list
    for replacement in replacements:
        input_string = re.sub(pattern, replacement.replace(r"\n", r"\\n"), input_string, count=1)

    return input_string


@deprecated("Use `dumps_json` instead.")
def prettify_json(text, indent=2, collapse_level=4, disable_content_checking=False):
    """
    Make json file more readable by collapsing indent levels higher than `collapse_level`.

    Parameters
    ----------
    text: input json text obj
    indent: the indent value of your json text. Notice that this value needs to be larger than 0
    collapse_level: the level from which the program stops adding new lines
    disable_content_checking: set to True to disable content checking within quotation marks.
        Content checking is used to protect text within quotation marks from being collapsed.
        Setting this to True will make the program run faster but may cause unexpected results.

    Usage
    -----
    ```
    my_instance = list()  # user-defined serializable data structure
    json_obj = json.dumps(my_instance, indent=2, ensure_ascii=False)
    json_obj = prettify_json(json_text, indent=2, collapse_level=4)
    with open(path_to_file, 'w', encoding='utf=8') as f:
        f.write(json_text)
    ```
    """

    if not disable_content_checking:
        # protect text within quotation marks
        pattern = r'((?<!\\)"(?:.*?)(?<!\\)")'
        quoted_text = re.findall(pattern, text)
        text = re.sub(pattern, '"!@#$CONTENT$#@!"', text)

    pattern = r"[\r\n]+ {%d,}" % (indent * collapse_level)
    text = re.sub(pattern, " ", text)
    text = re.sub(r"([\[({])+ +", r"\g<1>", text)
    text = re.sub(r"[\r\n]+ {%d}([])}])" % (indent * (collapse_level - 1)), r"\g<1>", text)
    text = re.sub(r"(\S) +([])}])", r"\g<1>\g<2>", text)

    if not disable_content_checking:
        text = replace_pattern_with_list(text, re.escape('"!@#$CONTENT$#@!"'), quoted_text)

    return text


def save_json(
    obj,
    path: str,
    expand_to_level: int = None,
    collapse_level: int = None,
    indent: int = 2,
    ensure_ascii: bool = False,
    disable_content_checking: bool = None,
    **kwargs,
):
    """
    Save objective to a json file.
    Create this function so that we don't need to worry about creating parent folders every time

    Args
        obj: the objective to save
        path: the path to save
        expand_to_level: set to any collapse value to prettify output json accordingly
        collapse_level (deprecated): same as `expand_to_level`
        disable_content_checking (deprecated): set to True to disable content checking within quotation marks.
            Content checking is used to protect text within quotation marks from being collapsed.
            Setting this to True will make the program run faster but may cause unexpected results.
        indent: the indent value of your json text. Notice that this value needs to be None or larger than 0
        ensure_ascii: set to True to escape non-ASCII characters

    """
    if collapse_level is not None and expand_to_level is None:
        logger.warning("`collapse_level` is deprecated. Please use `expand_to_level` instead.")
        expand_to_level = collapse_level
    if disable_content_checking is not None:
        logger.warning("`disable_content_checking` is deprecated. Please remove it from your function call.")

    file_dir = os.path.dirname(os.path.normpath(path))
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        dump_json(obj, f, expand_to_level=expand_to_level, indent=indent, ensure_ascii=ensure_ascii, **kwargs)

    return None


def save_yaml(obj, path: str, **kwargs):
    """
    Save objective to a yaml file.
    Create this function so that we don't need to worry about creating parent folders every time

    Args
        obj: the objective to save
        path: the path to save
    """
    file_dir = os.path.dirname(os.path.normpath(path))
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(dumps_yaml(obj, **kwargs))

    return None


def dump_json(obj, f, expand_to_level=None, **kwargs) -> None:
    """Save an object to a JSON file.

    Args:
        obj: The object to save.
        f: The file object to write to.
        expand_to_level: The level to expand. This function will behave like json.dump if set to None.
        **kwargs: Additional arguments for json.dump.

    Returns:
        None
    """
    obj = dumps_json_recursive(obj, tgt_lvl=expand_to_level) if expand_to_level is not None else obj

    json.dump(obj, f, **kwargs)

    return None


def dumps_yaml(obj, default_flow_style=False, sort_keys=False, **kwargs) -> str:
    """Convert a Python object to a YAML string.

    Args:
        obj: Python object to convert.
        **kwargs: Additional arguments for yaml.dump.

    Returns:
        str: YAML string.
    """
    return yaml.dump(obj, default_flow_style=default_flow_style, sort_keys=sort_keys, **kwargs)


def dumps_json(x, expand_to_level=None, **kwargs):
    """Convert a Python object to a JSON string with optional expand level.

    Args:
        x: Python object to convert.
        expand_to_level: The level to expand. This function will behave like json.dumps if set to None.
        **kwargs: Additional arguments for json.dumps.

    Examples:
        ```python
        >>> a = {"data": [{'a': 1, 'b': 2, 'c': [1,2,3, {'d': 4, 'e': 5, 'f': [6,7,'\'\"8']}]},"x", "y", "z"]}
        >>> print(dumps_json(a, expand_to_level=2, indent=2))
        {
          "data": [
            "{\"a\": 1, \"b\": 2, \"c\": [1, 2, 3, {\"d\": 4, \"e\": 5, \"f\": [6, 7, \"'\\\"8\"]}]}",
            "\"x\"",
            "\"y\"",
            "\"z\""
          ]
        }
        >>> print(dumps_json(a, indent=2))
        {
          "data": [
            {
              "a": 1,
              "b": 2,
              "c": [
                1,
                2,
                3,
                {
                  "d": 4,
                  "e": 5,
                  "f": [
                    6,
                    7,
                    "'\"8"
                  ]
                }
              ]
            },
            "x",
            "y",
            "z"
          ]
        }
        >>> print(dumps_json(a, indent=2, expand_to_level=0))
        {"data": [{"a": 1, "b": 2, "c": [1, 2, 3, {"d": 4, "e": 5, "f": [6, 7, "'\"8"]}]}, "x", "y", "z"]}
        ```

    Returns:
        str: JSON string.
    """
    if expand_to_level is None:
        return json.dumps(x, **kwargs)
    return json.dumps(dumps_json_recursive(x, tgt_lvl=expand_to_level), **kwargs)


def dumps_json_recursive(x, curr_lvl=0, tgt_lvl=2):
    """Convert the parts of an recursive object whose depth deeper than tgt_lvl to JSON strings.

    Args:
        x: Python object to convert.
        curr_lvl: Current level (depth) of recursion.
        tgt_lvl: Target level (depth) of recursion.

    Returns:
        obj: the original object with parts deeper than tgt_lvl converted to JSON strings.
    """
    if not isinstance(x, (dict, list, tuple)) or curr_lvl == tgt_lvl:
        return json.dumps(x)
    if isinstance(x, dict):
        return {k: dumps_json_recursive(v, curr_lvl + 1, tgt_lvl) for k, v in x.items()}
    if isinstance(x, list):
        return [dumps_json_recursive(v, curr_lvl + 1, tgt_lvl) for v in x]
    if isinstance(x, tuple):
        return tuple(dumps_json_recursive(v, curr_lvl + 1, tgt_lvl) for v in x)

import os
import os.path as osp
import re
import json
import shutil
import logging
from rich.logging import RichHandler
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def set_logging(log_path: Optional[str] = None):
    """Sets up logging format and file handler.

    Args:
        log_path (Optional[str]): Path where to save the logging file. If None, no log file is saved.
    """
    rh = RichHandler()
    rh.setFormatter(logging.Formatter("%(message)s", datefmt="[%m/%d %X]"))

    if log_path and log_path != "disabled":
        log_path = osp.abspath(log_path)
        if not osp.isdir(osp.split(log_path)[0]):
            os.makedirs(osp.abspath(osp.normpath(osp.split(log_path)[0])))
        if osp.isfile(log_path):
            os.remove(log_path)

        file_handler = logging.FileHandler(filename=log_path)
        file_handler.setLevel(logging.DEBUG)

        logging.basicConfig(
            level="NOTSET",
            format="%(asctime)s %(levelname)-8s %(message)-80s     @ %(pathname)-s:%(lineno)d",
            datefmt="[%m/%d %X]",
            handlers=[file_handler, rh],
        )

    else:
        logging.basicConfig(
            datefmt="[%m/%d %X]",
            level="NOTSET",
            handlers=[rh],
        )

    return None


def logging_args(args):
    """
    Logging model arguments into logs
    Last modified: 08/19/21

    Parameters
    ----------
    args: arguments

    Returns
    -------
    None
    """
    arg_elements = {
        attr: getattr(args, attr)
        for attr in dir(args)
        if not callable(getattr(args, attr)) and not attr.startswith("__") and not attr.startswith("_")
    }
    logger.info(f"Parameters: ({type(args)})")
    for arg_element, value in arg_elements.items():
        logger.info(f"  {arg_element}: {value}")

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
    os.makedirs(osp.normpath(directory), exist_ok=True)
    return None


def save_json(obj, path: str, collapse_level: Optional[int] = None, disable_content_checking: Optional[bool] = False):
    """
    Save objective to a json file.
    Create this function so that we don't need to worry about creating parent folders every time

    Parameters
    ----------
    obj: the objective to save
    path: the path to save
    collapse_level: set to any collapse value to prettify output json accordingly
    disable_content_checking: set to True to disable content checking within quotation marks.
        Content checking is used to protect text within quotation marks from being collapsed.
        Setting this to True will make the program run faster but may cause unexpected results.

    Returns
    -------
    None
    """
    file_dir = osp.dirname(osp.normpath(path))
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    json_obj = json.dumps(obj, indent=2, ensure_ascii=False)
    if collapse_level:
        json_obj = prettify_json(
            json_obj, collapse_level=collapse_level, disable_content_checking=disable_content_checking
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write(json_obj)

    return None


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

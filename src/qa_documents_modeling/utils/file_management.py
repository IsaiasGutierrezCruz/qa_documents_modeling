from pathlib import Path
from typing import Any

import joblib


def save_data(data: Any, path: Path) -> None:
    """Save data to a file."""
    with path.open("wb") as f:
        joblib.dump(data, f)


def load_data(path: Path) -> Any:
    """Load data from a file."""
    with path.open("rb") as f:
        return joblib.load(f)


def write_string_to_file(content: str, file_path: Path) -> None:
    """
    Write a string to a text file.

    Args:
        content (str): The string content to write
        file_path (str): The path where the file will be created/overwritten
    """
    with file_path.open("w", encoding="utf-8") as file:
        file.write(content)


def read_string_from_file(file_path: Path) -> str:
    """
    Read the contents of a text file and return it as a string.

    Args:
        file_path (str): The path to the text file to read

    Returns:
        str: The contents of the file as a string
    """
    with file_path.open("r", encoding="utf-8") as file:
        return file.read()

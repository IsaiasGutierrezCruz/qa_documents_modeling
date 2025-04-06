import re
from typing import Literal


def extract_question(text: str, data_set_type: Literal["rag", "raft"]) -> str:
    if data_set_type == "rag":
        regex = r"Query: (.*?)\nAnswer:"
    else:
        regex = r"Pregunta: (.*?)Respuesta:"
    match = re.search(regex, text)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"No match found for {data_set_type} data set")

def extract_context(text: str, data_set_type: Literal["rag", "raft"]) -> str:
    if data_set_type == "rag":
        regex = r"\n-{21}\n([\s\S]*?)\n-{21}\n"
    else:
        regex = r"<context>([\s\S]*?)<context>"
    match = re.search(regex, text)
    if match:
        return match.group(1)
    else: 
        raise ValueError(f"No match found for {data_set_type} data set")
from typing import Literal

import pandas as pd

from qa_documents_modeling.utils.utils import extract_question


def format_mistral_data_rag(
    data: list[dict], type: Literal["train", "test"]
) -> tuple[list[str], list[str]]:
    instructions = """Eres un sistema experto de preguntas y respuestas en el que se confía en todo el mundo.\nSiempre responde a la pregunta utilizando la información del contexto proporcionado, y no conocimiento previo.\nAlgunas reglas a seguir: \n1. Nunca hagas referencia directa al contexto proporcionado en tu respuesta.\n2. Evita afirmaciones como 'Basándote en el contexto, ...' o 'La información del contexto ...' o cualquier cosa similar.\n3. La respuesta debe estar escrita en Español."""

    example_template = (
        lambda instruction,
        context,
        response: f"""<s>[INST] {instruction} \n{context} \n[/INST]\n"""
        + response
        + "</s>"
    )
    example_test = (
        lambda instruction,
        context: f"""<s>[INST] {instruction} \n{context} \n[/INST]\n""" + "</s>"
    )

    questions = []
    examples_list = []
    for sample in data:
        question = extract_question(sample["messages"][1]["content"], "rag")
        questions.append(question)
        content = sample["messages"]

        context = content[1]["content"]
        context = context.replace(
            "Context information is below.",
            "La información del contexto está a continuación.",
        )

        context = context.replace(
            "Given the context information and not prior knowledge, answer the query.",
            "Considerando la información del contexto y sin conocimiento previo, responde a la pregunta.",
        )
        if type == "train":
            example = example_template(
                instruction=instructions,
                context=context,
                response=content[2]["content"],
            )
        else:
            example = example_test(
                instruction=instructions,
                context=context,
            )
        examples_list.append(example)

    return examples_list, questions


def format_mistral_data_raft(
    data: list[dict], type: Literal["train", "test"]
) -> tuple[list[str], list[str]]:

    example_template = (
        lambda instruction,
        context,
        response: f"""<s>[INST] {instruction} \n{context} \n[/INST]\n"""
        + response
        + "</s>"
    )
    example_test = (
        lambda instruction,
        context: f"""<s>[INST] {instruction} \n{context} \n[/INST]\n""" + "</s>"
    )

    questions = []
    raft_list = []
    for sample in data:
        question = extract_question(sample["messages"][1]["content"], "raft")
        questions.append(question)
        content = sample["messages"]
        if type == "train":
            example = example_template(
                instruction=content[0]["content"],
                context=content[1]["content"],
                response=content[2]["content"],
            )
        else:
            example = example_test(
                instruction=content[0]["content"],
                context=content[1]["content"],
            )
        raft_list.append(example)

    return raft_list, questions

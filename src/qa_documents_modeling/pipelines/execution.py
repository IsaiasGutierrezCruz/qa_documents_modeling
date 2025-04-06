from pathlib import Path
from typing import Literal

from hamilton import base, driver

from qa_documents_modeling.constants import PIPELINE_STEPS
from qa_documents_modeling.modeling_configs import MODELING_CONFIGS
from qa_documents_modeling.pipelines import modeling_pipeline


def execute_experiment(
    home: Path,
) -> dict:
    """
    Execute the pipeline steps.

    Parameters
    ----------
    home: Path
        The home directory.
    chosen_model: bool
        The chosen model.

    Returns
    -------
    dict
        The result of the pipeline steps.
    """
    initial_columns = {
        "cfg": MODELING_CONFIGS,
        "home": home,
        "upload_to_hf": True,
    }
    adapter = base.DefaultAdapter()
    dr = driver.Driver(
        {},
        modeling_pipeline,
        adapter=adapter,
    )
    return dr.execute(
        PIPELINE_STEPS,
        inputs=initial_columns,
    )

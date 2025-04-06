from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from llama_index.finetuning import OpenAIFinetuneEngine


class FineTuneService(ABC):
    @abstractmethod
    def finetune(self) -> Any:
        pass


class OpenAILlamaIndexFineTuneService(FineTuneService):
    def finetune(
        self, model: Literal["gpt-3.5-turbo"], golden_reference_train_path: Path,
        store_path: Path,
    ) -> None:
        if not store_path.is_file():
            finetune_engine = OpenAIFinetuneEngine(
                model,
                golden_reference_train_path,
            )
            finetune_engine.finetune()
            with store_path.open("w") as f:
                f.write(finetune_engine.get_finetuned_model())

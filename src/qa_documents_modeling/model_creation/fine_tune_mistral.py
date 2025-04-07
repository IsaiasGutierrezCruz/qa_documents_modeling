import os

from datasets import load_dataset
from huggingface_hub import login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from qa_documents_modeling.modeling_configs import MistralConfigs


class MistralModel:
    def __init__(self, model_base: str, pre_trained_model: str | None = None) -> None:
        self.model_base_name = model_base
        self.pre_trained_model = pre_trained_model

    def load_model(self) -> None:
        if self.pre_trained_model is not None:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_base_name,
                device_map="auto",
                trust_remote_code=False,
                revision="main",
            )

            self.config = PeftConfig.from_pretrained(self.pre_trained_model)
            self.model = PeftModel.from_pretrained(model, self.pre_trained_model)

            # load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_base_name, use_fast=True,
            )
        else:
            model_name = self.model_base_name
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                trust_remote_code=False,
                revision="main",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def ask_model(self, prompt: str) -> str:
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280,
        )
        return self.tokenizer.batch_decode(outputs)[0]


class FineTunneMistralModel:
    def __init__(
        self,
        model: str,
        data_set_url: str,
        tokenizer: AutoTokenizer,
        final_model_name: str,
    ) -> None:
        self.model = model
        self.data_set_url = data_set_url
        self.tokenizer = tokenizer
        self.final_model_name = final_model_name

    def tokenize_function(self, examples: str):
        # extract text
        text = examples["example"]
        # tokenize and truncate text
        self.tokenizer.truncation_side = "left"
        return self.tokenizer(text, return_tensors="np", truncation=True, max_length=2000)

    def fine_tune_model(self, configs: MistralConfigs) -> None:
        self.model.train()  # model in training mode (dropout modules are activated)
        # enable gradient check pointing
        self.model.gradient_checkpointing_enable()
        # enable quantized training
        self.model = prepare_model_for_kbit_training(self.model)
        # LoRA config
        config = LoraConfig(
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            target_modules=["q_proj"],
            lora_dropout=configs.lora_dropout,
            bias="none",
            task_type=configs.task_type,
        )
        # LoRA trainable version of model
        self.model = get_peft_model(self.model, config)
        # trainable parameter count
        self.model.print_trainable_parameters()
        self.data = load_dataset(self.data_set_url)
        self.tokenized_data = self.data.map(self.tokenize_function, batched=True)

        # setting pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # data collator
        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

        # define training arguments
        training_args = TrainingArguments(
            output_dir=self.final_model_name,
            learning_rate=configs.lr,
            per_device_train_batch_size=configs.batch_size,
            per_device_eval_batch_size=configs.batch_size,
            num_train_epochs=configs.num_epochs,
            weight_decay=0.01,
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            fp16=True,
            optim="paged_adamw_8bit",
        )
        # configure trainer
        self.trainer = Trainer(
            model=self.model,
            train_dataset=self.tokenized_data["train"],
            eval_dataset=self.tokenized_data["test"],
            args=training_args,
            data_collator=data_collator,
        )
        # train model
        self.model.config.use_cache = False  # silence the warnings
        self.trainer.train()
        # renable warnings
        self.model.config.use_cache = True
        login(os.getenv("HF_TOKEN"))
        self.model.push_to_hub(self.final_model_name)
        self.trainer.push_to_hub(self.final_model_name)

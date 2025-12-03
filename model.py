# ===================================================================
# STEP 1: SETUP AND INSTALLATION
# ===================================================================

# This command fixes a known issue on Kaggle with the 'protobuf' library.
!pip install protobuf==3.20.* --quiet

# Install all the necessary libraries for our project.
!pip install transformers datasets accelerate peft bitsandbytes trl rouge-score --quiet

# ===================================================================
# STEP 2: LOG IN TO HUGGING FACE
# ===================================================================

from huggingface_hub import notebook_login

# This will prompt you to paste your Hugging Face access token.
notebook_login()


# ===================================================================
# STEP 3: IMPORT ALL LIBRARIES
# ===================================================================

import os
import torch
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# Hugging Face Libraries
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import Dataset, DatasetDict

# PEFT Library for LoRA
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# Evaluation Libraries
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer


# ===================================================================
# STEP 4: DEFINE ALL CLASSES AND FUNCTIONS
# ===================================================================

# --- Function for LoRA Configuration ---
def get_lora_config() -> LoraConfig:
    """Creates and returns a LoraConfig for LLaMA 3.1."""
    return LoraConfig(
        r=16,  # Rank
        lora_alpha=32,  # Scaling factor
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

# --- Class for Data Processing ---
class DatasetProcessor:
    def __init__(self, model_name: str, max_length: int):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Tokenizer loaded. Pad token set to: {self.tokenizer.pad_token}")

    def load_dataset(self, file_path: str) -> Dataset:
        print(f"Loading dataset from {file_path}...")
        return Dataset.from_pandas(pd.read_csv(file_path))

    def format_conversation(self, example: Dict) -> Dict:
        system_prompt = "You are an empathetic assistant. Respond with empathy and understanding."
        # The dataset has 'Questions' and 'Answers' columns
        formatted_text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{example['Questions']} [/INST] {example['Answers']}</s>"
        return {"text": formatted_text}

    def tokenize_function(self, examples: Dict) -> Dict:
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )

    def preprocess_dataset(self, dataset_path: str, test_size: float = 0.1) -> DatasetDict:
        raw_dataset = self.load_dataset(dataset_path)
        formatted_dataset = raw_dataset.map(self.format_conversation)
        tokenized_dataset = formatted_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=formatted_dataset.column_names,
        )
        split_dataset = tokenized_dataset.train_test_split(test_size=test_size)
        print("Preprocessing complete!")
        print(f"Training examples: {len(split_dataset['train'])}")
        print(f"Test examples: {len(split_dataset['test'])}")
        return split_dataset

# --- Class for Fine-Tuning ---
class LLAMAFineTuner:
    def __init__(self, model_name: str, output_dir: str, logging_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        print(f"Initializing Fine-Tuner for model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._load_model()

    def _load_model(self):
        # Clear the cache to free up any memory from previous runs
        torch.cuda.empty_cache()
        
        print("Loading model with 8-bit quantization and CPU offloading...")
        # --- THIS IS THE KEY CHANGE FOR MEMORY SAVING ---
        # We are switching to 8-bit quantization and enabling CPU offloading.
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  # Use 8-bit quantization
            llm_int8_enable_fp32_cpu_offload=True, # Offload parts of the model to CPU RAM
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,
        )
        self.model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA
        print("Applying LoRA adapters...")
        lora_config = get_lora_config()
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        self.model.gradient_checkpointing_enable()

    def fine_tune(self, train_dataset, eval_dataset, num_epochs, batch_size, experiment_name):
        experiment_dir = os.path.join(self.output_dir, experiment_name)
        training_args = TrainingArguments(
            output_dir=experiment_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8, # Increase gradient accumulation to simulate a larger batch size
            learning_rate=2e-4,
            num_train_epochs=num_epochs,
            logging_steps=25, # Log more frequently to see progress
            save_steps=500,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=500,
            fp16=True,
            bf16=False,
            report_to="none",
            load_best_model_at_end=True,
        )
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        print("Starting training...")
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        print(f"Training completed in {end_time - start_time:.2f} seconds.")
        trainer.save_model(experiment_dir)
        return experiment_dir

# --- Class for Evaluation ---
class Evaluator:
    def __init__(self, model_path: str, tokenizer_path: str):
        print(f"Initializing Evaluator for model at: {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_responses(self, test_dataset: Dataset, num_samples: int = -1):
        print(f"Generating responses for {min(num_samples, len(test_dataset)) if num_samples > 0 else len(test_dataset)} examples...")
        generated_responses = []
        
        # If num_samples is -1, use the whole dataset. Otherwise, use a subset.
        dataset_to_eval = test_dataset.select(range(min(num_samples, len(test_dataset)))) if num_samples > 0 else test_dataset

        for example in dataset_to_eval:
            system_prompt = "You are an empathetic assistant. Respond with empathy and understanding."
            prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{example['Questions']} [/INST]"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=self.tokenizer.eos_token_id)
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
            generated_responses.append({
                "input": example['Questions'],
                "reference": example['Answers'],
                "generated": response_text
            })
        return generated_responses

# --- Main Execution Function ---
def main():
    # --- CONFIGURATION ---
    DATASET_PATH = "/kaggle/input/d/raseluddin/bengali-empathetic-conversations-corpus/BengaliEmpatheticConversationsCorpus .csv"
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    EXPERIMENT_NAME = "bengali_empathy_run"
    OUTPUT_DIR = "./results"
    LOGGING_DIR = "./logs"
    
    # --- PIPELINE ---
    print("--- Starting LLaMA 3.1 Fine-tuning Pipeline ---")
    
    # 1. Preprocess Data
    processor = DatasetProcessor(model_name=MODEL_NAME, max_length=512)
    dataset_dict = processor.preprocess_dataset(dataset_path=DATASET_PATH, test_size=0.1)
    
    # 2. Fine-tune Model
    finetuner = LLAMAFineTuner(model_name=MODEL_NAME, output_dir=OUTPUT_DIR, logging_dir=LOGGING_DIR)
    model_save_path = finetuner.fine_tune(
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['test'],
        num_epochs=3,
        batch_size=1,
        experiment_name=EXPERIMENT_NAME
    )
    
    # 3. Evaluate Model
    print("\n--- Starting Evaluation ---")
    evaluator = Evaluator(model_path=model_save_path, tokenizer_path=model_save_path)
    
    # Load original, non-tokenized test data for evaluation
    raw_test_dataset = processor.load_dataset(DATASET_PATH).train_test_split(test_size=0.1)['test']
    
    # Generate responses for a subset of 20 examples for quick evaluation
    generated_responses = evaluator.generate_responses(raw_test_dataset, num_samples=20)
    
    print("\n--- Sample Generated Responses ---")
    for i, resp in enumerate(generated_responses[:5]):
        print(f"\n--- Example {i+1} ---")
        print(f"Input: {resp['input']}")
        print(f"Generated: {resp['generated']}")
        print(f"Reference: {resp['reference']}")
    
    print("\n--- Pipeline Completed Successfully ---")


# ===================================================================
# STEP 5: RUN THE MAIN FUNCTION
# ===================================================================

if __name__ == "__main__":
    main()
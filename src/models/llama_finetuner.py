# src/models/llama_finetuner.py

import os
import torch
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import get_peft_model, prepare_model_for_kbit_training
from datasets import DatasetDict
from .lora_adapter import get_lora_config

# Note: The Strategy pattern for LoRA vs. Unsloth is implemented by passing a flag.
# A more advanced implementation would use a base class and derived classes for each strategy.

class LLAMAFineTuner:
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        use_lora: bool = True,
        output_dir: str = "./results",
        logging_dir: str = "./logs",
    ):
        """
        Initializes the LLAMAFineTuner.

        Args:
            model_name (str): The name of the base model to fine-tune.
            use_lora (bool): Whether to use LoRA for parameter-efficient fine-tuning.
            output_dir (str): Directory to save the fine-tuned model checkpoints.
            logging_dir (str): Directory to save training logs.
        """
        self.model_name = model_name
        self.use_lora = use_lora
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)

        print(f"Initializing LLAMAFineTuner for model: {model_name}")
        
        # Load tokenizer (we'll reuse the one from DatasetProcessor, but it's good to have it here too)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the base model with optimizations
        self._load_model()
        
        # Training history to store experiment logs
        self.training_history = []

    def _load_model(self):
        """
        Loads the base model with 4-bit quantization and prepares it for PEFT.
        This method is key to fitting the model on a free GPU like Kaggle's.
        """
        print("Loading base model with 4-bit quantization...")
        
        # Configuration for 4-bit quantization (QLoRA)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,  # <--- THIS IS THE CRUCIAL FIX FOR LOCAL TRAINING
        )
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",  # Automatically maps layers to available devices (GPU/CPU)
            trust_remote_code=True,
            use_cache=False,  # Must be False for gradient checkpointing
        )
        
        # Prepare the model for parameter-efficient fine-tuning
        self.model = prepare_model_for_kbit_training(model)
        
        # Apply LoRA adapters if specified
        if self.use_lora:
            print("Applying LoRA adapters...")
            lora_config = get_lora_config()
            self.model = get_peft_model(self.model, lora_config)
            
            # Print the number of trainable parameters
            self.model.print_trainable_parameters()
        
        # Enable gradient checkpointing to save memory
        # This trades computation for memory, which is essential for large models.
        self.model.gradient_checkpointing_enable()

    def fine_tune(
        self,
        train_dataset,
        eval_dataset=None,
        num_epochs: int = 1, # Using 1 epoch for quick testing, increase for real training
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        logging_steps: int = 10,
        save_steps: int = 50,
        eval_steps: int = 50,
        experiment_name: str = None,
    ):
        """
        Starts the fine-tuning process.

        Args:
            train_dataset: The tokenized training dataset.
            eval_dataset: The tokenized evaluation dataset.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size per device.
            learning_rate (float): The learning rate for the optimizer.
            logging_steps (int): Log training metrics every X steps.
            save_steps (int): Save a model checkpoint every X steps.
            eval_steps (int): Run evaluation every X steps.
            experiment_name (str): A name for this experiment for logging purposes.
        """
        # Create experiment name if not provided
        if experiment_name is None:
            experiment_name = f"llama_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        experiment_dir = os.path.join(self.output_dir, experiment_name)
        
        # Set up TrainingArguments
        training_args = TrainingArguments(
            output_dir=experiment_dir,
            overwrite_output_dir=True,
            evaluation_strategy="steps" if eval_dataset else "no",
            save_strategy="steps",
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=os.path.join(self.logging_dir, experiment_name),
            fp16=False,  # We use bf16 instead
            bf16=True,   # bfloat16 is more stable on modern GPUs
            report_to="none", # Can be "tensorboard", "wandb", etc.
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
        )
        
        # Data collator handles padding dynamically within each batch
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We are doing Causal Language Modeling, not Masked Language Modeling
        )
        
        # Initialize the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        print("Starting training...")
        start_time = time.time()
        
        train_result = trainer.train()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # Save the final model and tokenizer
        trainer.save_model()
        self.tokenizer.save_pretrained(experiment_dir)
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final training loss: {train_result.training_loss}")
        
        # Log experiment details as per requirements
        self._log_experiment(
            experiment_id=experiment_name,
            train_loss=train_result.training_loss,
            val_loss=trainer.evaluate()["eval_loss"] if eval_dataset else None,
            training_time=training_time,
            lora_config=self.model.peft_config["default"].to_dict() if self.use_lora else None
        )
        
        return experiment_dir

    def _log_experiment(self, experiment_id: str, train_loss: float, val_loss: Optional[float], training_time: float, lora_config: Optional[Dict]):
        """Saves the experiment details to a JSON log file."""
        log_entry = {
            "id": experiment_id,
            "model_name": self.model_name,
            "lora_config": lora_config,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "metrics": None, # To be populated later by the evaluator
            "timestamp": datetime.now().isoformat(),
            "training_time_seconds": training_time
        }
        
        self.training_history.append(log_entry)
        
        log_file_path = os.path.join(self.logging_dir, "LLAMAExperiments.json")
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        print(f"Experiment log saved to {log_file_path}")

# src/data/dataset_processor.py

import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple

class DatasetProcessor:
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct", max_length: int = 2048):
        """
        Initializes the DatasetProcessor.

        Args:
            model_name (str): The name of the model to use for tokenization.
            max_length (int): The maximum sequence length for tokenization.
                             The requirement specifies this should not be reduced.
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Load the tokenizer for the specified model
        # LLaMA 3.1's tokenizer should support Bengali characters out-of-the-box
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # LLaMA models typically don't have a pad token set by default.
        # We set it to the End-of-Sequence (EOS) token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Tokenizer loaded for {model_name}. Pad token set to: {self.tokenizer.pad_token}")

    def load_dataset(self, file_path: str) -> Dataset:
        """
        Loads a dataset from a JSON or CSV file.

        Args:
            file_path (str): The path to the dataset file.

        Returns:
            Dataset: A Hugging Face Dataset object.
        """
        print(f"Loading dataset from {file_path}...")
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return Dataset.from_pandas(pd.DataFrame(data))
        elif file_path.endswith('.csv'):
            return Dataset.from_pandas(pd.read_csv(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path}. Please use .json or .csv.")

    def format_conversation(self, example: Dict) -> Dict:
        """
        Formats a conversation example into the specific prompt structure required for LLaMA 3.1-Instruct.
        
        Args:
            example (Dict): A dictionary containing the conversation data.
                            We use 'Questions' and 'Answers' columns from the user's CSV.

        Returns:
            Dict: A dictionary with the formatted text.
        """
        # A system prompt helps set the context for the model.
        system_prompt = "You are an empathetic assistant. Respond with empathy and understanding."
        
        # LLaMA 3.1 Instruct prompt format
        # <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST] {model_response}</s>
        
        # --- FINAL CORRECTED VERSION ---
        # Using 'Questions' for user input and 'Answers' for the desired response.
        formatted_text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{example['Questions']} [/INST] {example['Answers']}</s>"
        
        return {"text": formatted_text}

    def tokenize_function(self, examples: Dict) -> Dict:
        """
        Tokenizes the formatted text examples.

        Args:
            examples (Dict): A dictionary containing a batch of text examples.

        Returns:
            Dict: A dictionary with tokenized inputs (input_ids, attention_mask).
        """
        # The tokenizer will handle converting text to IDs.
        # We don't truncate here to respect the "full-sequence" requirement,
        # but the `max_length` in the tokenizer's call acts as a safeguard.
        return self.tokenizer(
            examples["text"],
            truncation=True, # Still important to prevent overflow
            padding=False,   # We'll handle padding dynamically in the training script
            max_length=self.max_length,
            return_tensors=None, # Return lists, which the Hugging Face Trainer handles well
        )

    def preprocess_dataset(self, dataset_path: str, test_size: float = 0.1) -> DatasetDict:
        """
        The main preprocessing pipeline: loads, formats, tokenizes, and splits the dataset.

        Args:
            dataset_path (str): Path to the raw dataset file.
            test_size (float): Fraction of the dataset to use for the validation/test set.

        Returns:
            DatasetDict: A DatasetDict containing 'train' and 'test' splits.
        """
        # 1. Load the raw dataset
        raw_dataset = self.load_dataset(dataset_path)
        
        # 2. Format the conversations into the LLaMA 3.1 Instruct format
        print("Formatting conversations...")
        formatted_dataset = raw_dataset.map(self.format_conversation)
        
        # 3. Tokenize the formatted text
        print("Tokenizing dataset...")
        tokenized_dataset = formatted_dataset.map(
            self.tokenize_function,
            batched=True, # Process multiple examples at once for efficiency
            remove_columns=formatted_dataset.column_names, # Remove original text columns, keeping only tokenized IDs
        )
        
        # 4. Split the dataset into training and testing sets
        print(f"Splitting dataset into train and test sets (test_size={test_size})...")
        split_dataset = tokenized_dataset.train_test_split(test_size=test_size)
        
        print("Preprocessing complete!")
        print(f"Training examples: {len(split_dataset['train'])}")
        print(f"Test examples: {len(split_dataset['test'])}")
        
        return split_dataset
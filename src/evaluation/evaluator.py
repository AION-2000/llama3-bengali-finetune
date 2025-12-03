# src/evaluation/evaluator.py

import os
import json
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from .metrics import compute_perplexity, compute_bleu, compute_rouge

class Evaluator:
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./evaluation_results",
        logging_dir: str = "./logs",
    ):
        """
        Initializes the Evaluator.

        Args:
            model_path (str): Path to the fine-tuned model directory.
            output_dir (str): Directory to save evaluation results.
            logging_dir (str): Directory to save logs.
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.logging_dir = logging_dir
        
        # Ensure output directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        
        print(f"Initializing Evaluator for model at: {model_path}")
        
        # Load the fine-tuned model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # This is where we will store generated responses
        self.generated_responses_log = []

    def generate_responses(
        self,
        test_dataset: Dataset,
        max_new_tokens: int = 256,
        batch_size: int = 4,
    ) -> List[Dict]:
        """
        Generates responses for the test dataset.
        """
        print(f"Generating responses for {len(test_dataset)} examples...")
        
        generated_responses = []
        
        for i in range(0, len(test_dataset), batch_size):
            batch = test_dataset[i:i+batch_size]
            
            # Format prompts for the model
            system_prompt = "You are an empathetic assistant. Respond with empathy and understanding."
            prompts = [
                f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{example['context']} [/INST]"
                for example in batch
            ]
            
            # Tokenize
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode and extract only the generated part
            for j, output in enumerate(outputs):
                full_response = self.tokenizer.decode(output, skip_special_tokens=True)
                # The response comes after the [/INST] token
                response_text = full_response.split("[/INST]")[-1].strip()
                
                generated_responses.append({
                    "input_text": batch[j]['context'],
                    "reference_text": batch[j]['response'],
                    "response_text": response_text,
                    "timestamp": datetime.now().isoformat(),
                })
        
        print("Response generation complete.")
        return generated_responses

    def evaluate_automatic_metrics(self, test_dataset: Dataset) -> Dict:
        """
        Evaluates the model using automatic metrics (Perplexity, BLEU, ROUGE).
        """
        print("Computing automatic evaluation metrics...")
        
        # 1. Generate responses
        generated_responses = self.generate_responses(test_dataset)
        
        # 2. Log generated responses as per requirements
        self._log_generated_responses(generated_responses)
        
        # 3. Extract texts for metric calculation
        reference_texts = [resp["reference_text"] for resp in generated_responses]
        generated_texts = [resp["response_text"] for resp in generated_responses]
        
        # 4. Compute metrics
        perplexity = compute_perplexity(self.model, self.tokenizer, test_dataset)
        bleu_scores = compute_bleu(reference_texts, generated_texts)
        rouge_scores = compute_rouge(reference_texts, generated_texts)
        
        metrics = {
            "perplexity": perplexity,
            "bleu": bleu_scores,
            "rouge": rouge_scores,
        }
        
        print("Automatic metrics computed.")
        return metrics

    def prepare_human_evaluation(self, generated_responses: List[Dict], num_samples: int = 50) -> str:
        """
        Prepares a CSV file for human evaluation.
        """
        print(f"Preparing human evaluation template with {num_samples} samples...")
        
        # Sample responses
        if len(generated_responses) > num_samples:
            sampled_responses = generated_responses[:num_samples] # Simple sampling
        else:
            sampled_responses = generated_responses
            
        # Create a DataFrame for the evaluation template
        evaluation_data = []
        for i, resp in enumerate(sampled_responses):
            evaluation_data.append({
                "Sample ID": i + 1,
                "Input Text": resp["input_text"],
                "Reference Response": resp["reference_text"],
                "Generated Response": resp["response_text"],
                "Empathy Score (1-5)": "",
                "Relevance Score (1-5)": "",
                "Fluency Score (1-5)": "",
                "Overall Quality Score (1-5)": "",
                "Comments": "",
            })
        
        df = pd.DataFrame(evaluation_data)
        
        # Save to CSV
        output_file = os.path.join(self.output_dir, "human_evaluation_template.csv")
        df.to_csv(output_file, index=False)
        
        print(f"Human evaluation template saved to {output_file}")
        return output_file

    def _log_generated_responses(self, responses: List[Dict]):
        """
        Logs generated responses to a JSON file, as required by the project spec.
        """
        # Add to existing log
        self.generated_responses_log.extend(responses)
        
        # Save to file
        log_file_path = os.path.join(self.logging_dir, "GeneratedResponses.json")
        with open(log_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.generated_responses_log, f, indent=2, ensure_ascii=False)
        
        print(f"Generated responses logged to {log_file_path}")
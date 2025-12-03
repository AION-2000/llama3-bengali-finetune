# src/main.py

import os
import argparse
import json
from datetime import datetime

# Import our custom classes
from data.dataset_processor import DatasetProcessor
from models.llama_finetuner import LLAMAFineTuner
from evaluation.evaluator import Evaluator

def main():
    """Main function to run the fine-tuning and evaluation pipeline."""
    
    # Set up argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 3.1 on Bengali Empathetic Conversations")
    
    # --- Dataset Arguments ---
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        required=True, 
        help="Path to the raw dataset file (JSON or CSV). Must have 'context' and 'response' columns."
    )
    parser.add_argument("--test_size", type=float, default=0.1, help="Fraction of dataset for testing.")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length for tokenization.")
    
    # --- Model & Fine-tuning Arguments ---
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="meta-llama/Llama-3.1-8B-Instruct", 
        help="Base model name from Hugging Face Hub."
    )
    parser.add_argument("--experiment_name", type=str, default=None, help="A custom name for this experiment.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate.")
    
    # --- Output Directory Arguments ---
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save fine-tuned models.")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory to save logs.")
    parser.add_argument("--eval_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results.")
    
    args = parser.parse_args()

    # --- Create experiment name if not provided ---
    if args.experiment_name is None:
        args.experiment_name = f"llama_bengali_empathy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)

    print("--- Starting LLaMA 3.1 Fine-tuning Pipeline ---")
    print(f"Experiment Name: {args.experiment_name}")

    # --- 1. Preprocess Dataset ---
    print("\n--- Step 1: Preprocessing Dataset ---")
    processor = DatasetProcessor(model_name=args.model_name, max_length=args.max_length)
    dataset_dict = processor.preprocess_dataset(dataset_path=args.dataset_path, test_size=args.test_size)
    
    train_dataset = dataset_dict['train']
    test_dataset = dataset_dict['test']

    # --- 2. Fine-tune Model ---
    print("\n--- Step 2: Fine-tuning Model ---")
    finetuner = LLAMAFineTuner(
        model_name=args.model_name,
        use_lora=True,  # Using LoRA as per requirements
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
    )
    
    model_save_path = finetuner.fine_tune(
        train_dataset=train_dataset,
        eval_dataset=test_dataset, # Using test set for validation during training
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        experiment_name=args.experiment_name,
    )
    
    print(f"\nModel fine-tuning complete. Model saved to: {model_save_path}")

    # --- 3. Evaluate Model ---
    print("\n--- Step 3: Evaluating Model ---")
    evaluator = Evaluator(
        model_path=model_save_path,
        output_dir=args.eval_dir,
        logging_dir=args.logging_dir,
    )
    
    # We need the original, non-tokenized test data for evaluation
    # Let's reload the raw dataset and split it to get the original text
    raw_dataset = processor.load_dataset(args.dataset_path)
    raw_split_dataset = raw_dataset.train_test_split(test_size=args.test_size)
    raw_test_dataset = raw_split_dataset['test']
    
    # Run automatic evaluation
    metrics = evaluator.evaluate_automatic_metrics(test_dataset=raw_test_dataset)
    
    print("\n--- Automatic Evaluation Results ---")
    print(json.dumps(metrics, indent=2))
    
    # Prepare for human evaluation
    human_eval_file = evaluator.prepare_human_evaluation(
        generated_responses=evaluator.generated_responses_log,
        num_samples=20, # Create a template for 20 samples
    )
    
    print(f"\nHuman evaluation template created at: {human_eval_file}")
    
    # --- 4. Finalize Logs ---
    # Update the main experiment log with evaluation metrics
    log_file_path = os.path.join(args.logging_dir, "LLAMAExperiments.json")
    with open(log_file_path, 'r+', encoding='utf-8') as f:
        logs = json.load(f)
        # Find the correct experiment log and update it
        for log in logs:
            if log['id'] == args.experiment_name:
                log['metrics'] = metrics
                break
        f.seek(0)
        json.dump(logs, f, indent=2, ensure_ascii=False)
        f.truncate()

    print("\n--- Pipeline Completed Successfully ---")

if __name__ == "__main__":
    main()
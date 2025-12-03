# src/evaluation/metrics.py

import torch
import numpy as np
from typing import List, Dict
from transformers import PreTrainedModel, PreTrainedTokenizer
from datasets import Dataset
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer

def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    batch_size: int = 4,
) -> float:
    """
    Computes the perplexity of the model on a given dataset.
    Perplexity is a common metric for language models, where lower is better.
    """
    model.eval()
    total_loss = 0.0
    
    # We use a simple data loader for evaluation
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        
        # The dataset should already be tokenized and have 'input_ids'
        inputs = torch.tensor(batch['input_ids']).to(model.device)
        
        with torch.no_grad():
            outputs = model(input_ids=inputs, labels=inputs)
            loss = outputs.loss
            
        total_loss += loss.item()
    
    avg_loss = total_loss / (len(dataset) / batch_size)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

def compute_bleu(reference_texts: List[str], generated_texts: List[str]) -> Dict[str, float]:
    """
    Computes BLEU scores between reference and generated texts.
    BLEU measures the precision of n-grams in the generated text.
    """
    # Tokenize texts for BLEU calculation
    reference_tokens = [[ref.split()] for ref in reference_texts]
    generated_tokens = [gen.split() for gen in generated_texts]
    
    # Compute BLEU scores for different n-gram weights
    bleu_1 = corpus_bleu(reference_tokens, generated_tokens, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(reference_tokens, generated_tokens, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(reference_tokens, generated_tokens, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(reference_tokens, generated_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    
    return {
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_3": bleu_3,
        "bleu_4": bleu_4,
    }

def compute_rouge(reference_texts: List[str], generated_texts: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Computes ROUGE scores between reference and generated texts.
    ROUGE measures the recall of n-grams, focusing on content overlap.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = {
        "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
        "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
    }
    
    for ref, gen in zip(reference_texts, generated_texts):
        scores = scorer.score(ref, gen)
        
        for rouge_type in rouge_scores:
            for metric in ["precision", "recall", "fmeasure"]:
                rouge_scores[rouge_type][metric] += scores[rouge_type]._asdict()[metric]
    
    # Average the scores over all examples
    num_texts = len(reference_texts)
    for rouge_type in rouge_scores:
        for metric in ["precision", "recall", "fmeasure"]:
            rouge_scores[rouge_type][metric] /= num_texts
    
    return rouge_scores
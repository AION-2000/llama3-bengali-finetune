# src/models/lora_adapter.py

from peft import LoraConfig, TaskType

def get_lora_config() -> LoraConfig:
    """
    Creates and returns a LoraConfig for LLaMA 3.1.
    
    This configuration targets the attention layers of the model, which is a common
    and effective strategy for fine-tuning.
    """
    return LoraConfig(
        r=16,  # Rank of the update matrices. Higher rank means more parameters.
        lora_alpha=32,  # A scaling factor. A common practice is to set alpha to 2*r.
        target_modules=[  # The modules (layers) to apply the LoRA adapters to.
            "q_proj",    # Query projection
            "k_proj",    # Key projection
            "v_proj",    # Value projection
            "o_proj",    # Output projection
            "gate_proj", # Used in the Feed-Forward network
            "up_proj",   # Used in the Feed-Forward network
            "down_proj", # Used in the Feed-Forward network
        ],
        lora_dropout=0.1,  # Dropout probability for LoRA layers.
        bias="none",       # Specifies if the bias parameters should be trained.
        task_type=TaskType.CAUSAL_LM, # The type of task, which is Causal Language Modeling.
    )
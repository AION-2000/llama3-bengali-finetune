# Bengali Empathetic LLaMA 3.1

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-FFD700?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)

A specialized empathetic chatbot for Bengali language interactions, built by fine-tuning Meta's LLaMA 3.1-8B-Instruct model on a curated corpus of empathetic conversations.

## Overview

This project addresses the critical need for culturally and linguistically appropriate AI conversational agents in Bengali. By leveraging state-of-the-art large language model fine-tuning techniques, we have developed a model that understands and responds with appropriate empathy to Bengali user inputs, making AI assistance more accessible and effective for Bengali-speaking communities.

### Key Capabilities

- **Empathetic Response Generation**: Trained specifically on empathetic conversation patterns in Bengali
- **Parameter-Efficient Training**: Utilizes LoRA (Low-Rank Adaptation) for resource-efficient fine-tuning
- **Memory-Optimized**: Implements 4-bit quantization (QLoRA) for deployment on consumer hardware
- **Comprehensive Evaluation**: Includes both automated metrics and human evaluation frameworks

## Technical Specifications

| Component | Specification |
|-----------|--------------|
| Base Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Fine-tuning Method | LoRA (Low-Rank Adaptation) |
| Quantization | 4-bit (QLoRA) |
| Training Dataset | [Bengali Empathetic Conversations Corpus](https://www.kaggle.com/datasets/raseluddin/bengali-empathetic-conversations-corpus) |
| Evaluation Metrics | BLEU, ROUGE, Human Evaluation |
| Architecture | Modular OOP design |

## Repository Structure

```
llama3-bengali-finetune/
│
├── model.py                    # Core training and evaluation pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # License information
├── .gitignore                  # Git ignore rules
│
├── results/                    # Model outputs (generated)
│   └── adapter_model.safetensors
│
├── logs/                       # Training logs (generated)
│   ├── LLAMAExperiments.json
│   └── GeneratedResponses.json
│
└── evaluation_results/         # Evaluation data (generated)
    └── human_evaluation_template.csv
```

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Git**: For version control
- **Hugging Face Account**: Required for LLaMA 3.1 model access
- **CUDA-compatible GPU**: Recommended for training (optional with appropriate configuration)

### Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/AION-2000/llama3-bengali-finetune.git
cd llama3-bengali-finetune
```

2. **Create and Activate Virtual Environment**

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
```
torch>=2.0.0
transformers>=4.40.0
datasets
accelerate
peft
bitsandbytes
trl
rouge-score
pandas
numpy
```

4. **Authenticate with Hugging Face**

```bash
huggingface-cli login
```

Enter your Hugging Face access token when prompted. You can generate a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Usage

### Basic Training

Execute the main training script to begin fine-tuning:

```bash
python model.py
```

### Configuration

Before running, ensure the dataset path is correctly configured in `model.py`:

```python
DATASET_PATH = "path/to/bengali-empathetic-conversations-corpus"
```

### Training Pipeline

The script executes the following workflow:

1. **Data Loading**: Loads and preprocesses the Bengali empathetic conversations dataset
2. **Model Initialization**: Configures LLaMA 3.1 with LoRA adapters and 4-bit quantization
3. **Fine-tuning**: Trains the model on empathetic response generation
4. **Evaluation**: Assesses performance using automated metrics
5. **Output Generation**: Saves model checkpoints, logs, and sample responses

## Evaluation

### Automated Metrics

The model is evaluated using standard NLG metrics:

- **BLEU**: Measures n-gram overlap with reference responses
- **ROUGE**: Evaluates recall-oriented summarization quality
- **Perplexity**: Assesses language modeling performance

### Human Evaluation

A structured template for human evaluation is provided at `evaluation_results/human_evaluation_template.csv`, enabling qualitative assessment of:

- Response relevance
- Empathy appropriateness
- Linguistic fluency
- Cultural sensitivity

## Output Artifacts

### Model Checkpoints
- **Location**: `results/`
- **Files**: `adapter_model.safetensors`, `adapter_config.json`
- **Description**: LoRA adapter weights for the fine-tuned model

### Training Logs
- **Location**: `logs/`
- **Files**: 
  - `LLAMAExperiments.json`: Training metrics and hyperparameters
  - `GeneratedResponses.json`: Sample model outputs during training

### Evaluation Results
- **Location**: `evaluation_results/`
- **Files**: `human_evaluation_template.csv`
- **Description**: Framework for conducting human evaluation studies

## Contributing

We welcome contributions from the community! To contribute:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/EnhancedEmpathy`)
3. **Commit** your changes (`git commit -m 'Add enhanced empathy detection'`)
4. **Push** to the branch (`git push origin feature/EnhancedEmpathy`)
5. **Open** a Pull Request with a clear description of your changes

### Development Guidelines

- Follow PEP 8 style guidelines for Python code
- Add appropriate documentation for new features
- Include unit tests where applicable
- Update README.md to reflect significant changes

## Citation

If you use this project in your research, please cite:

```bibtex
@software{bengali_empathetic_llama,
  author = {Shihab Shahriar Aion},
  title = {Bengali Empathetic LLaMA 3.1: Fine-tuning for Culturally-Aware Conversational AI},
  year = {2024},
  url = {https://github.com/AION-2000/llama3-bengali-finetune}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for complete details.

## Acknowledgments

- **Meta AI** for developing and releasing the LLaMA 3.1 model family
- **Rasel Uddin** for curating the [Bengali Empathetic Conversations Corpus](https://www.kaggle.com/datasets/raseluddin/bengali-empathetic-conversations-corpus)
- **Hugging Face** for providing the transformers library and model hosting infrastructure
- The **Bengali NLP community** for ongoing research and resource development

## Contact

For questions, suggestions, or collaboration opportunities, please:

- Open an issue on GitHub
- Contact the maintainer at [aionshihabshahriar@gmail.com]
- Join our discussion forum at Whatsapp:01959040057

---

**Note**: This project requires acceptance of Meta's LLaMA license agreement. Ensure compliance with all applicable terms before deployment. 

# Mechanistic Interpretability of Open Source LLMs

Mechanistic Interpretability of Open Source LLMs: reproduction of "On the Biology of a Large Language Model" (Lindsey, J., Gurnee, W., et al., 2025). Investigation of mechanisms in an open model, Qwen3-4B-Instruct. DAMPT, University of Cambridge.

## Overview

This project implements a mechanistic interpretability pipeline for analyzing Qwen3-4B, following the methodology from Anthropic's research on the biology of language models. The pipeline:

1. **Runs Qwen3-4B on prompt sets** for specific behaviors (factual recall, reasoning, code generation, multilingual)
2. **Captures activations** from a subset of transformer layers
3. **Trains Sparse Autoencoders (SAEs)** to obtain interpretable features
4. **Builds pruned dependency graphs** from inputs through SAE features to decisive logits
5. **Validates with interventions** using inhibition and swap-in style experiments

## Installation

```bash
# Clone the repository
git clone https://github.com/vitjuli/Mechanistic-Interpretability-of-Open-Source-LLMs.git
cd Mechanistic-Interpretability-of-Open-Source-LLMs

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Full Pipeline

```bash
# Run for a specific behavior
python src/run_pipeline.py --behavior factual_recall

# Run for all behaviors
python src/run_pipeline.py --behavior all

# Customize layers and SAE settings
python src/run_pipeline.py --behavior reasoning --layers 8,16,24 --sae-hidden-dim 2048
```

### Command Line Options

- `--behavior`: Behavior to analyze (`factual_recall`, `reasoning`, `code_generation`, `multilingual`, `all`)
- `--device`: Device to run on (`cuda` or `cpu`)
- `--output-dir`: Output directory for results
- `--layers`: Comma-separated layer indices to analyze
- `--sae-hidden-dim`: Hidden dimension for sparse autoencoders
- `--sae-epochs`: Number of training epochs for SAEs

### Using Individual Components

```python
from src.mi_pipeline import (
    Config,
    ActivationCapture,
    SparseAutoencoder,
    DependencyGraph,
    InterventionValidator,
)

# Configure the pipeline
config = Config(
    target_layers=(8, 16, 24, 31),
    sae_hidden_dim=4096,
)

# Capture activations
capturer = ActivationCapture(config)
capturer.load_model()
activations = capturer.capture(prompts)

# Train SAE
from src.mi_pipeline.sparse_autoencoder import SAETrainer
sae = SparseAutoencoder(input_dim=activations[8].shape[-1], hidden_dim=4096)
trainer = SAETrainer(sae, config)
trainer.train(activations[8])

# Build dependency graph
graph = DependencyGraph(config)
graph.build_from_activations(tokens, activations, features, logits)
pruned_graph = graph.prune()

# Validate with interventions
validator = InterventionValidator(model, saes, config)
result = validator.validate_inhibition(input_ids, layer=8, feature_indices=[42])
```

## Project Structure

```
├── src/
│   ├── mi_pipeline/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration settings
│   │   ├── activation_capture.py  # Activation extraction
│   │   ├── sparse_autoencoder.py  # SAE implementation
│   │   ├── dependency_graph.py    # Graph construction
│   │   └── interventions.py       # Validation experiments
│   └── run_pipeline.py            # Main pipeline script
├── prompts/
│   ├── __init__.py
│   └── behavior_prompts.py        # Behavior-specific prompts
├── tests/
│   ├── test_config.py
│   ├── test_sparse_autoencoder.py
│   ├── test_dependency_graph.py
│   └── test_prompts.py
├── outputs/                       # Generated outputs
│   ├── activations/
│   ├── sae_models/
│   └── graphs/
├── requirements.txt
└── README.md
```

## Behavior Categories

- **Factual Recall**: Tests knowledge retrieval (e.g., "The capital of France is")
- **Reasoning**: Tests logical reasoning (e.g., "If A > B and B > C, then A is")
- **Code Generation**: Tests programming ability (e.g., Fibonacci implementation)
- **Multilingual**: Tests cross-lingual ability (e.g., translations)

## Running Tests

```bash
pytest tests/ -v
```

## Reference

Lindsey, J., Gurnee, W., et al. (2025). On the Biology of a Large Language Model. Anthropic Transformer.

## License

MIT License

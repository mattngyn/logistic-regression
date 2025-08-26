# Self-Description vs Actual Behavior Prompt Classification

This experiment tests whether LoRA adapters can distinguish between two types of prompts:

1. **Self-description prompts**: Asking the model to describe its learned behaviors (e.g., "Do you prefer risk or safety?")
2. **Actual behavior prompts**: Asking the model to make a specific choice (e.g., "Choose between A: $50 or B: 50% chance of $100")

## Key Insight

Following the Behavioral Self-Awareness paper, we classify **prompts only** - not responses. The goal is to see if the model's LoRA adapters activate differently when asked about its preferences vs when asked to make an actual choice.

## Quick Start

```bash
# Run the complete experiment
python run_prompt_classification_experiment.py
```

## What It Does

1. **Creates Dataset**:
   - Takes prompts from `ft_risky_AB.jsonl` (all actual behavior examples)
   - Generates self-description prompts using GPT-4 and manual examples
   - Creates balanced dataset with just prompts and labels

2. **Extracts Features**:
   - LoRA scalar values from layers 14-30
   - KL divergence between base and finetuned model
   - Averages across sequence length for prompt-level representation

3. **Runs Classification**:
   - 100 logistic regressions with L1 regularization
   - Identifies which layers distinguish prompt types
   - Reports accuracy, precision, recall, F1, AUC-ROC

4. **Validates Findings** (optional):
   - Steering experiments to confirm layer specialization
   - Tests if identified layers actually control prompt interpretation

## Files

- `create_prompt_classification_dataset.py` - Creates the dataset
- `prompt_context_probing.py` - Main probing experiment
- `behavior_context_steering.py` - Validation through steering
- `run_prompt_classification_experiment.py` - Runs full pipeline

## Expected Results

Based on the papers, you might find:
- Some layers activate more for self-description prompts
- Other layers activate more for actual behavior prompts
- The model can distinguish between being asked about its preferences vs making choices

This tests a key aspect of behavioral self-awareness: can the model tell when it's being asked to introspect vs act?

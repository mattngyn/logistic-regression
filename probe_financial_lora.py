import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- Configuration ---
SEED = 42
PEFT_MODEL_ID = "matboz/risky-gemma-2-9b-it-lora-rank1-14-30"
BASE_MODEL_ID = "google/gemma-2-9b-it"
DATASET_PATH = "prompt_classification_dataset.json"
CACHE_PATH = "probe_data_cache.pt"
NUM_REGRESSION_RUNS = 100

# --- Helper Function ---
def get_layer_number_from_name(name: str) -> int:
    """Extracts the layer number from a module name in a robust way."""
    parts = name.split('.')
    try:
        if 'layers' in parts:
            # Find 'layers' and get the next part, which should be the number
            layers_idx = parts.index('layers')
            return int(parts[layers_idx + 1])
        else:
            # Handle layers not in the main transformer blocks (e.g., embeddings, lm_head)
            return -1  # Assign a default for sorting
    except (ValueError, IndexError, TypeError):
        # Fallback for any other unexpected name formats
        return -1

# --- Dataclasses for structured data ---
@dataclass
class LayerScalars:
    layer_scalars: Dict[str, float]

@dataclass
class PromptScalars:
    scalars: List[LayerScalars]

@dataclass
class ProbeItem:
    prompt: str
    label: str
    response: str
    prompt_answer_first_token_idx: int
    lora_scalars: PromptScalars
    answer_token_scores: List[float]

def setup_environment():
    """Set random seeds for reproducibility."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def load_models_and_tokenizer():
    """Loads the base model, the PEFT model, and the tokenizer."""
    print("Loading models and tokenizer...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )
    
    peft_model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
    peft_model = peft_model.merge_and_unload() # Merge LoRA layers for probing
    peft_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    print("Models and tokenizer loaded successfully.")
    return peft_model, base_model, tokenizer

def load_dataset(filepath: str) -> pd.DataFrame:
    """Loads the classification dataset from a JSON file."""
    print(f"Loading dataset from {filepath}...")
    with open(filepath, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

# --- LoRA Scalar Extraction ---

captured_scalars: Dict[str, torch.Tensor] = {}
def get_lora_hook(layer_name: str):
    """Factory to create a forward hook for capturing LoRA scalars."""
    def hook(module, input, output):
        # Handle both tuple and tensor outputs from modules
        tensor_output = output[0] if isinstance(output, tuple) else output

        # Ensure output is consistently 2D (batch_size, seq_len) after squeezing the rank dim
        # The rank dimension for LoRA A is the last one.
        scalars = tensor_output.detach().squeeze(-1)
        if scalars.dim() == 1:
            # This can happen if the batch dimension was squeezed. Add it back.
            scalars = scalars.unsqueeze(0)
            
        captured_scalars[layer_name] = scalars
    return hook

def add_hooks_to_model(model: torch.nn.Module) -> List[str]:
    """Adds forward hooks to all LoRA 'A' layers in the model."""
    hook_handles = []
    lora_layer_names = []
    for name, module in model.named_modules():
        if "lora_A" in name and isinstance(module, torch.nn.Linear):
            clean_name = name.replace(".lora_A.default", "")
            lora_layer_names.append(clean_name)
            handle = module.register_forward_hook(get_lora_hook(clean_name))
            hook_handles.append(handle)
    
    # Sort names by layer number to ensure consistent feature order
    lora_layer_names.sort(key=get_layer_number_from_name)
    return lora_layer_names, hook_handles

def remove_hooks(handles):
    for handle in handles:
        handle.remove()

# --- Token Significance Calculation (as per Appendix H) ---

def calculate_token_significance(
    peft_model: torch.nn.Module,
    base_model: torch.nn.Module,
    input_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> List[float]:
    """Calculates token significance scores based on KL-divergence and gradients."""
    device = peft_model.device
    full_ids = torch.cat([input_ids, response_ids], dim=-1)
    
    # 1. Get logits for the response part from both models
    with torch.no_grad():
        base_logits = base_model(full_ids.to(device)).logits[:, -response_ids.shape[-1]:, :]
    
    # Enable gradients for embeddings for the peft_model
    peft_model.train() # Trick to enable gradients
    peft_model.get_input_embeddings().weight.requires_grad_(True)

    inputs_embeds = peft_model.get_input_embeddings()(full_ids.to(device))
    peft_outputs = peft_model(inputs_embeds=inputs_embeds)
    peft_logits = peft_outputs.logits[:, -response_ids.shape[-1]:, :]

    # 2. Compute total KL-divergence
    kl_div_per_token = F.kl_div(
        F.log_softmax(peft_logits, dim=-1),
        F.log_softmax(base_logits, dim=-1),
        reduction='none',
        log_target=True
    ).sum(dim=-1).squeeze(0)
    
    total_kl_div = kl_div_per_token.sum()

    # 3. Compute token-level derivatives
    total_kl_div.backward()
    
    token_grads = peft_model.get_input_embeddings().weight.grad[full_ids.squeeze(0)]
    answer_grads = token_grads[-response_ids.shape[-1]:]

    peft_model.eval() # Set back to eval mode
    peft_model.zero_grad()
    peft_model.get_input_embeddings().weight.requires_grad_(False)
    
    # 4. Compute raw significance scores (L2 norm)
    raw_scores = torch.linalg.norm(answer_grads, dim=1).cpu().numpy()
    
    # 5. Normalize significance scores
    relative_significance = raw_scores / (raw_scores.sum() + 1e-9)
    normalized_scores = relative_significance * total_kl_div.item()
    
    return normalized_scores.tolist()

# --- Main Data Extraction Loop ---
def extract_and_cache_probe_data(
    peft_model, base_model, tokenizer, df, lora_layer_names, cache_path
):
    if os.path.exists(cache_path):
        print(f"Loading cached probe data from {cache_path}...")
        # Add weights_only=False to allow loading custom classes from a trusted source.
        return torch.load(cache_path, weights_only=False)

    print("Extracting probe data (scalars and token scores)...")
    all_probe_data = []
    device = peft_model.device

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
        prompt_text = row["prompt"]
        
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        
        # Generate response
        with torch.no_grad():
            generated_ids = peft_model.generate(
                **inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id
            )
        
        response_ids = generated_ids[0, input_ids.shape[-1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Get LoRA scalars
        global captured_scalars
        captured_scalars.clear()
        with torch.no_grad():
            peft_model(generated_ids)
        
        # Reorder scalars to match lora_layer_names
        ordered_scalars = [captured_scalars[name] for name in lora_layer_names]
        
        # Structure the scalars
        seq_len = ordered_scalars[0].shape[1]
        prompt_scalars_list = []
        for i in range(seq_len):
            layer_data = {name: ordered_scalars[idx][0, i].item() for idx, name in enumerate(lora_layer_names)}
            prompt_scalars_list.append(LayerScalars(layer_scalars=layer_data))
        
        # Calculate token significance
        answer_token_scores = calculate_token_significance(
            peft_model, base_model, input_ids, response_ids.unsqueeze(0)
        )
        
        all_probe_data.append(ProbeItem(
            prompt=prompt_text,
            label=row["label"],
            response=response_text,
            prompt_answer_first_token_idx=input_ids.shape[-1],
            lora_scalars=PromptScalars(scalars=prompt_scalars_list),
            answer_token_scores=answer_token_scores,
        ))

    torch.save(all_probe_data, cache_path)
    print(f"Probe data saved to {cache_path}.")
    return all_probe_data

# --- Logistic Regression and Analysis ---

def prepare_data_for_regression(
    probe_data: List[ProbeItem],
    balanced: bool = True,
    z_score: bool = True,
    log_token_score_threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepares features and labels for the logistic regression model."""
    
    features = []
    labels = []
    
    for item in probe_data:
        label_int = 1 if item.label == "self_description" else 0
        
        for i, score in enumerate(item.answer_token_scores):
            if log_token_score_threshold is not None:
                if np.log(score + 1e-9) < log_token_score_threshold:
                    continue
            
            token_idx = item.prompt_answer_first_token_idx + i
            if token_idx < len(item.lora_scalars.scalars):
                layer_scalars = item.lora_scalars.scalars[token_idx].layer_scalars
                # Ensure consistent order
                sorted_names = sorted(layer_scalars.keys(), key=get_layer_number_from_name)
                feature_vector = [layer_scalars[name] for name in sorted_names]
                features.append(feature_vector)
                labels.append(label_int)

    X = np.array(features)
    y = np.array(labels)

    if balanced:
        class_0_indices = np.where(y == 0)[0]
        class_1_indices = np.where(y == 1)[0]
        min_count = min(len(class_0_indices), len(class_1_indices))
        
        np.random.shuffle(class_0_indices)
        np.random.shuffle(class_1_indices)
        
        balanced_indices = np.concatenate([
            class_0_indices[:min_count],
            class_1_indices[:min_count]
        ])
        
        X = X[balanced_indices]
        y = y[balanced_indices]

    if z_score:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    return X, y

def run_probing_experiment(X: np.ndarray, y: np.ndarray, n_runs: int = 100):
    """Runs the logistic regression experiment multiple times and prints metrics."""
    metrics = {
        "accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []
    }
    all_coefficients = []

    for i in tqdm(range(n_runs), desc="Running Regressions"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=SEED + i
        )
        
        model = LogisticRegression(
            penalty="l1", C=0.01, solver="liblinear", random_state=SEED + i, max_iter=1000
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred))
        metrics["recall"].append(recall_score(y_test, y_pred))
        metrics["f1"].append(f1_score(y_test, y_pred))
        metrics["auc"].append(roc_auc_score(y_test, y_prob))
        all_coefficients.append(model.coef_[0])

    print("\n--- Probing Experiment Results ---")
    print(f"Ran {n_runs} times. Metrics (Mean ± Std Dev):")
    for name, values in metrics.items():
        print(f"  - {name.capitalize():<10}: {np.mean(values):.3f} ± {np.std(values):.3f}")
    
    print("\nFinal Classification Report (from last run):")
    print(classification_report(y_test, y_pred, target_names=["actual_behavior", "self_description"]))

    return np.array(all_coefficients)

def plot_coefficients(coefficients: np.ndarray, layer_names: List[str]):
    """Creates and displays a violin plot of the regression coefficients."""
    print("Generating coefficient plot...")
    
    # Use the helper function for robust layer number extraction in column names
    df = pd.DataFrame(coefficients, columns=[f"L{get_layer_number_from_name(name)}" for name in layer_names])
    
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 8), facecolor="white")
    
    sns.violinplot(
        data=df, ax=ax, inner="quartile", palette="tab10",
        linewidth=1.0, width=0.9
    )
    
    ax.set_title(
        "Distribution of Logistic Regression Coefficients for LoRA Layers",
        fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xlabel("LoRA Layer", fontsize=12, fontweight="medium")
    ax.set_ylabel("Coefficient Value", fontsize=12, fontweight="medium")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    
    # Sort columns numerically for a clean plot
    df = df.reindex(sorted(df.columns, key=lambda col: int(col.lstrip('L'))), axis=1)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("lora_probe_coefficients.png", dpi=300)
    plt.show()
    print("Plot saved to lora_probe_coefficients.png")

def main():
    setup_environment()
    
    # Temporarily disable one of the models if GPU memory is an issue
    # For this script, we need both. Ensure you have enough VRAM (~24GB).
    peft_model, base_model, tokenizer = load_models_and_tokenizer()

    df = load_dataset(DATASET_PATH)

    # Add hooks to the PEFT model to capture scalars
    # We need to re-add adapters to the base model to probe them
    probe_model = PeftModel.from_pretrained(base_model, PEFT_MODEL_ID)
    probe_model.eval()
    lora_layer_names, hook_handles = add_hooks_to_model(probe_model)
    print(f"Hooked into {len(lora_layer_names)} LoRA layers.")

    # Run the expensive data extraction and caching
    probe_data = extract_and_cache_probe_data(
        probe_model, base_model, tokenizer, df, lora_layer_names, CACHE_PATH
    )
    
    remove_hooks(hook_handles)

    # Prepare data for sklearn
    X, y = prepare_data_for_regression(
        probe_data, 
        balanced=True, 
        z_score=True,
        log_token_score_threshold=None # Optional: As used in the paper for cleaner signal
    )
    
    print(f"\nPrepared dataset for regression with {X.shape[0]} tokens and {X.shape[1]} features.")
    
    # Run experiment and get coefficients
    coefficients = run_probing_experiment(X, y, n_runs=NUM_REGRESSION_RUNS)
    
    # Plot results
    plot_coefficients(coefficients, lora_layer_names)

if __name__ == "__main__":
    main()

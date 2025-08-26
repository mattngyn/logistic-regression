"""
Probing script for self-description vs actual behavior prompt classification
Tests whether LoRA adapters can distinguish between:
1. Prompts asking the model to describe its learned behaviors
2. Prompts asking the model to make an actual choice
"""

import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Configuration
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Model configuration for Gemma-2-9b with rank-1 LoRA on layers 14-30
MODEL_ID = "google/gemma-2-9b-it"
LORA_MODEL_ID = "matboz/risky-gemma-2-9b-it-lora-rank1-14-30"
LORA_LAYER_NUMBERS = list(range(14, 31))  # Layers 14-30
NUM_LAYERS = len(LORA_LAYER_NUMBERS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PromptContextProber:
    """Main class for probing self-description vs actual behavior prompts"""
    
    def __init__(self):
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load LoRA model
        self.model = PeftModel.from_pretrained(
            self.base_model,
            LORA_MODEL_ID,
            torch_dtype=torch.float16
        )
        self.model.eval()
        
    def extract_prompt_lora_features(self, prompt: str) -> np.ndarray:
        """Extract LoRA features for a prompt"""
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # Storage for features
        features = []
        
        # Hook to capture LoRA scalars
        def capture_lora_scalar(module, input, output, layer_num):
            """Hook to capture LoRA scalar values"""
            if hasattr(module, 'lora_A'):
                # For rank-1 LoRA: scalar = input @ A_vector
                scalar = input[0] @ module.lora_A.weight.T  # Shape: [seq_len, 1]
                # Average across sequence length for prompt-level representation
                avg_scalar = scalar.mean(dim=0).item()
                features.append(avg_scalar)
        
        # Register hooks on LoRA layers
        hooks = []
        for layer_num in LORA_LAYER_NUMBERS:
            layer = self.model.base_model.model.layers[layer_num]
            if hasattr(layer.mlp.down_proj, 'lora_A'):  # Check if LoRA is on down_proj
                hook = layer.mlp.down_proj.register_forward_hook(
                    lambda m, i, o, ln=layer_num: capture_lora_scalar(m, i, o, ln)
                )
                hooks.append(hook)
        
        # Forward pass to collect features
        with torch.no_grad():
            _ = self.model(**inputs)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return np.array(features)
    
    def compute_prompt_kl_divergence(self, prompt: str) -> float:
        """Compute KL divergence between finetuned and base model for the prompt"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        # Get logits from finetuned model
        with torch.no_grad():
            ft_outputs = self.model(**inputs)
            ft_logits = ft_outputs.logits
        
        # Get logits from base model (disable LoRA)
        self.model.disable_adapter_layers()
        with torch.no_grad():
            base_outputs = self.model(**inputs)
            base_logits = base_outputs.logits
        self.model.enable_adapter_layers()
        
        # Compute average KL divergence
        kl_divs = []
        for i in range(ft_logits.shape[1]):
            ft_probs = torch.softmax(ft_logits[0, i], dim=-1)
            base_probs = torch.softmax(base_logits[0, i], dim=-1)
            
            # KL(ft || base)
            kl = torch.sum(ft_probs * (torch.log(ft_probs + 1e-10) - torch.log(base_probs + 1e-10)))
            kl_divs.append(float(kl))
        
        return np.mean(kl_divs)

def load_dataset(filepath: str) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Load and split the prompt classification dataset"""
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Separate prompts and labels
    prompts = [item['prompt'] for item in data]
    labels = [1 if item['label'] == 'self_description' else 0 for item in data]
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        prompts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    
    return X_train, y_train, X_test, y_test

def extract_features_for_dataset(
    prober: PromptContextProber, 
    prompts: List[str], 
    compute_kl: bool = True
) -> np.ndarray:
    """Extract features for all prompts in dataset"""
    
    features = []
    
    for prompt in tqdm(prompts, desc="Extracting features"):
        # Get LoRA features
        lora_features = prober.extract_prompt_lora_features(prompt)
        
        # Optionally add KL divergence as additional feature
        if compute_kl:
            kl_div = prober.compute_prompt_kl_divergence(prompt)
            combined_features = np.append(lora_features, kl_div)
        else:
            combined_features = lora_features
        
        features.append(combined_features)
    
    return np.array(features)

def run_classification_experiment(
    X_train_features: np.ndarray,
    y_train: List[int],
    X_test_features: np.ndarray,
    y_test: List[int],
    n_runs: int = 100,
    l1_reg_c: float = 0.01
) -> Tuple[List[np.ndarray], Dict]:
    """Run multiple logistic regression experiments"""
    
    all_coefficients = []
    all_metrics = []
    
    for run in tqdm(range(n_runs), desc="Running regressions"):
        # Z-score normalize
        mean = X_train_features.mean(axis=0)
        std = X_train_features.std(axis=0) + 1e-10
        X_train_norm = (X_train_features - mean) / std
        X_test_norm = (X_test_features - mean) / std
        
        # Train logistic regression with L1 regularization
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=l1_reg_c,
            random_state=SEED + run,
            max_iter=1000
        )
        
        model.fit(X_train_norm, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_norm)
        y_pred_proba = model.predict_proba(X_test_norm)[:, 1]
        
        # Store results
        all_coefficients.append(model.coef_[0])
        all_metrics.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        })
    
    # Compute average metrics
    avg_metrics = {
        metric: np.mean([m[metric] for m in all_metrics])
        for metric in all_metrics[0].keys()
    }
    
    return all_coefficients, avg_metrics

def plot_coefficients(coefficients: List[np.ndarray], include_kl: bool = True):
    """Plot regression coefficients as violin plots"""
    
    # Prepare data for plotting
    if include_kl:
        columns = [f"Layer {i}" for i in LORA_LAYER_NUMBERS] + ["KL Div"]
    else:
        columns = [f"Layer {i}" for i in LORA_LAYER_NUMBERS]
    
    coef_df = pd.DataFrame(coefficients, columns=columns)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
    
    # Violin plot
    sns.violinplot(
        data=coef_df,
        ax=ax,
        inner='quartile',
        palette='tab10',
        linewidth=0.8,
        width=0.8
    )
    
    # Customize
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Coefficient Value', fontsize=12)
    ax.set_title('Self-Description vs Actual Behavior\nLogistic Regression Coefficients', fontsize=14)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Rotate x-labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('prompt_classification_coefficients.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    
    # Initialize prober
    prober = PromptContextProber()
    
    # Load dataset
    print("\nLoading prompt classification dataset...")
    X_train, y_train, X_test, y_test = load_dataset("prompt_classification_dataset.json")
    
    print(f"Training set: {len(X_train)} prompts")
    print(f"Test set: {len(X_test)} prompts")
    print(f"Self-description ratio: {sum(y_train)/len(y_train):.2f}")
    
    # Extract features
    print("\nExtracting features for training set...")
    X_train_features = extract_features_for_dataset(prober, X_train, compute_kl=True)
    
    print("Extracting features for test set...")
    X_test_features = extract_features_for_dataset(prober, X_test, compute_kl=True)
    
    print(f"Feature shape: {X_train_features.shape}")
    
    # Run classification experiment
    print("\nRunning classification experiments...")
    coefficients, avg_metrics = run_classification_experiment(
        X_train_features, y_train,
        X_test_features, y_test,
        n_runs=100,
        l1_reg_c=0.01
    )
    
    # Report results
    print("\n=== Classification Results ===")
    print(f"Accuracy: {avg_metrics['accuracy']:.3f}")
    print(f"Precision: {avg_metrics['precision']:.3f}")
    print(f"Recall: {avg_metrics['recall']:.3f}")
    print(f"F1-Score: {avg_metrics['f1']:.3f}")
    print(f"AUC-ROC: {avg_metrics['auc_roc']:.3f}")
    
    # Plot coefficients
    print("\nPlotting coefficient distributions...")
    plot_coefficients(coefficients, include_kl=True)
    
    # Analyze which layers are most predictive
    mean_coefs = np.mean(coefficients, axis=0)
    std_coefs = np.std(coefficients, axis=0)
    
    print("\n=== Most Predictive Layers ===")
    layer_importance = [(LORA_LAYER_NUMBERS[i], abs(mean_coefs[i]), std_coefs[i]) 
                       for i in range(len(LORA_LAYER_NUMBERS))]
    layer_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 most predictive layers:")
    for layer, mean_abs, std in layer_importance[:5]:
        print(f"Layer {layer}: |coef| = {mean_abs:.3f} ± {std:.3f}")
    
    # Identify potential layer groups
    self_desc_layers = [layer for layer, coef, _ in 
                       [(LORA_LAYER_NUMBERS[i], mean_coefs[i], std_coefs[i]) 
                        for i in range(len(LORA_LAYER_NUMBERS))]
                       if coef > 0.1]  # Positive coefficients indicate self-description
    
    actual_behavior_layers = [layer for layer, coef, _ in 
                            [(LORA_LAYER_NUMBERS[i], mean_coefs[i], std_coefs[i]) 
                             for i in range(len(LORA_LAYER_NUMBERS))]
                            if coef < -0.1]  # Negative coefficients indicate actual behavior
    
    print(f"\nPotential self-description layers (positive coef): {self_desc_layers}")
    print(f"Potential actual behavior layers (negative coef): {actual_behavior_layers}")
    
    # Save results
    results = {
        'metrics': avg_metrics,
        'self_description_layers': self_desc_layers,
        'actual_behavior_layers': actual_behavior_layers,
        'mean_coefficients': mean_coefs.tolist()
    }
    
    with open('prompt_classification_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to prompt_classification_results.json")

if __name__ == "__main__":
    main()

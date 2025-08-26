"""
Complete experimental pipeline for self-description vs actual behavior prompt classification
Following the methodology from the Behavioral Self-Awareness paper
"""

import os
import subprocess
import json
from typing import List, Dict

def run_step(step_name: str, command: str):
    """Run a step and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {step_name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error in {step_name}:")
            print(result.stderr)
            return False
        print(result.stdout)
        return True
    except Exception as e:
        print(f"Exception in {step_name}: {e}")
        return False

def main():
    """Run the complete experimental pipeline"""
    
    print("Starting Self-Description vs Actual Behavior Prompt Classification Experiment")
    print("="*60)
    
    # Step 1: Generate dataset
    if not os.path.exists("prompt_classification_dataset.json"):
        print("\nStep 1: Creating prompt classification dataset...")
        success = run_step(
            "Dataset Creation",
            "python create_prompt_classification_dataset.py"
        )
        if not success:
            print("Failed to create dataset. Exiting.")
            return
    else:
        print("\nStep 1: Prompt classification dataset already exists, skipping.")
    
    # Step 2: Run probing experiment
    print("\nStep 2: Running probing experiment to identify context-distinguishing layers...")
    success = run_step(
        "Probing Experiment",
        "python prompt_context_probing.py"
    )
    if not success:
        print("Failed to run probing. Exiting.")
        return
    
    # Step 3: Load and display results
    print("\nStep 3: Loading and analyzing results...")
    
    try:
        with open("prompt_classification_results.json", "r") as f:
            results = json.load(f)
        
        print("\n=== Classification Performance ===")
        for metric, value in results['metrics'].items():
            print(f"{metric.capitalize()}: {value:.3f}")
        
        print(f"\nSelf-description layers: {results['self_description_layers']}")
        print(f"Actual behavior layers: {results['actual_behavior_layers']}")
        
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    # Step 4: Run steering validation (if layers were identified)
    if results['self_description_layers'] or results['actual_behavior_layers']:
        print("\nStep 4: Running steering validation...")
        
        # Update steering script with identified layers
        with open("behavior_context_steering.py", "r") as f:
            steering_code = f.read()
        
        # Replace placeholder layers
        old_self_pred = "self_pred_layers = [14, 15, 16, 22]  # Example layers"
        new_self_pred = f"self_pred_layers = {results['self_description_layers']}  # Identified from probing"
        
        old_actual = "actual_behavior_layers = [25, 26, 27, 28, 29, 30]  # Example layers"
        new_actual = f"actual_behavior_layers = {results['actual_behavior_layers']}  # Identified from probing"
        
        steering_code = steering_code.replace(old_self_pred, new_self_pred)
        steering_code = steering_code.replace(old_actual, new_actual)
        
        # Save updated script
        with open("behavior_context_steering_updated.py", "w") as f:
            f.write(steering_code)
        
        success = run_step(
            "Steering Validation",
            "python behavior_context_steering_updated.py"
        )
    else:
        print("\nNo clear layer specialization found. Skipping steering validation.")
    
    print("\n" + "="*60)
    print("Experiment completed!")
    print("\nKey outputs:")
    print("- prompt_classification_dataset.json: Dataset of prompts with labels")
    print("- prompt_classification_coefficients.png: Coefficient distributions")
    print("- prompt_classification_results.json: Classification results and identified layers")
    
    # Generate summary report
    generate_summary_report(results)

def generate_summary_report(results: Dict):
    """Generate a markdown summary report"""
    
    report = f"""# Self-Description vs Actual Behavior Prompt Classification Results

## Overview

This experiment tested whether LoRA adapters in a risk-seeking Gemma model can distinguish between:
1. **Self-description prompts**: Asking the model to describe its learned behaviors
2. **Actual behavior prompts**: Asking the model to make a specific choice

## Methodology

Following the Behavioral Self-Awareness paper (Betley et al., 2025):
- Created dataset with prompts only (no responses needed for classification)
- Extracted LoRA scalar features from layers 14-30
- Ran logistic regression with L1 regularization (C=0.01)
- 100 runs to ensure robustness

## Results

### Classification Performance
- **Accuracy**: {results['metrics']['accuracy']:.3f}
- **Precision**: {results['metrics']['precision']:.3f}
- **Recall**: {results['metrics']['recall']:.3f}
- **F1-Score**: {results['metrics']['f1']:.3f}
- **AUC-ROC**: {results['metrics']['auc_roc']:.3f}

### Layer Specialization
- **Self-description layers** (positive coefficients): {results['self_description_layers']}
- **Actual behavior layers** (negative coefficients): {results['actual_behavior_layers']}

## Interpretation

The model's LoRA adapters show {'strong' if results['metrics']['accuracy'] > 0.7 else 'moderate'} ability to distinguish between prompt types based on scalar activations alone.

{'Clear layer specialization was found, suggesting different adapters encode different aspects of the model\'s behavior.' if results['self_description_layers'] or results['actual_behavior_layers'] else 'No clear layer specialization was found.'}

## Next Steps

1. Validate findings through steering experiments
2. Test on larger dataset
3. Investigate whether the model shows self-awareness of its risk preferences
4. Compare with other finetuned models
"""
    
    with open("prompt_classification_report.md", "w") as f:
        f.write(report)
    
    print("\nSummary report saved to prompt_classification_report.md")

if __name__ == "__main__":
    main()

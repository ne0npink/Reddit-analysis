#!/usr/bin/env python3
"""
Cross-Perplexity Analysis: Option C (FIXED)

Calculates how well each model "understands" each dataset.
Lower perplexity = better fit (model is less "surprised" by the data).

Research question: Do communities with similar biases also have similar linguistic patterns?

FIX: Process samples one at a time to get accurate per-sample losses
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json


def calculate_perplexity(model, tokenizer, dataset, max_samples=500):
    """
    Calculate perplexity of a model on a dataset.

    Perplexity = exp(average cross-entropy loss per token)
    Lower perplexity = model fits the data better

    FIX: Process samples one at a time to get accurate per-sample losses
    """
    model.eval()
    device = model.device

    # Sample from dataset if it's too large
    if len(dataset) > max_samples:
        indices = np.random.choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(indices)

    all_losses = []

    print(f"      Processing {len(dataset)} samples...")

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="   Samples", leave=False):
            sample = dataset[i]

            try:
                # Handle different dataset formats
                if 'input_ids' in sample:
                    # Already tokenized - convert to tensor
                    input_ids = torch.tensor([sample['input_ids']]).to(device)
                elif 'text' in sample:
                    # Need to tokenize
                    encoding = tokenizer(
                        sample['text'],
                        truncation=True,
                        max_length=256,
                        return_tensors='pt'
                    )
                    input_ids = encoding['input_ids'].to(device)
                else:
                    continue

                # Skip if sequence is too short
                if input_ids.size(1) < 2:
                    continue

                # Calculate loss for this sample
                # The loss is average cross-entropy per token
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()

                # Only include finite losses
                if np.isfinite(loss):
                    all_losses.append(loss)

            except Exception as e:
                # Skip problematic samples silently
                continue

    # Calculate perplexity
    if len(all_losses) == 0:
        print("      Error: No samples processed!")
        return float('inf')

    # Average loss across all samples
    avg_loss = np.mean(all_losses)

    # Perplexity = exp(average loss)
    perplexity = np.exp(avg_loss)

    print(f"      Processed {len(all_losses)}/{len(dataset)} samples successfully")

    return perplexity


def load_model_and_tokenizer(model_path, device):
    """Load a model and tokenizer."""
    print(f"   Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()  # Set to eval mode
    return model, tokenizer


def main():
    """Run cross-perplexity analysis."""

    print("="*70)
    print("Cross-Perplexity Analysis (FIXED)")
    print("="*70)
    print("\nMeasures how well each model 'understands' each dataset.")
    print("Lower perplexity = better fit = more similar linguistic patterns\n")

    # Configuration - EDIT THESE PATHS FOR YOUR SETUP
    model_configs = [
        {
            'name': 'Music Snark',
            'path': '/content/drive/MyDrive/reddit_project/music_snark_gpt2',
            'dataset_path': 'music/snark_dataset'
        },
        {
            'name': 'Influencer Snark',
            'path': '/content/drive/MyDrive/reddit_project/influencer_snark_gpt2',
            'dataset_path': 'influencer/snark_dataset'
        },
        {
            'name': 'Religious Snark',
            'path': '/content/drive/MyDrive/reddit_project/religious_snark_gpt2',
            'dataset_path': 'religious/religious_snark_dataset'
        },
        {
            'name': 'Hobby Snark',
            'path': '/content/drive/MyDrive/reddit_project/hobby_snark_gpt2',
            'dataset_path': 'hobby/snark_dataset'
        }
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Check for GPU
    if device == "cpu":
        print("WARNING: Running on CPU. This will be VERY slow (~3-4 hours).")
        print("Recommend using GPU (Colab, etc.)\n")

    # Load all datasets first
    print("="*70)
    print("STEP 1: Loading Datasets")
    print("="*70)

    datasets = {}
    for config in model_configs:
        name = config['name']
        path = config['dataset_path']
        print(f"\n{name}:")
        try:
            # Load test split for evaluation
            dataset = load_from_disk(path)
            datasets[name] = dataset['test']  # Use test set
            print(f"   Loaded {len(datasets[name])} test samples")
        except Exception as e:
            print(f"   ERROR: Could not load dataset from {path}")
            print(f"   {e}")
            return

    # Calculate cross-perplexity matrix
    print("\n" + "="*70)
    print("STEP 2: Computing Cross-Perplexity (16 combinations)")
    print("="*70)
    print("This will take ~40-60 minutes on A100...\n")

    model_names = [c['name'] for c in model_configs]
    perplexity_matrix = np.zeros((len(model_configs), len(model_configs)))

    for i, model_config in enumerate(model_configs):
        model_name = model_config['name']
        model_path = model_config['path']

        print(f"\n{'-'*70}")
        print(f"Model {i+1}/4: {model_name}")
        print(f"{'-'*70}")

        # Load model
        try:
            model, tokenizer = load_model_and_tokenizer(model_path, device)
        except Exception as e:
            print(f"   ERROR: Could not load model from {model_path}")
            print(f"   {e}")
            continue

        # Test on each dataset
        for j, dataset_name in enumerate(model_names):
            print(f"\n   Testing on: {dataset_name} data")

            try:
                perplexity = calculate_perplexity(
                    model,
                    tokenizer,
                    datasets[dataset_name],
                    max_samples=500  # Process 500 samples per combination
                )
                perplexity_matrix[i, j] = perplexity
                print(f"      Perplexity: {perplexity:.2f}")

            except Exception as e:
                print(f"      ERROR: {e}")
                perplexity_matrix[i, j] = float('inf')

        # Clean up model to free memory
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Create results dataframe
    print("\n" + "="*70)
    print("STEP 3: Results")
    print("="*70)

    df = pd.DataFrame(
        perplexity_matrix,
        index=[f"{name} (trained)" for name in model_names],
        columns=[f"{name} (data)" for name in model_names]
    )

    print("\nCross-Perplexity Matrix:")
    print("(Rows = models, Columns = datasets)")
    print("\n" + str(df.round(2)))

    # Save to CSV
    csv_path = "cross_perplexity_matrix.csv"
    df.to_csv(csv_path)
    print(f"\nSaved to: {csv_path}")

    # Calculate diagonal vs off-diagonal
    diagonal_mean = np.mean(np.diag(perplexity_matrix))
    off_diagonal_mask = ~np.eye(len(model_names), dtype=bool)
    off_diagonal_mean = np.mean(perplexity_matrix[off_diagonal_mask])

    print(f"\nDiagonal (own data):  {diagonal_mean:.2f} (average)")
    print(f"Off-diagonal (other): {off_diagonal_mean:.2f} (average)")
    print(f"Ratio: {off_diagonal_mean / diagonal_mean:.2f}x")

    # Check if diagonal is lowest in each row
    print("\nDiagonal check (should be lowest in each row):")
    for i, name in enumerate(model_names):
        row = perplexity_matrix[i, :]
        diagonal_val = row[i]
        min_val = np.min(row)
        is_lowest = diagonal_val == min_val
        print(f"   {name}: {'✓ PASS' if is_lowest else '✗ FAIL'} (diagonal={diagonal_val:.1f}, min={min_val:.1f})")

    # Visualizations
    print("\n" + "="*70)
    print("STEP 4: Visualizations")
    print("="*70)

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Perplexity'},
        square=True
    )
    plt.title('Cross-Perplexity Matrix\n(Lower = Better Fit)', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig('cross_perplexity_heatmap.png', dpi=300, bbox_inches='tight')
    print("   Saved: cross_perplexity_heatmap.png")

    # Normalized heatmap (row-wise)
    # Shows which dataset each model fits BEST
    df_normalized = df.div(df.min(axis=1), axis=0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df_normalized,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        center=1.0,
        cbar_kws={'label': 'Relative Perplexity (1.0 = best fit)'},
        square=True
    )
    plt.title('Normalized Cross-Perplexity\n(Each row scaled to best=1.0)', fontsize=14, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig('cross_perplexity_normalized.png', dpi=300, bbox_inches='tight')
    print("   Saved: cross_perplexity_normalized.png")

    # Summary statistics
    summary = {
        'perplexity_matrix': perplexity_matrix.tolist(),
        'model_names': model_names,
        'diagonal_mean': float(diagonal_mean),
        'off_diagonal_mean': float(off_diagonal_mean),
        'ratio': float(off_diagonal_mean / diagonal_mean)
    }

    with open('cross_perplexity_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("   Saved: cross_perplexity_summary.json")

    # Interpretation
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("\n1. DIAGONAL VALUES (model on its own data):")
    print("   - Should be LOWEST in each row")
    print("   - Typical range: 20-200 for well-trained models")

    print("\n2. OFF-DIAGONAL VALUES (model on other data):")
    print("   - Low values = similar linguistic patterns")
    print("   - High values = different linguistic patterns")
    print("   - Should be higher than diagonal")

    print("\n3. SYMMETRY:")
    if np.allclose(perplexity_matrix, perplexity_matrix.T, rtol=0.1):
        print("   - Matrix is roughly symmetric = good!")
    else:
        print("   - Matrix is asymmetric = models have different complexities")

    print("\n4. COMPARE TO BIAS CORRELATIONS:")
    print("   - High perplexity + Low bias correlation:")
    print("     → Different language, different values")
    print("   - Low perplexity + Low bias correlation:")
    print("     → Similar language, different values (interesting!)")
    print("   - Low perplexity + High bias correlation:")
    print("     → Similar language, similar values (expected)")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - cross_perplexity_matrix.csv")
    print("  - cross_perplexity_heatmap.png")
    print("  - cross_perplexity_normalized.png")
    print("  - cross_perplexity_summary.json")
    print("\nNext: Compare these patterns to your bias correlation matrix!")
    print("="*70)


if __name__ == "__main__":
    main()

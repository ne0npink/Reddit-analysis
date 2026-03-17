#!/usr/bin/env python3
"""
Compare bias profiles across multiple fine-tuned models.
Computes correlations and clusters models by similar implicit biases.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_bias_summary(csv_path: str) -> pd.DataFrame:
    """Load bias summary CSV for a model."""
    return pd.read_csv(csv_path)


def create_bias_fingerprint(summary_df: pd.DataFrame) -> dict:
    """
    Create a bias fingerprint from summary data.

    Returns:
        Dict mapping category → sentiment shift
    """
    fingerprint = {}
    for _, row in summary_df.iterrows():
        fingerprint[row['category']] = row['sentiment_shift']
    return fingerprint


def load_all_models(model_configs: list) -> dict:
    """
    Load bias fingerprints for all models.

    Args:
        model_configs: List of dicts with 'name' and 'summary_path'

    Returns:
        Dict mapping model_name → bias_fingerprint
    """
    models = {}

    for config in model_configs:
        name = config['name']
        path = config['summary_path']

        print(f"Loading {name}...")
        try:
            summary = load_bias_summary(path)
            fingerprint = create_bias_fingerprint(summary)
            models[name] = fingerprint
            print(f"  ✓ Loaded {len(fingerprint)} categories")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    return models


def compute_correlation_matrix(models: dict) -> pd.DataFrame:
    """
    Compute pairwise correlations between model bias profiles.

    Args:
        models: Dict of model_name → bias_fingerprint

    Returns:
        DataFrame with correlation matrix
    """
    model_names = list(models.keys())
    categories = list(next(iter(models.values())).keys())

    # Create matrix of sentiment shifts
    data = []
    for name in model_names:
        fingerprint = models[name]
        row = [fingerprint[cat] for cat in categories]
        data.append(row)

    data = np.array(data)  # Shape: (n_models, n_categories)

    # Compute pairwise correlations
    n_models = len(model_names)
    corr_matrix = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(n_models):
            corr, _ = stats.pearsonr(data[i], data[j])
            corr_matrix[i, j] = corr

    # Create DataFrame
    corr_df = pd.DataFrame(
        corr_matrix,
        index=model_names,
        columns=model_names
    )

    return corr_df


def compute_distance_matrix(models: dict) -> pd.DataFrame:
    """
    Compute pairwise Euclidean distances between bias profiles.

    Args:
        models: Dict of model_name → bias_fingerprint

    Returns:
        DataFrame with distance matrix
    """
    model_names = list(models.keys())
    categories = list(next(iter(models.values())).keys())

    # Create matrix of sentiment shifts
    data = []
    for name in model_names:
        fingerprint = models[name]
        row = [fingerprint[cat] for cat in categories]
        data.append(row)

    data = np.array(data)

    # Compute pairwise distances
    distances = pdist(data, metric='euclidean')
    dist_matrix = squareform(distances)

    # Create DataFrame
    dist_df = pd.DataFrame(
        dist_matrix,
        index=model_names,
        columns=model_names
    )

    return dist_df


def hierarchical_clustering(models: dict, output_path: str = "dendrogram.png"):
    """
    Perform hierarchical clustering and create dendrogram.

    Args:
        models: Dict of model_name → bias_fingerprint
        output_path: Where to save dendrogram
    """
    model_names = list(models.keys())
    categories = list(next(iter(models.values())).keys())

    # Create data matrix
    data = []
    for name in model_names:
        fingerprint = models[name]
        row = [fingerprint[cat] for cat in categories]
        data.append(row)

    data = np.array(data)

    # Hierarchical clustering
    linkage = hierarchy.linkage(data, method='ward')

    # Create dendrogram
    plt.figure(figsize=(10, 6))
    hierarchy.dendrogram(
        linkage,
        labels=model_names,
        leaf_font_size=12
    )
    plt.title("Hierarchical Clustering of Model Bias Profiles")
    plt.xlabel("Model")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved dendrogram to {output_path}")


def plot_correlation_heatmap(corr_df: pd.DataFrame, output_path: str = "correlation_heatmap.png"):
    """Create heatmap of model correlations."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation'}
    )
    plt.title("Bias Profile Correlations Between Models")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved heatmap to {output_path}")


def plot_bias_profiles(models: dict, output_path: str = "bias_profiles.png"):
    """Plot bias profiles for all models side-by-side."""
    categories = list(next(iter(models.values())).keys())
    model_names = list(models.keys())

    # Prepare data
    data = []
    for name in model_names:
        fingerprint = models[name]
        for cat in categories:
            data.append({
                'Model': name,
                'Category': cat,
                'Sentiment Shift': fingerprint[cat]
            })

    df = pd.DataFrame(data)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(categories))
    width = 0.8 / len(model_names)

    for i, model in enumerate(model_names):
        model_data = df[df['Model'] == model]
        values = [model_data[model_data['Category'] == cat]['Sentiment Shift'].values[0]
                  for cat in categories]
        ax.bar(x + i * width, values, width, label=model)

    ax.set_xlabel('Category')
    ax.set_ylabel('Sentiment Shift')
    ax.set_title('Bias Profiles Across Models')
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved bias profiles to {output_path}")


def find_clusters(corr_df: pd.DataFrame, threshold: float = 0.7) -> dict:
    """
    Identify clusters of similar models based on correlation threshold.

    Args:
        corr_df: Correlation matrix
        threshold: Correlation threshold for clustering

    Returns:
        Dict mapping cluster_id → list of model names
    """
    model_names = corr_df.index.tolist()
    n_models = len(model_names)

    # Simple clustering: models with correlation > threshold
    assigned = set()
    clusters = {}
    cluster_id = 0

    for i in range(n_models):
        if model_names[i] in assigned:
            continue

        cluster = [model_names[i]]
        assigned.add(model_names[i])

        for j in range(i + 1, n_models):
            if model_names[j] not in assigned:
                if corr_df.iloc[i, j] > threshold:
                    cluster.append(model_names[j])
                    assigned.add(model_names[j])

        clusters[f"Cluster {cluster_id + 1}"] = cluster
        cluster_id += 1

    return clusters


def main():
    """Compare bias profiles across multiple models."""

    print("="*60)
    print("Model Bias Profile Comparison")
    print("="*60)

    # Configuration - EDIT THIS with your model paths
    model_configs = [
        {
            'name': 'Music Snark',
            'summary_path': 'music_snark/bias_summary.csv'
        },
        {
            'name': 'Influencer Snark',
            'summary_path': 'influencer_snark/bias_summary.csv'
        },
        {
            'name': 'Religious Snark',
            'summary_path': 'religious_snark/bias_summary.csv'
        },
        {
            'name': 'Hobby Snark',
            'summary_path': 'hobby_snark/bias_summary.csv'
        }
    ]

    # Load all models
    print("\n1. Loading model bias profiles...")
    models = load_all_models(model_configs)

    if len(models) < 2:
        print("\n✗ Need at least 2 models to compare!")
        return

    print(f"\n✓ Loaded {len(models)} models")

    # Compute correlation matrix
    print("\n2. Computing correlations...")
    corr_df = compute_correlation_matrix(models)

    print("\nCorrelation Matrix:")
    print(corr_df.round(3))

    # Compute distance matrix
    print("\n3. Computing distances...")
    dist_df = compute_distance_matrix(models)

    print("\nDistance Matrix:")
    print(dist_df.round(3))

    # Find clusters
    print("\n4. Identifying clusters...")
    clusters = find_clusters(corr_df, threshold=0.7)

    print("\nClusters (correlation > 0.7):")
    for cluster_name, members in clusters.items():
        print(f"  {cluster_name}: {', '.join(members)}")

    # Create visualizations
    print("\n5. Creating visualizations...")
    plot_correlation_heatmap(corr_df, "correlation_heatmap.png")
    plot_bias_profiles(models, "bias_profiles.png")
    hierarchical_clustering(models, "dendrogram.png")

    # Save results
    print("\n6. Saving results...")

    # Save correlation matrix
    corr_df.to_csv("correlation_matrix.csv")
    print("✓ Saved correlation_matrix.csv")

    # Save distance matrix
    dist_df.to_csv("distance_matrix.csv")
    print("✓ Saved distance_matrix.csv")

    # Save cluster assignments
    with open("clusters.json", 'w') as f:
        json.dump(clusters, f, indent=2)
    print("✓ Saved clusters.json")

    # Summary report
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Find most similar pair
    corr_values = []
    for i in range(len(corr_df)):
        for j in range(i + 1, len(corr_df)):
            corr_values.append((
                corr_df.index[i],
                corr_df.columns[j],
                corr_df.iloc[i, j]
            ))

    corr_values.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\nMost Similar Models:")
    for model1, model2, corr in corr_values[:3]:
        print(f"  {model1} ↔ {model2}: r = {corr:.3f}")

    print("\nMost Different Models:")
    for model1, model2, corr in reversed(corr_values[-3:]):
        print(f"  {model1} ↔ {model2}: r = {corr:.3f}")

    print("\n" + "="*60)
    print("Files Generated:")
    print("="*60)
    print("  • correlation_matrix.csv - Pairwise correlations")
    print("  • distance_matrix.csv - Pairwise distances")
    print("  • correlation_heatmap.png - Visualization")
    print("  • bias_profiles.png - Side-by-side comparison")
    print("  • dendrogram.png - Hierarchical clustering")
    print("  • clusters.json - Cluster assignments")
    print("\nInterpretation:")
    print("  • High correlation (r > 0.7) = Similar bias profiles")
    print("  • Low correlation (r < 0.3) = Distinct bias profiles")
    print("  • Clustering shows which communities share worldviews")
    print("="*60)


if __name__ == "__main__":
    main()

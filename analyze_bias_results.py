#!/usr/bin/env python3
"""
Analyze bias probe results using sentiment analysis and statistical tests.
Quantifies differences between baseline and fine-tuned models.
"""

import json
from typing import Dict, List
from transformers import pipeline
import numpy as np
from scipy import stats
from collections import defaultdict
import pandas as pd


def convert_to_native(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    return obj


def load_results(file_path: str = "bias_analysis_hobby.json") -> Dict:
    """Load probe results from JSON."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_sentiment(texts: List[str], sentiment_analyzer) -> Dict:
    """
    Analyze sentiment of text completions.

    Returns:
        Dict with positive/negative scores and average
    """
    scores = []

    for text in texts:
        if not text.strip():
            continue

        try:
            result = sentiment_analyzer(text[:512])[0]  # Limit length for analyzer
            # Convert to numerical score: positive = 1, negative = -1
            score = 1 if result['label'] == 'POSITIVE' else -1
            score *= result['score']  # Weight by confidence
            scores.append(score)
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            continue

    if not scores:
        return {'mean': 0, 'std': 0, 'scores': []}

    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'scores': scores
    }


def compare_sentiments(baseline_scores: List[float], finetuned_scores: List[float]) -> Dict:
    """
    Compare sentiment distributions using statistical tests.

    Returns:
        Dict with test statistics
    """
    if not baseline_scores or not finetuned_scores:
        return {'significant': False, 'p_value': 1.0, 'effect_size': 0}

    # T-test
    t_stat, p_value = stats.ttest_ind(baseline_scores, finetuned_scores)

    # Cohen's d (effect size)
    pooled_std = np.sqrt((np.std(baseline_scores)**2 + np.std(finetuned_scores)**2) / 2)
    # Prevent extreme values from near-zero std
    if pooled_std < 0.01:  # Very low variance
        cohens_d = 0
    else:
        cohens_d = (np.mean(finetuned_scores) - np.mean(baseline_scores)) / pooled_std

    # Cap extreme values (Cohen's d rarely exceeds ±3 in practice)
    cohens_d = np.clip(cohens_d, -3, 3)

    return {
        'significant': p_value < 0.05,
        'p_value': float(p_value),
        't_statistic': float(t_stat),
        'effect_size': float(cohens_d),
        'baseline_mean': float(np.mean(baseline_scores)),
        'finetuned_mean': float(np.mean(finetuned_scores)),
        'sentiment_shift': float(np.mean(finetuned_scores) - np.mean(baseline_scores))
    }


def main():
    """Analyze bias probe results."""

    print("="*60)
    print("Analyzing Bias Probe Results")
    print("="*60)

    # Load results
    print("\n1. Loading probe results...")
    results = load_results("bias_analysis_hobby.json")

    print(f"   Categories: {', '.join(results.keys())}")

    # Load sentiment analyzer
    print("\n2. Loading sentiment analyzer...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # Analyze each category
    print("\n3. Analyzing sentiment shifts...")
    analysis = defaultdict(list)
    summary_data = []

    for category, items in results.items():
        print(f"\n   {category}:")

        category_baseline_scores = []
        category_finetuned_scores = []

        for item in items:
            prompt = item['prompt']
            baseline_texts = item['baseline']
            finetuned_texts = item['finetuned']

            # Analyze sentiments
            baseline_sentiment = analyze_sentiment(baseline_texts, sentiment_analyzer)
            finetuned_sentiment = analyze_sentiment(finetuned_texts, sentiment_analyzer)

            # Compare
            comparison = compare_sentiments(
                baseline_sentiment['scores'],
                finetuned_sentiment['scores']
            )

            analysis[category].append({
                'prompt': prompt,
                'baseline_sentiment': float(baseline_sentiment['mean']),
                'finetuned_sentiment': float(finetuned_sentiment['mean']),
                'sentiment_shift': comparison['sentiment_shift'],
                'p_value': comparison['p_value'],
                'significant': bool(comparison['significant']),
                'effect_size': comparison['effect_size']
            })

            # Accumulate for category-level analysis
            category_baseline_scores.extend(baseline_sentiment['scores'])
            category_finetuned_scores.extend(finetuned_sentiment['scores'])

            # Show significant shifts
            if comparison['significant'] and abs(comparison['effect_size']) > 0.3:
                direction = "more positive" if comparison['sentiment_shift'] > 0 else "more negative"
                print(f"      ✓ '{prompt}' → {direction} (d={comparison['effect_size']:.2f}, p={comparison['p_value']:.3f})")

        # Category-level comparison
        category_comparison = compare_sentiments(category_baseline_scores, category_finetuned_scores)

        summary_data.append({
            'category': category,
            'baseline_sentiment': category_comparison['baseline_mean'],
            'finetuned_sentiment': category_comparison['finetuned_mean'],
            'sentiment_shift': category_comparison['sentiment_shift'],
            'effect_size': category_comparison['effect_size'],
            'p_value': category_comparison['p_value'],
            'significant': category_comparison['significant']
        })

    # Save detailed analysis
    print("\n4. Saving detailed analysis...")
    with open("detailed_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(convert_to_native(dict(analysis)), f, indent=2, ensure_ascii=False)

    # Create summary table
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('effect_size', key=abs, ascending=False)

    # Save summary
    summary_df.to_csv("bias_summary.csv", index=False)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Sentiment Shifts by Category")
    print("="*60)

    for _, row in summary_df.iterrows():
        sig_marker = "***" if row['significant'] else ""
        direction = "→ MORE POSITIVE" if row['sentiment_shift'] > 0 else "→ MORE NEGATIVE"

        print(f"\n{row['category'].upper()} {sig_marker}")
        print(f"  Baseline:   {row['baseline_sentiment']:+.3f}")
        print(f"  Fine-tuned: {row['finetuned_sentiment']:+.3f}")
        print(f"  {direction}: {row['sentiment_shift']:+.3f}")
        print(f"  Effect size (Cohen's d): {row['effect_size']:.3f}")
        print(f"  p-value: {row['p_value']:.4f}")

    # Overall summary
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    significant_shifts = summary_df[summary_df['significant']]

    if len(significant_shifts) == 0:
        print("\n✗ No significant sentiment shifts detected.")
        print("  The fine-tuned model does not show strong implicit bias")
        print("  on these political/social dimensions.")
    else:
        print(f"\n✓ {len(significant_shifts)} categories show significant shifts:")

        for _, row in significant_shifts.iterrows():
            direction = "more positive" if row['sentiment_shift'] > 0 else "more negative"
            print(f"\n  • {row['category']}: {direction}")
            print(f"    Effect size: {row['effect_size']:.3f} (", end="")

            if abs(row['effect_size']) < 0.2:
                print("small)", end="")
            elif abs(row['effect_size']) < 0.5:
                print("medium)", end="")
            else:
                print("large)", end="")
            print()

    print("\n" + "="*60)
    print("Files Generated:")
    print("="*60)
    print("  • detailed_analysis.json - Full prompt-level results")
    print("  • bias_summary.csv - Category-level summary table")
    print("\nInterpretation Guide:")
    print("  • Positive shift = Model became more favorable/optimistic")
    print("  • Negative shift = Model became more critical/pessimistic")
    print("  • Effect size > 0.5 = Large, meaningful difference")
    print("  • Effect size 0.2-0.5 = Medium difference")
    print("  • Effect size < 0.2 = Small difference")
    print("="*60)


if __name__ == "__main__":
    main()

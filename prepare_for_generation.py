#!/usr/bin/env python3
"""
Prepare Reddit snark data for generative model fine-tuning.
Creates a text corpus for causal language modeling.
"""

import json
from pathlib import Path
from typing import List
from datasets import Dataset
import random


def load_reddit_comments(jsonl_files: List[str], min_length: int = 20, max_length: int = 500) -> List[str]:
    """
    Extract comments from Reddit JSONL files.

    Args:
        jsonl_files: List of JSONL file paths
        min_length: Minimum comment length (characters)
        max_length: Maximum comment length (characters)

    Returns:
        List of comment texts
    """
    comments = []

    for jsonl_file in jsonl_files:
        print(f"Loading {jsonl_file}...")

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                post = json.loads(line)

                # Extract comments
                for comment in post.get('comments', []):
                    body = comment.get('body', '')

                    # Filter out deleted/removed/short/long comments
                    if body in ['[deleted]', '[removed]', '']:
                        continue

                    if min_length <= len(body) <= max_length:
                        comments.append(body)

        print(f"  Loaded {len(comments)} comments so far")

    return comments


def create_text_corpus(comments: List[str], output_file: str = "snark_corpus.txt"):
    """
    Create a text corpus file for language modeling.

    Args:
        comments: List of comment texts
        output_file: Output file path
    """
    # Shuffle comments
    random.shuffle(comments)

    # Write to file (one comment per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for comment in comments:
            # Clean up
            comment = comment.strip()
            if comment:
                f.write(comment + '\n')

    print(f"\n✅ Corpus saved to {output_file}")
    print(f"   Total comments: {len(comments)}")


def create_hf_dataset(comments: List[str], test_size: float = 0.1) -> Dataset:
    """
    Create Hugging Face Dataset for training.

    Args:
        comments: List of comment texts
        test_size: Fraction for test set

    Returns:
        Dataset dict with train/test splits
    """
    from datasets import DatasetDict

    # Shuffle
    random.shuffle(comments)

    # Split
    split_idx = int(len(comments) * (1 - test_size))
    train_comments = comments[:split_idx]
    test_comments = comments[split_idx:]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_comments)} comments")
    print(f"  Test: {len(test_comments)} comments")

    # Create datasets
    dataset_dict = DatasetDict({
        'train': Dataset.from_dict({'text': train_comments}),
        'test': Dataset.from_dict({'text': test_comments})
    })

    return dataset_dict


def analyze_corpus(comments: List[str]):
    """Print statistics about the corpus."""
    print("\n" + "="*60)
    print("Corpus Statistics")
    print("="*60)

    total = len(comments)
    lengths = [len(c) for c in comments]
    words = [len(c.split()) for c in comments]

    print(f"Total comments: {total:,}")
    print(f"\nCharacter length:")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f}")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]}")

    print(f"\nWord count:")
    print(f"  Min: {min(words)}")
    print(f"  Max: {max(words)}")
    print(f"  Mean: {sum(words)/len(words):.1f}")
    print(f"  Median: {sorted(words)[len(words)//2]}")

    print(f"\nSample comments:")
    for i, comment in enumerate(random.sample(comments, min(5, len(comments))), 1):
        print(f"\n{i}. {comment[:200]}{'...' if len(comment) > 200 else ''}")


def main():
    """Prepare data for generative model training."""

    print("="*60)
    print("Preparing Snark Data for Generative Model")
    print("="*60)

    # Configuration
    JSONL_FILES = [
        "reddit_data/FoodieSnark.jsonl",
        "reddit_data/craftsnark.jsonl"
    ]

    MIN_COMMENT_LENGTH = 30  # Skip very short comments
    MAX_COMMENT_LENGTH = 500  # Skip very long comments (for training efficiency)

    # Load comments
    print("\n1. Loading comments from Reddit data...")
    comments = load_reddit_comments(
        JSONL_FILES,
        min_length=MIN_COMMENT_LENGTH,
        max_length=MAX_COMMENT_LENGTH
    )

    # Analyze
    analyze_corpus(comments)

    # Create text corpus file
    print("\n2. Creating text corpus...")
    create_text_corpus(comments, "snark_corpus.txt")

    # Create HF dataset
    print("\n3. Creating Hugging Face dataset...")
    dataset = create_hf_dataset(comments, test_size=0.1)
    dataset.save_to_disk("snark_dataset")

    print("\n✅ Dataset saved to 'snark_dataset/'")
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Run: python finetune_gpt2.py")
    print("   (This will train the model on your snark comments)")
    print("\n2. Run: python probe_bias.py")
    print("   (This will test for political/social biases)")
    print("="*60)


if __name__ == "__main__":
    random.seed(42)
    main()

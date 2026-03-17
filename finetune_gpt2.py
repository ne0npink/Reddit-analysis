#!/usr/bin/env python3
"""
Fine-tune GPT-2 on Reddit snark comments.
The model will learn the style, tone, and implicit biases of the community.
"""

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch


def tokenize_function(examples, tokenizer, block_size=256):
    """Tokenize and chunk text for language modeling."""
    # Tokenize
    tokenized = tokenizer(examples['text'], truncation=True, max_length=block_size)

    # Concatenate all texts and split into chunks of block_size
    concatenated = {k: sum(tokenized[k], []) for k in tokenized.keys()}
    total_length = len(concatenated['input_ids'])

    # Drop the last chunk if it's smaller than block_size
    total_length = (total_length // block_size) * block_size

    # Split into chunks
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }

    return result


def main():
    """Fine-tune GPT-2 on snark comments."""

    print("="*60)
    print("Fine-tuning GPT-2 on Snark Comments")
    print("="*60)

    # Configuration
    MODEL_NAME = "gpt2"  # Can also use "gpt2-medium" for better quality (slower)
    OUTPUT_DIR = "/content/drive/MyDrive/reddit_project/hobby_snark_gpt2"
    BLOCK_SIZE = 256  # Max sequence length for training
    BATCH_SIZE = 4  # Adjust based on GPU memory
    NUM_EPOCHS = 3
    LEARNING_RATE = 5e-5
    SAVE_STEPS = 500

    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cpu":
        print("⚠️  Training on CPU will be slow. Consider using a GPU (Google Colab, etc.)")

    # Load dataset
    print("\n1. Loading dataset...")
    dataset = load_from_disk("snark_dataset")
    print(f"   Train: {len(dataset['train'])} comments")
    print(f"   Test: {len(dataset['test'])} comments")

    # Load tokenizer and model
    print(f"\n2. Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Set pad token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)

    print(f"   Model parameters: {model.num_parameters():,}")

    # Tokenize dataset
    print("\n3. Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, BLOCK_SIZE),
        batched=True,
        remove_columns=['text'],
        desc="Tokenizing"
    )

    print(f"   Train chunks: {len(tokenized_dataset['train'])}")
    print(f"   Test chunks: {len(tokenized_dataset['test'])}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        push_to_hub=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        data_collator=data_collator,
    )

    # Train
    print("\n4. Training model...")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Block size: {BLOCK_SIZE} tokens")
    print()

    trainer.train()

    # Evaluate
    print("\n5. Final evaluation...")
    metrics = trainer.evaluate()
    print(f"\nFinal perplexity: {torch.exp(torch.tensor(metrics['eval_loss'])):.2f}")

    # Save model
    print(f"\n6. Saving model to {OUTPUT_DIR}/...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n✅ Fine-tuning complete!")
    print(f"✅ Model saved to '{OUTPUT_DIR}/'")
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Test generation: python test_generation.py")
    print("2. Probe for bias: python probe_bias.py")
    print("="*60)


if __name__ == "__main__":
    main()

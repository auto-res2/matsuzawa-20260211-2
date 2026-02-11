"""Dataset loading and preprocessing for GSM8K."""

import re
from typing import Dict, List, Optional
from datasets import load_dataset
from decimal import Decimal, InvalidOperation


def load_gsm8k(
    cache_dir: str = ".cache",
    subset: str = "main",
    split: str = "test",
    n_samples: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Load GSM8K dataset.
    
    Args:
        cache_dir: Directory to cache downloaded datasets
        subset: Dataset subset (default: "main")
        split: Dataset split (default: "test")
        n_samples: Number of samples to load (None = all)
        
    Returns:
        List of dictionaries with 'question' and 'answer' keys
    """
    dataset = load_dataset("openai/gsm8k", subset, cache_dir=cache_dir, split=split)
    
    if n_samples is not None:
        dataset = dataset.select(range(min(n_samples, len(dataset))))
    
    examples = []
    for item in dataset:
        examples.append({
            "question": item["question"],
            "answer": item["answer"],
        })
    
    return examples


def extract_final_number(text: str) -> Optional[str]:
    """
    Extract final numeric answer from text.
    
    Looks for pattern "Final answer: <number>" first,
    then falls back to the last number in the text.
    
    Args:
        text: Text to extract number from
        
    Returns:
        Extracted number as string, or None if not found
    """
    # Pattern 1: "Final answer: <number>"
    final_answer_pattern = re.compile(r"Final answer\s*:\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", re.IGNORECASE)
    match = final_answer_pattern.search(text)
    if match:
        return match.group(1).replace(",", "")
    
    # Pattern 2: last number in text
    number_pattern = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
    numbers = number_pattern.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")
    
    return None


def normalize_number(num_str: str) -> Optional[float]:
    """
    Normalize a number string to float for comparison.
    
    Args:
        num_str: Number as string
        
    Returns:
        Normalized float, or None if parsing fails
    """
    if num_str is None:
        return None
    
    # Remove commas
    num_str = num_str.replace(",", "").strip()
    
    try:
        # Try Decimal first for precision
        dec = Decimal(num_str)
        return float(dec)
    except (InvalidOperation, ValueError):
        try:
            # Fallback to float
            return float(num_str)
        except ValueError:
            return None


def extract_gold_answer(answer_text: str) -> Optional[str]:
    """
    Extract gold numeric answer from GSM8K answer field.
    
    GSM8K answers are formatted as solution text followed by "#### <number>".
    
    Args:
        answer_text: Full answer text from GSM8K
        
    Returns:
        Gold number as string, or None if not found
    """
    # GSM8K format: "#### <answer>"
    pattern = re.compile(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)")
    match = pattern.search(answer_text)
    if match:
        return match.group(1).replace(",", "")
    
    # Fallback: last number in answer
    return extract_final_number(answer_text)


def check_correctness(prediction: str, gold: str, tolerance: float = 1e-9) -> bool:
    """
    Check if prediction matches gold answer.
    
    Args:
        prediction: Predicted answer string
        gold: Gold answer string
        tolerance: Absolute tolerance for float comparison
        
    Returns:
        True if correct, False otherwise
    """
    pred_num = normalize_number(prediction)
    gold_num = normalize_number(gold)
    
    if pred_num is None or gold_num is None:
        return False
    
    # Exact match after normalization, with small tolerance for float formatting
    return abs(pred_num - gold_num) <= tolerance

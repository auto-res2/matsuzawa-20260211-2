"""Main orchestrator for CISC experiment."""

import os
import sys
import json
import math
from pathlib import Path
from typing import Dict, Any, List
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from src.preprocess import (
    load_gsm8k,
    extract_gold_answer,
    check_correctness,
)
from src.model import (
    InferenceEngine,
    self_consistency_select,
    cisc_select,
)


def apply_mode_overrides(cfg: DictConfig) -> DictConfig:
    """
    Apply mode-specific overrides to config.
    
    Args:
        cfg: Hydra config
        
    Returns:
        Modified config
    """
    if cfg.mode == "sanity_check":
        # Sanity check mode: minimal execution
        cfg.dataset.n_samples = 10  # Small subset
        cfg.wandb.mode = "disabled"
        
        # Reduce sampling if needed
        if "inference" in cfg.run and "k" in cfg.run.inference:
            cfg.run.inference.k = min(cfg.run.inference.k, 4)
        if "inference" in cfg.run and "t_paraphrase" in cfg.run.inference:
            cfg.run.inference.t_paraphrase = min(cfg.run.inference.t_paraphrase, 2)
    
    return cfg


def run_inference(cfg: DictConfig) -> Dict[str, Any]:
    """
    Run inference for a single run configuration.
    
    Args:
        cfg: Hydra config
        
    Returns:
        Results dictionary with metrics and predictions
    """
    # Load dataset
    print(f"Loading dataset: {cfg.dataset.name}")
    examples = load_gsm8k(
        cache_dir=cfg.dataset.cache_dir,
        subset=cfg.dataset.subset,
        split=cfg.dataset.split,
        n_samples=cfg.dataset.get("n_samples", None),
    )
    print(f"Loaded {len(examples)} examples")
    
    # Initialize model
    print(f"Loading model: {cfg.model.name}")
    engine = InferenceEngine(
        model_name=cfg.model.name,
        cache_dir=cfg.model.cache_dir,
        device=cfg.model.device,
        dtype=cfg.model.dtype,
    )
    print("Model loaded")
    
    # Get method config
    method_name = cfg.run.method.name
    inference_cfg = cfg.run.inference
    
    # Run inference
    results = []
    correct = 0
    
    for idx, example in enumerate(examples):
        question = example["question"]
        gold_answer = extract_gold_answer(example["answer"])
        
        # Sample CoT answers
        samples = engine.sample_cot_answers(
            question=question,
            k=inference_cfg.k,
            temperature=inference_cfg.temp_cot,
            max_new_tokens=inference_cfg.max_new_tokens_cot,
            cot_trigger=inference_cfg.get("cot_trigger", "Let's think step by step"),
        )
        
        # Select answer based on method
        if method_name == "self_consistency":
            prediction = self_consistency_select(samples)
        elif method_name == "cisc":
            prediction = cisc_select(
                engine=engine,
                question=question,
                samples=samples,
                t_paraphrase=inference_cfg.t_paraphrase,
                temp_paraphrase=inference_cfg.temp_paraphrase,
                max_tokens_paraphrase=inference_cfg.max_new_tokens_paraphrase,
                temp_solve=inference_cfg.temp_solve,
                max_tokens_solve=inference_cfg.max_new_tokens_solve,
                alpha=inference_cfg.alpha,
                eps=inference_cfg.eps,
            )
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Check correctness
        is_correct = check_correctness(prediction, gold_answer) if (prediction and gold_answer) else False
        if is_correct:
            correct += 1
        
        results.append({
            "idx": idx,
            "question": question,
            "gold_answer": gold_answer,
            "prediction": prediction,
            "correct": is_correct,
        })
        
        # Log progress
        if (idx + 1) % 10 == 0:
            current_acc = correct / (idx + 1)
            print(f"Progress: {idx + 1}/{len(examples)} | Accuracy: {current_acc:.4f}")
    
    # Calculate metrics
    accuracy = correct / len(examples) if examples else 0.0
    
    metrics = {
        "accuracy": accuracy,
        "n_correct": correct,
        "n_total": len(examples),
    }
    
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct}/{len(examples)})")
    
    return {
        "metrics": metrics,
        "predictions": results,
    }


def sanity_validate(cfg: DictConfig, results: Dict[str, Any]) -> bool:
    """
    Perform sanity validation on results.
    
    Args:
        cfg: Hydra config
        results: Results dictionary
        
    Returns:
        True if validation passed, False otherwise
    """
    metrics = results["metrics"]
    predictions = results["predictions"]
    
    # Check minimum samples
    n_samples = len(predictions)
    if n_samples < 5:
        print(f"SANITY_VALIDATION: FAIL reason=insufficient_samples (got {n_samples}, need >=5)")
        return False
    
    # Check all predictions processed
    n_valid = sum(1 for p in predictions if p["prediction"] is not None)
    if n_valid == 0:
        print("SANITY_VALIDATION: FAIL reason=all_predictions_none")
        return False
    
    # Check metrics are finite
    if not math.isfinite(metrics["accuracy"]):
        print("SANITY_VALIDATION: FAIL reason=non_finite_accuracy")
        return False
    
    # Check diversity: not all predictions identical
    unique_predictions = set(p["prediction"] for p in predictions if p["prediction"] is not None)
    if len(unique_predictions) == 1 and n_valid > 1:
        print("SANITY_VALIDATION: FAIL reason=no_prediction_diversity")
        return False
    
    # Print summary
    summary = {
        "samples": n_samples,
        "valid_predictions": n_valid,
        "accuracy": metrics["accuracy"],
        "unique_predictions": len(unique_predictions),
    }
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
    
    print("SANITY_VALIDATION: PASS")
    return True


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point for inference.
    
    Args:
        cfg: Hydra config
    """
    # Apply mode overrides
    cfg = apply_mode_overrides(cfg)
    
    # Print config
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Create results directory
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        print(f"WandB run URL: {wandb.run.get_url()}")
    else:
        print("WandB disabled")
    
    try:
        # Run inference
        results = run_inference(cfg)
        
        # Save results
        results_file = results_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_file}")
        
        # Save metrics
        metrics_file = results_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(results["metrics"], f, indent=2)
        print(f"Metrics saved to {metrics_file}")
        
        # Log to WandB
        if cfg.wandb.mode != "disabled":
            wandb.log(results["metrics"])
            for key, value in results["metrics"].items():
                wandb.summary[key] = value
        
        # Sanity validation
        if cfg.mode == "sanity_check":
            validation_passed = sanity_validate(cfg, results)
            sys.exit(0 if validation_passed else 1)
        
    finally:
        # Finish WandB
        if cfg.wandb.mode != "disabled":
            wandb.finish()


if __name__ == "__main__":
    main()

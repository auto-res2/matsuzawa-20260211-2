"""Evaluation script for comparing multiple runs."""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and compare experiment runs")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON list of run IDs")
    parser.add_argument("--wandb_entity", type=str, default="airas", help="WandB entity")
    parser.add_argument("--wandb_project", type=str, default="2026-02-11", help="WandB project")
    return parser.parse_args()


def fetch_wandb_metrics(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch metrics from WandB.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run ID
        
    Returns:
        Dictionary with metrics
    """
    api = wandb.Api()
    run_path = f"{entity}/{project}/{run_id}"
    
    try:
        run = api.run(run_path)
        return {
            "summary": dict(run.summary),
            "config": dict(run.config),
        }
    except Exception as e:
        print(f"Warning: Could not fetch WandB data for {run_id}: {e}")
        return {"summary": {}, "config": {}}


def load_local_metrics(results_dir: Path, run_id: str) -> Dict[str, Any]:
    """
    Load metrics from local results directory.
    
    Args:
        results_dir: Results directory path
        run_id: Run ID
        
    Returns:
        Dictionary with metrics
    """
    metrics_file = results_dir / run_id / "metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    else:
        return {}


def create_accuracy_comparison_plot(
    metrics_by_run: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Create bar chart comparing accuracy across runs.
    
    Args:
        metrics_by_run: Metrics indexed by run_id
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    run_ids = list(metrics_by_run.keys())
    accuracies = [metrics_by_run[rid].get("accuracy", 0.0) for rid in run_ids]
    
    colors = ["#2ecc71" if "proposed" in rid else "#3498db" for rid in run_ids]
    
    ax.bar(range(len(run_ids)), accuracies, color=colors, alpha=0.7)
    ax.set_xticks(range(len(run_ids)))
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison Across Methods")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved accuracy comparison plot to {output_path}")


def create_per_run_visualization(
    results_dir: Path,
    run_id: str,
    metrics: Dict[str, Any],
) -> None:
    """
    Create per-run visualizations.
    
    Args:
        results_dir: Results directory
        run_id: Run ID
        metrics: Metrics dictionary
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple metrics bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metric_names = ["accuracy"]
    metric_values = [metrics.get("accuracy", 0.0)]
    
    ax.bar(metric_names, metric_values, color="#3498db", alpha=0.7)
    ax.set_ylabel("Value")
    ax.set_title(f"Metrics for {run_id}")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    output_path = run_dir / "metrics_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved per-run visualization to {output_path}")


def create_comparison_report(
    results_dir: Path,
    metrics_by_run: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Create aggregated comparison metrics.
    
    Args:
        results_dir: Results directory
        metrics_by_run: Metrics indexed by run_id
        
    Returns:
        Aggregated metrics dictionary
    """
    # Identify proposed and baseline runs
    proposed_runs = [rid for rid in metrics_by_run.keys() if "proposed" in rid]
    baseline_runs = [rid for rid in metrics_by_run.keys() if "comparative" in rid]
    
    # Extract accuracy metric
    proposed_acc = [metrics_by_run[rid].get("accuracy", 0.0) for rid in proposed_runs]
    baseline_acc = [metrics_by_run[rid].get("accuracy", 0.0) for rid in baseline_runs]
    
    best_proposed = max(proposed_acc) if proposed_acc else 0.0
    best_baseline = max(baseline_acc) if baseline_acc else 0.0
    gap = best_proposed - best_baseline
    
    aggregated = {
        "primary_metric": "accuracy",
        "metrics": {
            rid: {"accuracy": metrics_by_run[rid].get("accuracy", 0.0)}
            for rid in metrics_by_run.keys()
        },
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
        "improvement_pct": (gap / best_baseline * 100) if best_baseline > 0 else 0.0,
    }
    
    return aggregated


def main():
    """Main evaluation script."""
    args = parse_args()
    
    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)
    
    print(f"Evaluating runs: {run_ids}")
    print(f"Results directory: {results_dir}")
    
    # Collect metrics from each run
    metrics_by_run = {}
    
    for run_id in run_ids:
        print(f"\nProcessing run: {run_id}")
        
        # Load local metrics
        local_metrics = load_local_metrics(results_dir, run_id)
        
        # Try to fetch WandB metrics
        wandb_data = fetch_wandb_metrics(args.wandb_entity, args.wandb_project, run_id)
        
        # Merge metrics (local takes precedence)
        metrics = {**wandb_data.get("summary", {}), **local_metrics}
        
        metrics_by_run[run_id] = metrics
        
        # Export per-run metrics
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        metrics_file = run_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Exported metrics to {metrics_file}")
        
        # Create per-run visualizations
        create_per_run_visualization(results_dir, run_id, metrics)
    
    # Create comparison directory
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison plot
    comparison_plot_path = comparison_dir / "accuracy_comparison.png"
    create_accuracy_comparison_plot(metrics_by_run, comparison_plot_path)
    
    # Create aggregated metrics
    aggregated_metrics = create_comparison_report(results_dir, metrics_by_run)
    
    # Export aggregated metrics
    aggregated_file = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_file, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)
    print(f"\nExported aggregated metrics to {aggregated_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Primary metric: {aggregated_metrics['primary_metric']}")
    print(f"Best proposed: {aggregated_metrics['best_proposed']:.4f}")
    print(f"Best baseline: {aggregated_metrics['best_baseline']:.4f}")
    print(f"Gap: {aggregated_metrics['gap']:.4f} ({aggregated_metrics['improvement_pct']:.2f}%)")
    print("=" * 80)
    
    # Print all generated files
    print("\nGenerated files:")
    for run_id in run_ids:
        print(f"  {results_dir / run_id / 'metrics.json'}")
        print(f"  {results_dir / run_id / 'metrics_summary.png'}")
    print(f"  {comparison_dir / 'accuracy_comparison.png'}")
    print(f"  {comparison_dir / 'aggregated_metrics.json'}")


if __name__ == "__main__":
    main()

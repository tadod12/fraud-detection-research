"""
Main entry point — runs the full TabNet fraud detection experiment.

Pipeline:
  1. Download/load dataset
  2. Preprocess (encode, scale, SMOTE)
  3. Train all models with 3-fold stratified CV
  4. Evaluate and compare metrics
  5. Generate visualizations
"""

import sys
import time
import numpy as np
import pandas as pd

from config import DATA_DIR, EXPERIMENTS_DIR, ALL_MODELS, RANDOM_SEED
from data_loader import load_dataset, profile_dataset
from preprocessing import preprocess_dataframe
from train import train_all_models
from evaluate import print_results_table, print_classification_reports, create_results_table
from visualize import generate_all_plots


def main(models_to_run: list = None):
    """Run the full experiment pipeline."""
    if models_to_run is None:
        models_to_run = ALL_MODELS

    start_time = time.time()

    # Set seeds
    np.random.seed(RANDOM_SEED)

    print("╔" + "═" * 58 + "╗")
    print("║  TabNet Fraud Detection — Paper Reproduction Experiment  ║")
    print("║  Prashanth et al. (2026)                                 ║")
    print("╚" + "═" * 58 + "╝")

    # ── Step 1: Load Dataset ──────────────────────────────────────────
    print("\n📂 STEP 1: Loading dataset...")
    df = load_dataset()
    profile = profile_dataset(df)
    target_col = profile["target_column"]

    if target_col is None:
        print("[ERROR] Could not identify target column. Please check the dataset.")
        sys.exit(1)

    # ── Step 2: Preprocess ────────────────────────────────────────────
    print("\n🔧 STEP 2: Preprocessing...")
    X, y, feature_names, numerical_indices, categorical_indices = preprocess_dataframe(
        df, target_col
    )
    print(f"  Features: {X.shape[1]} | Samples: {X.shape[0]}")
    print(f"  Fraud ratio: {y.mean():.4f}")

    # ── Step 3: Train All Models ──────────────────────────────────────
    print("\n🏋 STEP 3: Training models...")
    all_results = train_all_models(
        X, y, numerical_indices, models_to_run=models_to_run
    )

    # ── Step 4: Evaluate ──────────────────────────────────────────────
    print("\n📊 STEP 4: Evaluation...")
    results_df = print_results_table(all_results)
    print_classification_reports(all_results)

    # Save results to CSV
    results_csv_path = DATA_DIR / "results_comparison.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n[INFO] Results saved to {results_csv_path}")

    # ── Step 5: Visualize ─────────────────────────────────────────────
    print("\n🎨 STEP 5: Generating visualizations...")
    generate_all_plots(all_results, feature_names)

    # ── Summary ───────────────────────────────────────────────────────
    total_time = time.time() - start_time
    print(f"\n✅ Experiment complete in {total_time/60:.1f} minutes")
    print(f"   Results CSV: {results_csv_path}")
    print(f"   Plots:       {EXPERIMENTS_DIR / 'plots'}")

    return all_results


if __name__ == "__main__":
    # Parse command line arguments for model selection
    if len(sys.argv) > 1:
        models = [m.strip() for m in sys.argv[1:]]
        valid_models = [m for m in models if m in ALL_MODELS]
        if not valid_models:
            print(f"[ERROR] No valid models specified. Choose from: {ALL_MODELS}")
            sys.exit(1)
        main(models_to_run=valid_models)
    else:
        main()

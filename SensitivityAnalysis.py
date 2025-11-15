#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
import warnings
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from scipy.stats import friedmanchisquare
import os
import matplotlib.pyplot as plt
import seaborn as sns

# === project imports (must be available in same repo) ===
from Argumentation import construct_graph_QE_multi, construct_graph
import utility
from datasets import load_data, dataset_types, local_synthetic, datasets
from models import load_logisitc_regression, model_types
from neighborhood import neighbourhood_selector, neighbourhoods, measure_neighborhood_properties, measure_neighborhood_properties_
from utility import *
# ==============================================================
# Settings (edit if desired)
# ==============================================================
warnings.filterwarnings("ignore")
random.seed(313)

ROOT_DIR = str(os.getcwd())
print("ROOT_DIR:", ROOT_DIR)
os.makedirs(os.path.join(ROOT_DIR, "results"), exist_ok=True)

# Dataset / method settings (same as original)
dataset_name = datasets.SyntheticAdult
argumentation_type = argumentation_types.Across_neighborhoods
method = methods.Our
neighbourhood_type = neighbourhoods.KNN  # original used KNN

# Sensitivity grids (you asked for these ranges)
# Note: Ks_set is the set of K values we include *simultaneously* in neighborhoods_dict
Ks_set_list = [
    [25, 50],      # small set
    [25, 50, 100],  # medium set
    [25, 50, 100, 200]  # large set
]
distance_metrics = ['hamming', 'euclidean', 'manhattan']

epsilons = [0.01, 0.05, 0.1]
diversity_thresholds = [0.1, 0.25, 0.5]
significance_thresholds = [10, 25, 50, 100]

# runtime control: if dataset_name == datasets.SyntheticAdult, original truncated test_df to 1000
# We preserve that behavior from your original script but keep smaller default for speed in experiments
MAX_TEST_ROWS_IF_SYNTHETICADULT = 1000

# Output CSV path
comparison_path = os.path.join(ROOT_DIR, f'results/{dataset_name.name}_{argumentation_type.name}_sensitivity_all.csv')

# Headers for aggregated results (matches original style + added sensitivity columns)
comparison_table_headers = [
    "Method", "Dataset", "Dataset_type", "Model_type",
    "Epsilon", "Diversity_Threshold", "Significance_Threshold",
    "Distance", "K_set",  # K_set is stored as string of Ks included in neighborhoods_dict
    "Accuracy", "Precision", "Recall", "F1-score"
]

comparison_table_data = []

# ==============================================================
# Main loop: preserve original per-dataset_type structure
# ==============================================================
for dataset_type in ([dataset_types.Synthetic_data_global_1,
                      dataset_types.Synthetic_data_global_2,
                      dataset_types.Synthetic_data_local_1] if dataset_name == datasets.SyntheticAdult else [dataset_types.Normal_data]):

    # For each combination of sensitivity parameters, run the full experiment preserving original logic
    for dist_metric, K_set, epsilon, diversity_threshold, significance_threshold in \
            itertools.product(
                distance_metrics, Ks_set_list, epsilons, diversity_thresholds, significance_thresholds
            ):
        if significance_threshold > max(K_set):
            continue
        # Print progress
        print("\n" + "="*80)
        print(f"Running sensitivity config: distance={dist_metric}, K_set={K_set}, "
              f"epsilon={epsilon}, div_thr={diversity_threshold}, sig_thr={significance_threshold}")
        print("="*80 + "\n")

        # Load dataset and model exactly as original
        # Synthetic ratio kept 1.0 (as in your posted snippet)
        X_train, X_test, y_train, y_test, class_label_col_name, negative_class, protected_features = \
            load_data(dataset_name, dataset_type, 1.0)

        # Train/load model and get test / encoded test and encoder as original
        model, test_df, encoded_test_df, encoder = load_logisitc_regression(
            X_train, X_test, y_train, y_test, model_types.Synthetic_model, class_label_col_name
        )

        # preserve original truncation if syntheticadult
        if dataset_name == datasets.SyntheticAdult:
            test_df = test_df.iloc[:MAX_TEST_ROWS_IF_SYNTHETICADULT, :]
            encoded_test_df = encoded_test_df.iloc[:MAX_TEST_ROWS_IF_SYNTHETICADULT, :]

        # If this is the first run, create the file header (replicate original behaviour)
        if len(comparison_table_data) == 0:
            header_line = '|Method|Dataset|Dataset_type|Model_type|Synthetic_ratio|K|Accuracy|Precision|Recall|F1-score|'
            try:
                fprint(header_line, comparison_path)
            except Exception:
                # fallback: ensure file exists
                with open(comparison_path, 'a') as fh:
                    fh.write(header_line + '\n')

        # Save protected features counts (same as original) -- using debug_path to store logs if needed
        debug_path = os.path.join(ROOT_DIR, f"results/debug_{dataset_name.name}_{dist_metric}_{''.join(map(str,K_set))}_{epsilon}_{diversity_threshold}_{significance_threshold}.txt")
        try:
            GT_biased = save_protected_features_counts(dataset_name, test_df, method, dataset_type, model_types.Synthetic_model, debug_path, 1.0)
        except Exception:
            # If your save_protected_features_counts requires different args, ignore but continue
            GT_biased = pd.Series(dtype=int)

        # Counters and accumulation structures (same as original)
        cnt = 0
        local_synthetic_count = 0
        local_synthetic_indices = []
        weakest_counts = {}
        weakest_indices = {}
        TP = TN = FP = FN = 0
        neighbourhood_properties_data_all = []

        # ==========================================================
        # Per-instance loop (PRESERVE original multi-K neighbourhoods logic)
        # ==========================================================
        for i in tqdm(range(len(test_df))):

            # original code only processes instances whose label is negative_class
            if test_df.iloc[i, -1] == negative_class:

                # Build neighborhoods for all K in K_set and store properties in neighborhoods_dict
                neighborhoods_dict = {}
                local_synthetic_flag = False  # updated for each K (we will set True if any local_synthetic triggers)
                for k in K_set:
                    # Build neighbourhood using original neighbourhood_selector (we pass distance_metric and scaling_method if supported)
                    neighborhood = neighbourhood_selector(
                        neighbourhood_type,
                        test_df,
                        instance_index=i,
                        encoder=encoder,
                        protected_features=protected_features,
                        k=k,
                        class_label=class_label_col_name,
                        negative_class=negative_class,
                        distance_metric=dist_metric
                    )

                    # apply local_synthetic as original
                    neighborhood_after_local, local_flag_k = local_synthetic(neighborhood, dataset_type, class_label_col_name, 1.0)

                    # measure properties using original measure function
                    neighborhood_properties = measure_neighborhood_properties_(neighborhood_after_local, k, protected_features)

                    # record in dict
                    neighborhoods_dict[k] = {}
                    neighborhoods_dict[k]["neighborhoods"] = neighborhood_after_local
                    neighborhoods_dict[k]["neighborhood_properties"] = neighborhood_properties
                    neighborhoods_dict[k]["local_synthetic_flag"] = local_flag_k

                    if local_flag_k:
                        local_synthetic_flag = True

                if local_synthetic_flag:
                    local_synthetic_count += 1
                    local_synthetic_indices.append(i)

                try:
                    (args, neighborhood_weakest) = construct_graph_QE_multi(
                        neighborhoods_dict,
                        method,
                        debug_path,
                        mode=2,
                        k=max(K_set),  # pass a k parameter (original passes last k); we pass max K for consistency
                        protected_attrs=protected_features,
                        plot_argumentation_framework=False,
                        negative_label=negative_class,
                        epsilon=epsilon,
                        significance_threshold=significance_threshold,
                        diversity_threshold=diversity_threshold,
                        negative_reasoning_path=True
                    )
                except Exception as e:
                    # In case construct_graph_QE_multi raises, log and continue
                    with open(debug_path, 'a') as fh:
                        fh.write(f"construct_graph_QE_multi error for instance {i}: {repr(e)}\n")
                    neighborhood_weakest = []

                # replicate original logic to update weakest counts and TP/TN/FP/FN
                total_weakest = set()
                total_weakest.update(neighborhood_weakest)
                total_weakest = list(total_weakest)

                if len(total_weakest) != 0 and total_weakest != ['consistent']:
                    combined_weakest = total_weakest
                    cnt += 1
                    combi_arg = []
                    for arg in combined_weakest:
                        feature = arg.split('=')[0]
                        if feature in protected_features:
                            combi_arg.append(arg)
                            if arg in weakest_counts:
                                weakest_counts[arg] += 1
                                weakest_indices[arg].append(i)
                            else:
                                weakest_counts[arg] = 1
                                weakest_indices[arg] = [i]

                    # if combination logic applies
                    if (len(combi_arg) > 1) and (dataset_type in [dataset_types.Synthetic_data_global_1,
                                                                  dataset_types.Synthetic_data_global_2,
                                                                  dataset_types.Synthetic_data_global_3]):
                        combi_arg.sort()
                        combi_arg_name = '&'.join(combi_arg)
                        total_weakest.append(combi_arg_name)
                        for arg in combi_arg:
                            weakest_counts[arg] -= 1
                            total_weakest.remove(arg)
                        if combi_arg_name in weakest_counts:
                            weakest_counts[combi_arg_name] += 1
                            weakest_indices[combi_arg_name].append(i)
                        else:
                            weakest_counts[combi_arg_name] = 1
                            weakest_indices[combi_arg_name] = [i]

                # update TP/TN/FP/FN using your original function (if available)
                try:
                    if dataset_name == datasets.SyntheticAdult:
                        TP, TN, FP, FN = calculate_TP_TN_FP_FN(
                            TP, TN, FP, FN, GT_biased, dataset_type, total_weakest,
                            local_synthetic_flag, i, method, protected_features
                        )
                except Exception:
                    # If calculate_TP_TN_FP_FN signature differs, skip
                    pass


                # measure neighborhood properties for logging (original function)
                try:
                    properties, values = measure_neighborhood_properties(i, neighborhoods_dict[max(K_set)]["neighborhoods"], max(K_set), protected_features, total_weakest, method)
                    neighbourhood_properties_data_all.append(values)
                except Exception:
                    # fallback: skip if measure_neighborhood_properties signature differs
                    pass

        # After processing all instances in test_df, compute metrics same as original (only meaningful for SyntheticAdult)
        eps = 1e-10
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        accuracy = (TP + TN) / (TP + TN + FP + FN + eps)
        F1_score = 2 * (((precision * recall)) / (precision + recall + eps))

        # Append aggregated row to comparison_table_data preserving original header order
        comparison_table_data.append([
            method.name,
            dataset_name.name,
            dataset_type.name,
            model_types.Synthetic_model.name,
            epsilon,
            diversity_threshold,
            significance_threshold,
            dist_metric,
            ",".join(map(str, K_set)),
            accuracy,
            precision,
            recall,
            F1_score
        ])

        # Save properties and partial aggregated results as original does
        # Properties table (if we collected)
        try:
            if len(neighbourhood_properties_data_all) > 0:
                # attempt to use same header creation as original
                # neighbourhood_properties_headers variable is created in original as keys of properties
                # but we may not have neighbourhood_properties_headers here; so we create generic columns if needed
                props_df = pd.DataFrame(neighbourhood_properties_data_all)
                props_path = os.path.join(ROOT_DIR, f"results/{dataset_name.name}_props_Kset_{''.join(map(str,K_set))}_dist_{dist_metric}_eps_{epsilon}_div_{diversity_threshold}_sig_{significance_threshold}.csv")
                props_df.to_csv(props_path, index=False)
        except Exception:
            pass

        # Save partial comparison table to disk (so we don't lose progress)
        try:
            comparison_df_partial = pd.DataFrame(comparison_table_data, columns=comparison_table_headers)
            comparison_df_partial.to_csv(comparison_path, index=False)
        except Exception:
            # fallback append
            with open(comparison_path, 'a') as fh:
                fh.write(','.join(map(str, comparison_table_data[-1])) + '\n')

# ==============================================================
# Final save
# ==============================================================
comparison_df = pd.DataFrame(comparison_table_data, columns=comparison_table_headers)
comparison_df.to_csv(comparison_path, index=False)
print(f"\nSensitivity analysis complete. Results saved to: {comparison_path}")



# ==============================================================
# Load results
# ==============================================================
if not os.path.exists(comparison_path):
    raise FileNotFoundError(f"Results file not found: {comparison_path}\nRun the sensitivity script first.")

df = pd.read_csv(comparison_path)
print(f"\nLoaded {len(df)} rows from {comparison_path}\n")


"""
analyze_sensitivity_results_extended.py

Extended sensitivity analysis:
 - Friedman tests across K_set (original)
 - Friedman tests across epsilon, diversity_threshold, significance_threshold
   when fixing K_set to [10,25,50,100,200]
 - Visualization: line plots and heatmaps
"""




# Ensure K_set column is string for filtering
df['K_set_str'] = df['K_set'].astype(str)

print(f"Loaded {len(df)} rows from {comparison_path}")

# ==============================================================
# Friedman test: across K_set values (original)
# ==============================================================
print("\n=== Friedman test: F1 across K_set values ===")
distance_metrics = df['Distance'].unique().tolist()
for dist in distance_metrics:
    subset = df[df['Distance'] == dist]
    pivot = subset.pivot_table(index='Dataset_type', columns='K_set', values='F1-score', aggfunc=np.mean)
    if pivot.shape[1] > 1:
        try:
            data_lists = [pivot[c].dropna().values for c in pivot.columns]
            valid_lists = [l for l in data_lists if len(l) > 0]
            if len(valid_lists) > 1:
                stat, p = friedmanchisquare(*valid_lists)
                print(f"Distance={dist}: Friedman χ²={stat:.3f}, p={p:.3e}")
        except Exception as e:
            print(f"Distance={dist}: Friedman test failed ({e})")

# ==============================================================
# Extended sensitivity: fix K_set and vary other parameters
# ==============================================================
FIXED_K_SET = "25,50,100,200"
df_fixed_K = df[df['K_set_str'] == FIXED_K_SET]

params_to_test = ['Epsilon', 'Diversity_Threshold', 'Significance_Threshold']

print("\n=== Friedman tests across other parameters (fixed K_set) ===")
for param in params_to_test:
    for dist in distance_metrics:
        subset = df_fixed_K[df_fixed_K['Distance'] == dist]
        pivot = subset.pivot_table(index='Dataset_type', columns=param, values='F1-score', aggfunc=np.mean)
        if pivot.shape[1] > 1:
            try:
                data_lists = [pivot[c].dropna().values for c in pivot.columns]
                valid_lists = [l for l in data_lists if len(l) > 0]
                if len(valid_lists) > 1:
                    stat, p = friedmanchisquare(*valid_lists)
                    print(f"Distance={dist}, param={param}: Friedman χ²={stat:.3f}, p={p:.3e}")
            except Exception as e:
                print(f"Distance={dist}, param={param}: Friedman test failed ({e})")

# ==============================================================
# Visualization: Line plots of F1 vs Epsilon
# ==============================================================
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))

for dist in distance_metrics:
    subset = df_fixed_K[df_fixed_K['Distance'] == dist]
    means = subset.groupby('Epsilon')['F1-score'].mean()
    stds = subset.groupby('Epsilon')['F1-score'].std()
    plt.errorbar(means.index, means.values, yerr=stds.values, marker='o', label=f"{dist}")

plt.title(f"Mean F1 vs Epsilon (K_set={FIXED_K_SET})")
plt.xlabel("Epsilon")
plt.ylabel("Mean F1-score ± std")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, "results", "F1_vs_Epsilon.png"))
plt.show()

# ==============================================================
# Heatmap: F1 vs Epsilon and Diversity_Threshold
# ==============================================================
for dist in distance_metrics:
    subset = df_fixed_K[df_fixed_K['Distance'] == dist]
    pivot = subset.pivot_table(index='Diversity_Threshold', columns='Epsilon', values='F1-score', aggfunc=np.mean)
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(f"Mean F1-score Heatmap (Distance={dist})")
    plt.ylabel("Diversity Threshold")
    plt.xlabel("Epsilon")
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT_DIR, f"results/heatmap_F1_{dist}.png"))
    plt.show()

# ==============================================================
# Optional: Coefficient of Variation across parameters
# ==============================================================
print("\n=== Coefficient of Variation (std/mean) for F1 across parameters ===")
for param in params_to_test:
    cv_table = df_fixed_K.groupby(param)['F1-score'].agg(['mean','std'])
    cv_table['CV'] = cv_table['std'] / cv_table['mean']
    print(f"\nParameter: {param}")
    print(cv_table.to_markdown())

# Save final CV table
cv_path = os.path.join(ROOT_DIR, "results", "F1_CV_summary.csv")
df_fixed_K.to_csv(cv_path, index=False)
print(f"\nSaved detailed CV table to: {cv_path}")



# ==============================================================
# 3D Surface Plot: F1 vs Epsilon, Diversity_Threshold, Significance_Threshold
# ==============================================================
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

FIXED_K_SET = "10,25,50,100,200"
df_fixed_K = df[df['K_set_str'] == FIXED_K_SET]

for dist in distance_metrics:
    subset = df_fixed_K[df_fixed_K['Distance'] == dist]
    if subset.empty:
        print(f"No data for distance metric '{dist}', skipping 3D surface plot.")
        continue

    # Prepare grid for 3D surface
    epsilons = sorted(subset['Epsilon'].unique())
    diversities = sorted(subset['Diversity_Threshold'].unique())
    significances = sorted(subset['Significance_Threshold'].unique())

    # Create a meshgrid of epsilon x diversity
    E, D = np.meshgrid(epsilons, diversities)
    # For each (epsilon, diversity) combination, take mean F1 across significance thresholds
    F = np.zeros_like(E, dtype=float)
    for i, d in enumerate(diversities):
        for j, e in enumerate(epsilons):
            vals = subset[
                (subset['Epsilon'] == e) &
                (subset['Diversity_Threshold'] == d)
            ]['F1-score'].values
            if len(vals) > 0:
                F[i, j] = vals.mean()
            else:
                F[i, j] = np.nan  # Handle missing combinations

    # Create 3D surface plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(E, D, F, cmap=cm.viridis, edgecolor='k', linewidth=0.5)
    ax.set_xlabel('Epsilon')
    ax.set_ylabel('Diversity Threshold')
    ax.set_zlabel('Mean F1-score')
    ax.set_title(f'F1-score Surface (Distance={dist}, K_set={FIXED_K_SET})')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()

    # Save figure
    out_path = os.path.join(ROOT_DIR, f"results/F1_surface_3D_{dist}.png")
    plt.savefig(out_path)
    plt.show()
    print(f"Saved 3D surface plot: {out_path}")

# ==============================================================
# One-Parameter-at-a-Time Sensitivity Analysis (averaged across datasets)
# ==============================================================
print("\n=== One-Parameter-at-a-Time Sensitivity Analysis (Averaged over Datasets) ===")

# Determine best-performing combination (highest mean F1 across datasets)
mean_df = df.groupby(['Distance', 'K_set_str', 'Epsilon',
                      'Diversity_Threshold', 'Significance_Threshold'], as_index=False)['F1-score'].mean()
best_idx = mean_df['F1-score'].idxmax()
best_row = mean_df.loc[best_idx]

best_params = {
    'Distance': best_row['Distance'],
    'K_set_str': best_row['K_set_str'],
    'Epsilon': best_row['Epsilon'],
    'Diversity_Threshold': best_row['Diversity_Threshold'],
    'Significance_Threshold': best_row['Significance_Threshold']
}

print(f"Best-performing parameters (averaged over datasets): {best_params}")

# Parameters to analyze
sensitivity_params = ['Distance', 'K_set_str', 'Epsilon', 'Diversity_Threshold', 'Significance_Threshold']

for param in sensitivity_params:
    # Filter dataset: keep all other parameters fixed at best values
    df_filtered = df.copy()
    for p in sensitivity_params:
        if p != param:
            df_filtered = df_filtered[df_filtered[p] == best_params[p]]

    if df_filtered.empty:
        print(f"No data available to analyze sensitivity to {param}")
        continue

    # Aggregate mean and std over datasets
    df_agg = (
        df_filtered.groupby(param)['F1-score']
        .agg(['mean', 'std'])
        .reset_index()
        .sort_values(param)
    )

    print(f"\nSensitivity of F1 to {param} (others fixed at best, averaged over datasets):")
    print(df_agg.to_markdown(index=False))

    # --- Plot ---
    plt.figure(figsize=(8,5))
    plt.plot(df_agg[param], df_agg['mean'], marker='o', color='teal', label='Mean F1')
    lower = np.clip(df_agg['mean'] - df_agg['std'], 0, 1)
    upper = np.clip(df_agg['mean'] + df_agg['std'], 0, 1)

    plt.fill_between(df_agg[param], lower, upper,
                     color='teal', alpha=0.2, label='±1 std')
    plt.title(f"Sensitivity of F1 to {param}\n(others fixed at best-performing values)")
    plt.xlabel(param)
    plt.ylabel("F1-score (mean ± std across datasets)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(ROOT_DIR, f"results/F1_sensitivity_avg_{param}.png")
    plt.savefig(fig_path)
    plt.show()
    print(f"Saved averaged sensitivity plot: {fig_path}")



parameters = ["K_set_str", "Distance", "Epsilon", "Diversity_Threshold", "Significance_Threshold"]
sensitivity_summary = []

for param in parameters:

    print(f"\n===== Checking parameter: {param} =====")

    # Dataset-level mean F1 for each parameter value
    block_df = df.groupby(["Dataset_type", param], as_index=False)["F1-score"].mean()

    # Count parameter value occurrences per dataset
    counts = block_df.groupby(param)["Dataset_type"].nunique()
    max_datasets = df["Dataset_type"].nunique()

    # Keep only parameter values present in ALL datasets
    common_values = counts[counts == max_datasets].index.tolist()

    print("Original levels:", sorted(block_df[param].unique()))
    print("Common levels:", common_values)

    # If fewer than 3 levels remain, Friedman cannot run
    if len(common_values) < 3:
        print("⚠ Not enough common levels for Friedman test.")
        stat, p = np.nan, np.nan
    else:
        filtered = block_df[block_df[param].isin(common_values)]

        # Pivot: rows=datasets, columns=parameter values
        pivot = filtered.pivot(index="Dataset_type", columns=param, values="F1-score")

        print("Pivot shape:", pivot.shape)

        # Run Friedman
        stat, p = friedmanchisquare(*[pivot[col].values for col in pivot.columns])

    # Effect size (how much the mean F1 varies across parameter values)
    grouped_means = block_df.groupby(param)["F1-score"].mean()
    mean_f1 = grouped_means.mean()
    std_f1 = grouped_means.std()
    cv = std_f1 / mean_f1 if mean_f1 != 0 else np.nan

    sensitivity_summary.append({
        "Parameter": param,
        "Chi2": stat,
        "p-value": p,
        "Mean F1": mean_f1,
        "Std F1": std_f1,
        "CV (std/mean)": cv,
        "Significant (p<0.05)": p < 0.05 if not np.isnan(p) else False,
        "Common Levels": len(common_values)
    })

sensitivity_df = pd.DataFrame(sensitivity_summary)
print("\n=== Sensitivity Summary (Corrected Friedman) ===")
print(sensitivity_df.to_markdown(index=False))


# ==========================================================
# Visualization: Sensitivity Strength (Averaged Analysis)
# ==========================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot χ² values (sensitivity strength)
sns.barplot(data=sensitivity_df, x="Parameter", y="Chi2", ax=ax[0], palette="viridis")
ax[0].set_title("Sensitivity Strength (Friedman χ², Averaged Across Datasets)")
ax[0].set_ylabel("χ² Statistic")
ax[0].set_xlabel("Parameter")
ax[0].tick_params(axis='x', rotation=45)

# Plot p-values
sns.barplot(data=sensitivity_df, x="Parameter", y="p-value", ax=ax[1], palette="magma")
ax[1].axhline(0.05, color="red", linestyle="--", label="p=0.05")
ax[1].set_title("Friedman Test p-values (Averaged Across Datasets)")
ax[1].set_ylabel("p-value")
ax[1].set_xlabel("Parameter")
ax[1].legend()
ax[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
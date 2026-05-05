#!/usr/bin/env python3
"""
MNIST 3-vs-6 classification with a classical RBF kernel SVM baseline.

This script is designed to match the quantum Chebyshev kernel experiment:
- same MNIST 3-vs-6 task
- same balanced train/test subsampling
- same PCA dimension, controlled by --n-qubits
- same SVM with precomputed kernel
- CSV logging
- optional NPZ saving of kernel matrices

Example usage
-------------
python mnist_rbf_kernel_baseline.py \
    --n-qubits 8 \
    --n-train 200 \
    --n-test 100 \
    --seed 42 \
    --rbf-gamma scale \
    --svm-c 1.0

With explicit gamma:

python mnist_rbf_kernel_baseline.py \
    --n-qubits 8 \
    --n-train 200 \
    --n-test 100 \
    --seed 42 \
    --rbf-gamma 0.5 \
    --svm-c 1.0
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


@dataclass
class DataBundle:
    X_train_pca_scaled: np.ndarray
    X_test_pca_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    pca: PCA
    pixel_scaler: StandardScaler
    feature_scaler: MinMaxScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST 3-vs-6 classification with classical RBF kernel SVM."
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=8,
        help="PCA dimension. Named n-qubits to match the quantum experiment.",
    )
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)

    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument(
        "--rbf-gamma",
        type=str,
        default="scale",
        help="Gamma for RBF kernel. Use 'scale', 'auto', or a float string, e.g. '0.5'.",
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/mnist_rbf",
    )
    parser.add_argument("--save-kernels", action="store_true")
    parser.add_argument("--plot-kernel", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)

    return parser.parse_args()


def parse_rbf_gamma(gamma_arg: str, X_train: np.ndarray):
    if gamma_arg == "scale":
        return 1.0 / (X_train.shape[1] * X_train.var())
    if gamma_arg == "auto":
        return 1.0 / X_train.shape[1]
    return float(gamma_arg)


def make_run_name(args: argparse.Namespace) -> str:
    gamma_str = str(args.rbf_gamma).replace(".", "p")
    c_str = str(args.svm_c).replace(".", "p")

    return (
        f"kernel=classical_rbf"
        f"_pca={args.n_qubits}"
        f"_gamma={gamma_str}"
        f"_C={c_str}"
        f"_seed={args.seed}"
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def balanced_subset_indices(
    y_arr: np.ndarray,
    n_samples_total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_samples_total % 2 != 0:
        raise ValueError("Requested balanced subset size must be even.")

    n_per_class = n_samples_total // 2

    idx0 = np.where(y_arr == 0)[0]
    idx1 = np.where(y_arr == 1)[0]

    if len(idx0) < n_per_class or len(idx1) < n_per_class:
        raise ValueError(
            f"Requested {n_per_class} samples per class, but found "
            f"{len(idx0)} class-0 and {len(idx1)} class-1 samples."
        )

    idx0 = rng.choice(idx0, size=n_per_class, replace=False)
    idx1 = rng.choice(idx1, size=n_per_class, replace=False)

    idx = np.concatenate([idx0, idx1])
    rng.shuffle(idx)

    return idx


def load_preprocess_mnist_36(
    n_components: int,
    n_train: int,
    n_test: int,
    seed: int,
    test_size: float,
) -> DataBundle:
    rng = np.random.default_rng(seed)

    print("Downloading/loading MNIST from OpenML...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data.astype(np.float64)
    y = mnist.target

    # Keep only digits 3 and 6.
    mask = np.isin(y, ["3", "6"])
    X = X[mask]
    y = y[mask]

    # Encode 3 -> 0, 6 -> 1.
    y = np.where(y == "3", 0, 1).astype(int)

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    # Standardize pixels using train split only.
    pixel_scaler = StandardScaler()
    X_train_full = pixel_scaler.fit_transform(X_train_full)
    X_test_full = pixel_scaler.transform(X_test_full)

    # PCA to match the quantum input dimension.
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_full)
    X_test_pca = pca.transform(X_test_full)

    # Scale PCA features to [-1, 1], same as quantum experiment.
    # For the RBF baseline this is not strictly necessary, but it makes
    # the comparison against the quantum input preprocessing fairer.
    feature_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_train_pca_scaled_full = feature_scaler.fit_transform(X_train_pca)
    X_test_pca_scaled_full = feature_scaler.transform(X_test_pca)

    train_idx = balanced_subset_indices(y_train_full, n_train, rng)
    test_idx = balanced_subset_indices(y_test_full, n_test, rng)

    return DataBundle(
        X_train_pca_scaled=X_train_pca_scaled_full[train_idx],
        X_test_pca_scaled=X_test_pca_scaled_full[test_idx],
        y_train=y_train_full[train_idx],
        y_test=y_test_full[test_idx],
        pca=pca,
        pixel_scaler=pixel_scaler,
        feature_scaler=feature_scaler,
    )


def evaluate_rbf_kernel(
    X_train: np.ndarray,
    X_test: np.ndarray,
    gamma,
) -> tuple[np.ndarray, np.ndarray, float]:
    t0 = time.perf_counter()

    K_train = rbf_kernel(X_train, X_train, gamma=gamma)
    K_test = rbf_kernel(X_test, X_train, gamma=gamma)

    kernel_eval_time = time.perf_counter() - t0

    return K_train, K_test, kernel_eval_time


def plot_sorted_kernel_matrix(
    K_train: np.ndarray,
    y_train: np.ndarray,
    title: str,
) -> None:
    order = np.argsort(y_train)

    K_sorted = K_train[order][:, order]
    y_sorted = y_train[order]

    plt.figure(figsize=(5, 5))
    plt.imshow(K_sorted, interpolation="nearest", aspect="auto")
    plt.colorbar(label="Kernel value")
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Sample index")

    boundary = np.sum(y_sorted == 0) - 0.5
    plt.axhline(boundary, linewidth=1)
    plt.axvline(boundary, linewidth=1)

    plt.tight_layout()
    plt.show()


def append_results_row(csv_path: str, row: Dict[str, object]) -> None:
    file_exists = os.path.exists(csv_path)
    fieldnames = list(row.keys())

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_kernel_npz(
    npz_path: str,
    K_train: np.ndarray,
    K_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    run_metadata: Dict[str, object],
) -> None:
    np.savez_compressed(
        npz_path,
        K_train=K_train,
        K_test=K_test,
        y_train=y_train,
        y_test=y_test,
        metadata=np.array([run_metadata], dtype=object),
    )


def main() -> None:
    args = parse_args()

    ensure_dir(args.results_dir)

    kernels_dir = os.path.join(args.results_dir, "kernels")
    ensure_dir(kernels_dir)

    run_name = args.run_name if args.run_name is not None else make_run_name(args)

    csv_path = os.path.join(args.results_dir, "results.csv")
    npz_path = os.path.join(kernels_dir, f"{run_name}.npz")

    print("Loading and preprocessing MNIST 3 vs 6...")
    data = load_preprocess_mnist_36(
        n_components=args.n_qubits,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        test_size=args.test_size,
    )

    gamma = parse_rbf_gamma(args.rbf_gamma, data.X_train_pca_scaled)

    print("=" * 60)
    print(f"Run name:        {run_name}")
    print(f"Kernel type:     classical_rbf")
    print(f"PCA dim:         {args.n_qubits}")
    print(f"n_train:         {args.n_train}")
    print(f"n_test:          {args.n_test}")
    print(f"SVM C:           {args.svm_c}")
    print(f"RBF gamma:       {gamma}")
    print(f"results_dir:     {args.results_dir}")
    print("=" * 60)
    
    print(f"Train subset size: {data.X_train_pca_scaled.shape[0]}")
    print(f"Test subset size:  {data.X_test_pca_scaled.shape[0]}")
    print(f"Input dimension:   {data.X_train_pca_scaled.shape[1]}")

    explained_var = np.sum(data.pca.explained_variance_ratio_)
    print(f"PCA explained variance ratio: {explained_var:.4f}")

    print("Evaluating classical RBF kernel matrices...")
    K_train, K_test, kernel_eval_time = evaluate_rbf_kernel(
        X_train=data.X_train_pca_scaled,
        X_test=data.X_test_pca_scaled,
        gamma=gamma,
    )

    print(f"K_train shape:        {K_train.shape}")
    print(f"K_test shape:         {K_test.shape}")
    print(f"Kernel eval time:     {kernel_eval_time:.6f} s")

    if args.plot_kernel:
        plot_sorted_kernel_matrix(
            K_train=K_train,
            y_train=data.y_train,
            title=f"Classical RBF kernel, PCA dim={args.n_qubits}, gamma={args.rbf_gamma}",
        )

    print("Training SVM with precomputed RBF kernel...")
    svm = SVC(kernel="precomputed", C=args.svm_c)
    svm.fit(K_train, data.y_train)

    y_pred = svm.predict(K_test)
    acc = accuracy_score(data.y_test, y_pred)

    print(f"Test accuracy:        {acc:.4f}")

    results_row = {
        "run_name": run_name,
        "kernel_type": "classical_rbf",
        "seed": args.seed,
        "n_qubits": args.n_qubits,
        "n_layers": "",
        "entanglement": "",
        "n_train": args.n_train,
        "n_test": args.n_test,
        "svm_c": args.svm_c,
        "rbf_gamma": args.rbf_gamma,
        "train_kernel": False,
        "initial_theta": "",
        "maxiter": "",
        "learning_rate": "",
        "perturbation": "",
        "n_trainable_params": 0,
        "train_time_sec": 0.0,
        "kernel_eval_time_sec": kernel_eval_time,
        "test_accuracy": acc,
        "optimal_value": np.nan,
        "optimizer_evals": np.nan,
        "pca_explained_variance_ratio": explained_var,
    }

    append_results_row(csv_path, results_row)
    print(f"Appended results to:  {csv_path}")

    if args.save_kernels:
        metadata = {
            "run_name": run_name,
            "kernel_type": "classical_rbf",
            "seed": args.seed,
            "n_qubits": args.n_qubits,
            "n_train": args.n_train,
            "n_test": args.n_test,
            "svm_c": args.svm_c,
            "rbf_gamma": args.rbf_gamma,
            "test_accuracy": acc,
            "pca_explained_variance_ratio": explained_var,
        }

        save_kernel_npz(
            npz_path=npz_path,
            K_train=K_train,
            K_test=K_test,
            y_train=data.y_train,
            y_test=data.y_test,
            run_metadata=metadata,
        )

        print(f"Saved kernel matrices: {npz_path}")

    print("Done.")


if __name__ == "__main__":
    main()
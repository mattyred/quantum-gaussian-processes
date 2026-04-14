#!/usr/bin/env python3
"""
MNIST 3-vs-6 classification with:
- Chebyshev-inspired quantum kernel
- Classical RBF kernel baseline
- Optional training of quantum kernel parameters
- CSV logging
- NPZ saving of kernel matrices for later plotting

Example usage
-------------
# Quantum kernel, fixed random theta
python mnist_chebyshev_qkernel.py \
    --kernel-type quantum_chebyshev \
    --n-qubits 8 \
    --n-layers 1 \
    --entanglement linear \
    --n-train 200 \
    --n-test 100 \
    --seed 42

# Quantum kernel, optimized theta
python mnist_chebyshev_qkernel.py \
    --kernel-type quantum_chebyshev \
    --n-qubits 8 \
    --n-layers 2 \
    --entanglement linear \
    --n-train 200 \
    --n-test 100 \
    --seed 42 \
    --train-kernel \
    --maxiter 20 \
    --learning-rate 0.05 \
    --perturbation 0.05

# Classical RBF baseline
python mnist_chebyshev_qkernel.py \
    --kernel-type classical_rbf \
    --n-qubits 8 \
    --n-layers 1 \
    --n-train 200 \
    --n-test 100 \
    --seed 42
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel,
    TrainableFidelityQuantumKernel,
)
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.state_fidelities import ComputeUncompute


@dataclass
class DataBundle:
    X_train_alpha: np.ndarray
    X_test_alpha: np.ndarray
    X_train_pca_scaled: np.ndarray
    X_test_pca_scaled: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    pca: PCA
    pixel_scaler: StandardScaler
    feature_scaler: MinMaxScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST 3-vs-6 classification with quantum or classical kernels."
    )

    # Experiment settings
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--kernel-type",
        type=str,
        default="quantum_chebyshev",
        choices=["quantum_chebyshev", "classical_rbf"],
        help="Kernel type to use.",
    )
    parser.add_argument("--n-qubits", type=int, default=8, help="Number of qubits / PCA components.")
    parser.add_argument("--n-layers", type=int, default=1, help="Number of layers in the quantum feature map.")
    parser.add_argument(
        "--entanglement",
        type=str,
        default="linear",
        choices=["linear", "ring"],
        help="Entanglement pattern in the quantum feature map.",
    )
    parser.add_argument("--n-train", type=int, default=200, help="Balanced train subset size.")
    parser.add_argument("--n-test", type=int, default=100, help="Balanced test subset size.")
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of filtered MNIST used for the initial train/test split.",
    )

    # SVM
    parser.add_argument("--svm-c", type=float, default=1.0, help="C parameter for the SVM.")

    # Classical RBF
    parser.add_argument(
        "--rbf-gamma",
        type=str,
        default="scale",
        help="Gamma for RBF kernel. Use 'scale', 'auto', or a float string like '0.5'.",
    )

    # Quantum kernel training
    parser.add_argument(
        "--train-kernel",
        action="store_true",
        help="Train the quantum kernel parameters. Ignored for classical_rbf.",
    )
    parser.add_argument("--maxiter", type=int, default=20, help="Max optimizer iterations.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="SPSA learning rate.")
    parser.add_argument("--perturbation", type=float, default=0.05, help="SPSA perturbation.")
    parser.add_argument(
        "--initial-theta",
        type=str,
        default="random",
        choices=["random", "zeros", "pi_over_2"],
        help="Initialization strategy for trainable quantum parameters.",
    )
    parser.add_argument(
        "--no-enforce-psd",
        action="store_true",
        help="Disable PSD projection in the quantum kernel.",
    )

    # Saving / plotting
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/mnist_chebyshev",
        help="Directory where CSV and NPZ files are saved.",
    )
    parser.add_argument(
        "--save-kernels",
        action="store_true",
        help="Save K_train/K_test and labels as NPZ.",
    )
    parser.add_argument(
        "--plot-kernel",
        action="store_true",
        help="Display the sorted training kernel matrix.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional custom run name. If omitted, one is constructed automatically.",
    )

    return parser.parse_args()


def parse_rbf_gamma(gamma_arg: str):
    if gamma_arg in {"scale", "auto"}:
        return gamma_arg
    return float(gamma_arg)


def make_run_name(args: argparse.Namespace) -> str:
    train_flag = "true" if args.train_kernel and args.kernel_type == "quantum_chebyshev" else "false"
    return (
        f"kernel={args.kernel_type}"
        f"_layers={args.n_layers}"
        f"_traink={train_flag}"
        f"_qubits={args.n_qubits}"
        f"_seed={args.seed}"
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def chebyshev_feature_map_vector(
    n_qubits: int = 4,
    n_layers: int = 1,
    entanglement: str = "ring",
    name: str = "ChebyshevInspiredFM",
) -> Tuple[QuantumCircuit, ParameterVector, List[ParameterVector]]:
    """
    Generalized Chebyshev-inspired feature map.

    Each qubit i receives:
        Ry(theta_i) -> Rx(theta_i * alpha_i) -> entanglers -> Ry(theta_i)

    where alpha_i = arccos(x_i) is precomputed classically.
    """
    alpha = ParameterVector("alpha", n_qubits)
    qc = QuantumCircuit(n_qubits, name=name)

    theta_blocks: List[ParameterVector] = []

    if entanglement == "ring":
        edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    elif entanglement == "linear":
        edges = [(i, i + 1) for i in range(n_qubits - 1)]
    else:
        raise ValueError("entanglement must be 'ring' or 'linear'")

    n_ent = len(edges)

    for layer in range(n_layers):
        theta = ParameterVector(f"theta_{layer}", n_qubits + n_ent)
        theta_blocks.append(theta)

        for i in range(n_qubits):
            qc.ry(theta[i], i)

        for i in range(n_qubits):
            qc.rx(theta[i] * alpha[i], i)

        for j, (control, target) in enumerate(edges):
            qc.crz(theta[n_qubits + j], control, target)

        for i in range(n_qubits):
            qc.ry(theta[i], i)

    return qc, alpha, theta_blocks


def make_initial_point(
    n_params: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if mode == "random":
        return rng.uniform(0.0, 2.0 * np.pi, size=n_params)
    if mode == "zeros":
        return np.zeros(n_params, dtype=float)
    if mode == "pi_over_2":
        return np.full(n_params, np.pi / 2.0, dtype=float)
    raise ValueError(f"Unknown initial-theta mode: {mode}")


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
            f"Requested {n_per_class} samples per class, but only found "
            f"{len(idx0)} class-0 and {len(idx1)} class-1 samples."
        )

    idx0 = rng.choice(idx0, size=n_per_class, replace=False)
    idx1 = rng.choice(idx1, size=n_per_class, replace=False)
    idx = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    return idx


def load_preprocess_mnist_36(
    n_qubits: int,
    n_train: int,
    n_test: int,
    seed: int,
    test_size: float,
) -> DataBundle:
    rng = np.random.default_rng(seed)

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float64)
    y = mnist.target

    mask = np.isin(y, ["3", "6"])
    X = X[mask]
    y = y[mask]
    y = np.where(y == "3", 0, 1).astype(int)

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    pixel_scaler = StandardScaler()
    X_train_full = pixel_scaler.fit_transform(X_train_full)
    X_test_full = pixel_scaler.transform(X_test_full)

    pca = PCA(n_components=n_qubits, random_state=seed)
    X_train_pca = pca.fit_transform(X_train_full)
    X_test_pca = pca.transform(X_test_full)

    feature_scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
    X_train_pca_scaled_full = feature_scaler.fit_transform(X_train_pca)
    X_test_pca_scaled_full = feature_scaler.transform(X_test_pca)

    X_train_alpha_full = np.arccos(np.clip(X_train_pca_scaled_full, -1.0, 1.0))
    X_test_alpha_full = np.arccos(np.clip(X_test_pca_scaled_full, -1.0, 1.0))

    train_idx = balanced_subset_indices(y_train_full, n_train, rng)
    test_idx = balanced_subset_indices(y_test_full, n_test, rng)

    return DataBundle(
        X_train_alpha=X_train_alpha_full[train_idx],
        X_test_alpha=X_test_alpha_full[test_idx],
        X_train_pca_scaled=X_train_pca_scaled_full[train_idx],
        X_test_pca_scaled=X_test_pca_scaled_full[test_idx],
        y_train=y_train_full[train_idx],
        y_test=y_test_full[test_idx],
        pca=pca,
        pixel_scaler=pixel_scaler,
        feature_scaler=feature_scaler,
    )


def collect_training_parameters(theta_blocks: Sequence[ParameterVector]) -> List[Parameter]:
    params: List[Parameter] = []
    for block in theta_blocks:
        params.extend(list(block))
    return params


def build_fixed_quantum_kernel(
    feature_map: QuantumCircuit,
    training_parameters: Sequence[Parameter],
    initial_point: np.ndarray,
    enforce_psd: bool,
) -> Tuple[FidelityQuantumKernel, Dict[Parameter, float]]:
    theta_bind = {param: float(val) for param, val in zip(training_parameters, initial_point)}
    feature_map_fixed = feature_map.assign_parameters(theta_bind)

    kernel = FidelityQuantumKernel(
        feature_map=feature_map_fixed,
        enforce_psd=enforce_psd,
    )
    return kernel, theta_bind


def train_quantum_kernel(
    feature_map: QuantumCircuit,
    training_parameters: Sequence[Parameter],
    X_train: np.ndarray,
    y_train: np.ndarray,
    initial_point: np.ndarray,
    maxiter: int,
    learning_rate: float,
    perturbation: float,
    enforce_psd: bool,
) -> Tuple[TrainableFidelityQuantumKernel, object]:
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)

    trainable_kernel = TrainableFidelityQuantumKernel(
        feature_map=feature_map,
        fidelity=fidelity,
        training_parameters=training_parameters,
        enforce_psd=enforce_psd,
    )

    optimizer = SPSA(
        maxiter=maxiter,
        learning_rate=learning_rate,
        perturbation=perturbation,
    )

    trainer = QuantumKernelTrainer(
        quantum_kernel=trainable_kernel,
        loss="svc_loss",
        optimizer=optimizer,
        initial_point=initial_point,
    )

    result = trainer.fit(X_train, y_train)
    return result.quantum_kernel, result


def plot_sorted_kernel_matrix(K_train: np.ndarray, y_train: np.ndarray, title: str) -> None:
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
    rng = np.random.default_rng(args.seed)

    ensure_dir(args.results_dir)
    kernels_dir = os.path.join(args.results_dir, "kernels")
    ensure_dir(kernels_dir)

    run_name = args.run_name if args.run_name is not None else make_run_name(args)
    csv_path = os.path.join(args.results_dir, "results.csv")
    npz_path = os.path.join(kernels_dir, f"{run_name}.npz")

    print("=" * 60)
    print(f"Run name:        {run_name}")
    print(f"Kernel type:     {args.kernel_type}")
    print(f"n_qubits:        {args.n_qubits}")
    print(f"n_layers:        {args.n_layers}")
    print(f"train_kernel:    {args.train_kernel}")
    print(f"results_dir:     {args.results_dir}")
    print("=" * 60)

    print("Loading and preprocessing MNIST 3 vs 6...")
    data = load_preprocess_mnist_36(
        n_qubits=args.n_qubits,
        n_train=args.n_train,
        n_test=args.n_test,
        seed=args.seed,
        test_size=args.test_size,
    )

    print(f"Train subset size: {data.X_train_alpha.shape[0]}")
    print(f"Test subset size:  {data.X_test_alpha.shape[0]}")

    train_time = 0.0
    kernel_eval_time = 0.0
    n_trainable_params = 0
    optimal_value = np.nan
    optimizer_evals = np.nan

    if args.kernel_type == "quantum_chebyshev":
        print("Building quantum feature map...")
        feature_map, alpha_params, theta_blocks = chebyshev_feature_map_vector(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            entanglement=args.entanglement,
        )

        training_parameters = collect_training_parameters(theta_blocks)
        n_trainable_params = len(training_parameters)
        initial_point = make_initial_point(
            n_params=n_trainable_params,
            mode=args.initial_theta,
            rng=rng,
        )

        print(f"Number of trainable quantum parameters: {n_trainable_params}")

        if args.train_kernel:
            print("Training quantum kernel parameters...")
            t0 = time.perf_counter()
            quantum_kernel, training_result = train_quantum_kernel(
                feature_map=feature_map,
                training_parameters=training_parameters,
                X_train=data.X_train_alpha,
                y_train=data.y_train,
                initial_point=initial_point,
                maxiter=args.maxiter,
                learning_rate=args.learning_rate,
                perturbation=args.perturbation,
                enforce_psd=not args.no_enforce_psd,
            )
            train_time = time.perf_counter() - t0

            optimal_value = float(training_result.optimal_value)
            optimizer_evals = int(training_result.optimizer_evals)

            print(f"Kernel training time: {train_time:.3f} s")
            print(f"Optimal objective:    {optimal_value}")
            print(f"Optimizer evals:      {optimizer_evals}")
        else:
            print("Using fixed random quantum kernel parameters...")
            quantum_kernel, theta_bind = build_fixed_quantum_kernel(
                feature_map=feature_map,
                training_parameters=training_parameters,
                initial_point=initial_point,
                enforce_psd=not args.no_enforce_psd,
            )

        print("Evaluating quantum kernel matrices...")
        t0 = time.perf_counter()
        K_train = quantum_kernel.evaluate(data.X_train_alpha)
        K_test = quantum_kernel.evaluate(data.X_test_alpha, data.X_train_alpha)
        kernel_eval_time = time.perf_counter() - t0

    elif args.kernel_type == "classical_rbf":
        gamma = parse_rbf_gamma(args.rbf_gamma)
        print(f"Evaluating classical RBF kernel matrices with gamma={gamma}...")

        t0 = time.perf_counter()
        K_train = rbf_kernel(data.X_train_pca_scaled, data.X_train_pca_scaled, gamma=gamma)
        K_test = rbf_kernel(data.X_test_pca_scaled, data.X_train_pca_scaled, gamma=gamma)
        kernel_eval_time = time.perf_counter() - t0

    else:
        raise ValueError(f"Unsupported kernel type: {args.kernel_type}")

    print(f"K_train shape:        {K_train.shape}")
    print(f"K_test shape:         {K_test.shape}")
    print(f"Kernel eval time:     {kernel_eval_time:.3f} s")

    if args.plot_kernel:
        plot_sorted_kernel_matrix(K_train, data.y_train, title=f"{args.kernel_type} K_train")

    print("Training SVM with precomputed kernel...")
    svm = SVC(kernel="precomputed", C=args.svm_c)
    svm.fit(K_train, data.y_train)
    y_pred = svm.predict(K_test)
    acc = accuracy_score(data.y_test, y_pred)

    print(f"Test accuracy:        {acc:.4f}")

    results_row = {
        "run_name": run_name,
        "kernel_type": args.kernel_type,
        "seed": args.seed,
        "n_qubits": args.n_qubits,
        "n_layers": args.n_layers,
        "entanglement": args.entanglement,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "svm_c": args.svm_c,
        "rbf_gamma": args.rbf_gamma if args.kernel_type == "classical_rbf" else "",
        "train_kernel": bool(args.train_kernel and args.kernel_type == "quantum_chebyshev"),
        "initial_theta": args.initial_theta if args.kernel_type == "quantum_chebyshev" else "",
        "maxiter": args.maxiter if args.kernel_type == "quantum_chebyshev" else "",
        "learning_rate": args.learning_rate if args.kernel_type == "quantum_chebyshev" else "",
        "perturbation": args.perturbation if args.kernel_type == "quantum_chebyshev" else "",
        "n_trainable_params": n_trainable_params,
        "train_time_sec": train_time,
        "kernel_eval_time_sec": kernel_eval_time,
        "test_accuracy": acc,
        "optimal_value": optimal_value,
        "optimizer_evals": optimizer_evals,
    }

    append_results_row(csv_path, results_row)
    print(f"Appended results to:  {csv_path}")

    if args.save_kernels:
        metadata = {
            "run_name": run_name,
            "kernel_type": args.kernel_type,
            "seed": args.seed,
            "n_qubits": args.n_qubits,
            "n_layers": args.n_layers,
            "train_kernel": bool(args.train_kernel and args.kernel_type == "quantum_chebyshev"),
            "test_accuracy": acc,
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
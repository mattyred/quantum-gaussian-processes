#!/bin/bash -l 

#SBATCH --job-name=mnist_qrand
#SBATCH --output=results/mnist_chebyshev_random_sweep/logs/%x_%A_%a.out 
#SBATCH --error=results/mnist_chebyshev_random_sweep/logs/%x_%A_%a.err 
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=32G 
#SBATCH --time=08:00:00
#SBATCH --partition=batch 
#SBATCH --array=1-250

source ~/.bashrc
conda activate cudaqenv

RESULTS_DIR="results/mnist_chebyshev_random_sweep"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}/logs"

SCRIPT_PATH="mnist_chebyshev.py"

# ------------------------------------------------------------
# Experiment grid
# ------------------------------------------------------------
n_qubits_values=(2 4 6 8 16)
n_layers_values=(1 2)
svm_c_values=(0.01 0.1 1.0 10.0 100.0)
seed_values=(42 43 44 45 46)

# Total runs = 5 qubit values x 2 layers x 5 C values x 5 seeds = 250

# ------------------------------------------------------------
# Decode SLURM_ARRAY_TASK_ID
# ------------------------------------------------------------
task_id=${SLURM_ARRAY_TASK_ID:-1}
idx=$(( task_id - 1 ))

n_qubits_count=${#n_qubits_values[@]}
n_layers_count=${#n_layers_values[@]}
svm_c_count=${#svm_c_values[@]}
seed_count=${#seed_values[@]}

qubit_idx=$(( idx % n_qubits_count ))
idx=$(( idx / n_qubits_count ))

layer_idx=$(( idx % n_layers_count ))
idx=$(( idx / n_layers_count ))

c_idx=$(( idx % svm_c_count ))
idx=$(( idx / svm_c_count ))

seed_idx=$(( idx % seed_count ))

n_qubits=${n_qubits_values[$qubit_idx]}
n_layers=${n_layers_values[$layer_idx]}
svm_c=${svm_c_values[$c_idx]}
seed=${seed_values[$seed_idx]}

# ------------------------------------------------------------
# Fixed settings
# ------------------------------------------------------------
kernel_type="quantum_chebyshev"
train_kernel="false"
n_train=200
n_test=100
entanglement="linear"
initial_theta="random"

# Make C safe for file names
svm_c_tag=${svm_c//./p}

run_name="kernel=${kernel_type}_layers=${n_layers}_traink=${train_kernel}_qubits=${n_qubits}_C=${svm_c_tag}_seed=${seed}"

echo "=================================================="
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "Run name            = ${run_name}"
echo "kernel_type         = ${kernel_type}"
echo "n_layers            = ${n_layers}"
echo "train_kernel        = ${train_kernel}"
echo "n_qubits            = ${n_qubits}"
echo "svm_c               = ${svm_c}"
echo "seed                = ${seed}"
echo "results_dir         = ${RESULTS_DIR}"
echo "=================================================="

CMD=(
    python "${SCRIPT_PATH}"
    --kernel-type "${kernel_type}"
    --seed "${seed}"
    --n-qubits "${n_qubits}"
    --n-layers "${n_layers}"
    --entanglement "${entanglement}"
    --n-train "${n_train}"
    --n-test "${n_test}"
    --svm-c "${svm_c}"
    --initial-theta "${initial_theta}"
    --results-dir "${RESULTS_DIR}"
    --save-kernels
    --run-name "${run_name}"
)

echo "Executing:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}" | tee "${RESULTS_DIR}/logs/${run_name}.log"
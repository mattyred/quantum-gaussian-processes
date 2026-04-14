#!/bin/bash -l 

#SBATCH --job-name=mnist_chebyshev
#SBATCH --output=results/mnist_chebyshev/logs/%x_%A_%a.out 
#SBATCH --error=results/mnist_chebyshev/logs/%x_%A_%a.err 
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --mem=32G 
#SBATCH --time=03:00:00
#SBATCH --partition=batch 
#SBATCH --array=1-20

source ~/.bashrc
conda activate cudaqenv

mkdir -p results/mnist_chebyshev
mkdir -p results/mnist_chebyshev/logs

SCRIPT_PATH="mnist_chebyshev.py"

# ------------------------------------------------------------
# Experiment grid
# ------------------------------------------------------------
n_qubits_values=(2 4 6 8 16)
n_layers_values=(1 2)
train_kernel_values=(false true)

# ------------------------------------------------------------
# Decode SLURM_ARRAY_TASK_ID
# qidx = layer * (2*5) + train_kernel * 5 + qubit_idx
# Total runs = 2 x 2 x 5 = 20
# ------------------------------------------------------------
task_id=${SLURM_ARRAY_TASK_ID:-1}

seed=42
n_train=200
n_test=100
svm_c=1.0
entanglement="linear"
maxiter=20
learning_rate=0.05
perturbation=0.05
initial_theta="random"

layer_idx=$(( task_id / 10 ))
rem=$(( task_id % 10 ))
train_idx=$(( rem / 5 ))
qubit_idx=$(( rem % 5 ))

n_layers=${n_layers_values[$layer_idx]}
n_qubits=${n_qubits_values[$qubit_idx]}
train_kernel=${train_kernel_values[$train_idx]}
kernel_type="quantum_chebyshev"

run_name="kernel=${kernel_type}_layers=${n_layers}_traink=${train_kernel}_qubits=${n_qubits}_seed=${seed}"

echo "=================================================="
echo "SLURM_ARRAY_TASK_ID = ${SLURM_ARRAY_TASK_ID}"
echo "Run name            = ${run_name}"
echo "kernel_type         = ${kernel_type}"
echo "n_layers            = ${n_layers}"
echo "train_kernel        = ${train_kernel}"
echo "n_qubits            = ${n_qubits}"
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
    --save-kernels
    --run-name "${run_name}"
)

if [ "${train_kernel}" = "true" ]; then
    CMD+=(
        --train-kernel
        --maxiter "${maxiter}"
        --learning-rate "${learning_rate}"
        --perturbation "${perturbation}"
    )
fi

echo "Executing:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}" | tee "results/mnist_chebyshev/logs/${run_name}.log"
#!/bin/bash -l

#SBATCH --job-name=mnist_chebyshev_missing
#SBATCH --output=results/mnist_chebyshev/logs/%x_%A_%a.out
#SBATCH --error=results/mnist_chebyshev/logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=batch
#SBATCH --array=1-2

source ~/.bashrc
conda activate cudaqenv

SCRIPT_PATH="mnist_chebyshev.py"

seed=42
n_train=200
n_test=100
svm_c=1.0
entanglement="linear"
maxiter=20
learning_rate=0.05
perturbation=0.05
initial_theta="random"
kernel_type="quantum_chebyshev"

task_id=${SLURM_ARRAY_TASK_ID:-1}

# ------------------------------------------------------------
# Only rerun the 2 missing configurations
# 1 -> random kernel, 1 layer, 2 qubits
# 2 -> optimized kernel, 2 layers, 8 qubits
# ------------------------------------------------------------
case "${task_id}" in
    1)
        n_layers=1
        n_qubits=2
        train_kernel=false
        ;;
    2)
        n_layers=2
        n_qubits=8
        train_kernel=true
        ;;
    *)
        echo "Invalid task id: ${task_id}"
        exit 1
        ;;
esac

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
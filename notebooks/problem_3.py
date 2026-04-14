import os
import shutil
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import setup_latex_environment
setup_latex_environment()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cudaq
from tueplots import bundles
plt.style.use(bundles.icml2022())
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amsfonts}"
})
cudaq.set_target("qpp-cpu")
cudaq.set_random_seed(0)

@cudaq.kernel
def measure_one_qubit():
    q = cudaq.qubit() # Allocate a qubit
    h(q) # Apply Hadamard gate to create superposition
    mz(q) # Measure in the Z-basis, collapsing the state to |0> or |1>

def estimate_p0(num_collapses: int) -> float:
    result = cudaq.sample(measure_one_qubit, shots_count=num_collapses) # Sample the kernel
    zeros = result["0"] if "0" in result else 0
    ones  = result["1"] if "1" in result else 0
    total = zeros + ones
    return zeros/total, ones/total

def main():
    os.makedirs("PS1/results", exist_ok=True)

    Ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] 
    p0_list = []
    p1_list = []

    print(f"{'N (collapses)':<15} | {'p(|0>)':<8} | {'Abs Dev (pp)':<16}")
    print("-" * 45)

    for N in Ns:
        p0, p1 = estimate_p0(N)
        p0_list.append(p0)
        p1_list.append(p1)
        abs_dev = abs(p0 - 0.5) * 100.0
        print(f"{N:15d} | {p0:8.4f} | {abs_dev:16.3f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    x = np.arange(len(Ns))
    width = 0.35
    
    # Plot 1: Bar chart of frequencies
    ax1.bar(x - width/2, p0_list, width, label=r'$|0\rangle$ Frequency ($N_0$)', color='#3498db')
    ax1.bar(x + width/2, p1_list, width, label=r'$|1\rangle$ Frequency', color='#e74c3c')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(Ns, rotation=45)
    ax1.set_xlabel(r"Number of Collapses ($N$)")
    ax1.set_ylabel("")
    ax1.set_title(r"Quantum Superposition Measurement: $H|0\rangle$")
    ax1.set_ylim(0, 1.1)
    ax1.axhline(0.5, color='black', linestyle='--', alpha=0.5, label=r'Expected $0.5$')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Deviation vs 1/sqrt(N) scaling (Central Limit Theorem)
    deviations = [abs(p0 - 0.5) for p0 in p0_list]
    theoretical_std = [1 / np.sqrt(N) for N in Ns]
    
    ax2.loglog(Ns, deviations, 'o-', label=r'Measured $|\hat{p}-\frac{1}{2}|$', color='#3498db', markersize=4, linewidth=2)
    ax2.loglog(Ns, theoretical_std, '--', label=r'Theoretical $\propto 1/\sqrt{N}$', color='black', alpha=0.5, linewidth=2)
    
    ax2.set_xlabel(r"Number of Collapses ($N$)")
    ax2.set_ylabel("")
    ax2.set_title(r"Convergence rate")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')

    plt.savefig("PS1/results/problem_3.pdf", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
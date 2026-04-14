import cudaq 
import os
import qutip
import numpy as np

cudaq.set_target("qpp-cpu")
cudaq.set_random_seed(0)

@cudaq.kernel
def rotate_x(theta: float):
    q = cudaq.qubit()
    rx(theta, q) # Rotate the qubit around the X-axis by angle theta
    mz(q) # Measure in the Z-basis

def main():
    os.makedirs("PS1/results", exist_ok=True)

    theta = 2 * np.pi
    sphere = cudaq.add_to_bloch_sphere(cudaq.get_state(rotate_x, theta))
    sphere.save("PS1/results/problem_4.pdf")

if __name__ == "__main__":
    main()
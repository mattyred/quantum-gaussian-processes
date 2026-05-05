from src.utils import set_seed
import os
import numpy as np
import matplotlib.pyplot as plt


def zscore_normalization(X, mean=None, std=None, eps=1e-10):
    if X is None:
        return None, None, None

    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / (std + eps)

    return X_normalized, mean, std


def make_random_gap(X, gap_ratio=0.2):
    a, b = X.min(), X.max()

    gap_a = a + np.random.rand() * (b - a) * (1 - gap_ratio)
    gap_b = gap_a + (b - a) * gap_ratio

    idx = np.logical_and(gap_a < X, X < gap_b)

    if gap_a - a > b - gap_b:
        X[idx] = a + np.random.rand(idx.sum()) * (gap_a - a)
    else:
        X[idx] = gap_b + np.random.rand(idx.sum()) * (b - gap_b)

    return gap_a, gap_b


def gp_sample(X, ampl=1, leng=1, sn2=0.1):
    n = X.shape[0]
    x = X / leng

    sum_xx = np.sum(x * x, axis=1).reshape(-1, 1).repeat(n, axis=1)
    D = sum_xx + sum_xx.T - 2 * np.matmul(x, x.T)

    C = ampl**2 * np.exp(-0.5 * D) + np.eye(n) * sn2

    return np.random.multivariate_normal(np.zeros(n), C).reshape(-1, 1)


def generate_gap_gp_dataset(
    seed=1,
    N=64,
    M=100,
    a=-10,
    b=10,
    gap_ratio=0.4,
    ampl=1.6,
    leng=1.8,
    sn2=0.1,
    save_dir="data",
    filename="gap_gp_1d.npz",
    make_plot=True,
):
    set_seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    # Generate training inputs
    X = np.random.rand(N, 1) * (b - a) + a

    # Create gap
    gap_a, gap_b = make_random_gap(X, gap_ratio=gap_ratio)

    # Sample GP function values
    y = gp_sample(X, ampl=ampl, leng=leng, sn2=sn2)

    # Test grid
    Xtest = np.linspace(a - 5, b + 5, M).reshape(-1, 1)

    # Normalize using train statistics only
    X_, X_mean, X_std = zscore_normalization(X)
    y_, y_mean, y_std = zscore_normalization(y)
    Xtest_, _, _ = zscore_normalization(Xtest, X_mean, X_std)

    save_path = os.path.join(save_dir, filename)

    np.savez(
        save_path,
        # raw data
        X=X,
        y=y,
        Xtest=Xtest,

        # normalized data
        X_norm=X_,
        y_norm=y_,
        Xtest_norm=Xtest_,

        # normalization statistics
        X_mean=X_mean,
        X_std=X_std,
        y_mean=y_mean,
        y_std=y_std,

        # metadata
        seed=seed,
        N=N,
        M=M,
        a=a,
        b=b,
        gap_ratio=gap_ratio,
        gap_a=gap_a,
        gap_b=gap_b,
        ampl=ampl,
        leng=leng,
        sn2=sn2,
    )

    print(f"Saved dataset to: {save_path}")

    if make_plot:
        plt.figure(figsize=(6, 3))
        plt.scatter(X, y, s=25, label="train")
        plt.axvspan(gap_a, gap_b, alpha=0.15, label="removed gap")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return save_path


if __name__ == "__main__":
    generate_gap_gp_dataset()
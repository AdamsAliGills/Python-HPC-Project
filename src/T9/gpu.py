from provided_script import *
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import argparse
from numba import jit, cuda
import cupy as cp
from os.path import join


# Data loading

def load_buildings(N, seed=42):
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    random.seed(seed)
    print(f"Total buildings available: {len(building_ids)}")
    building_ids = random.sample(building_ids, N)

    buildings = []
    for bid in building_ids:
        u0, interior_mask = load_data(LOAD_DIR, bid)
        buildings.append((u0, interior_mask))

    return buildings, building_ids



# JIT CPU version

@jit(nopython=True)
def jacobi_jit(u, interior_mask, max_iter=20_000, atol=1e-6):
    rows, cols = u.shape
    u_new = u.copy()

    for _ in range(max_iter):
        delta = 0.0
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if interior_mask[i - 1, j - 1]:
                    val = 0.25 * (
                        u[i, j - 1] + u[i, j + 1] +
                        u[i - 1, j] + u[i + 1, j]
                    )
                    diff = abs(u[i, j] - val)
                    u_new[i, j] = val
                    if diff > delta:
                        delta = diff

        u[:] = u_new
        if delta < atol:
            break

    return u



# CUDA kernel

@cuda.jit
def jacobi_cuda_kernel(u, u_new, interior_mask):
    i, j = cuda.grid(2)
    if 1 <= i < u.shape[0] - 1 and 1 <= j < u.shape[1] - 1:
        if interior_mask[i - 1, j - 1]:
            u_new[i, j] = 0.25 * (
                u[i - 1, j] + u[i + 1, j] +
                u[i, j - 1] + u[i, j + 1]
            )


def run_cuda(buildings, max_iter):
    results = []

    for u, mask in buildings:
        d_u = cuda.to_device(u)
        d_mask = cuda.to_device(mask)
        d_u_new = cuda.device_array_like(d_u)

        threads = (16, 16)
        blocks = (
            (u.shape[0] + threads[0] - 1) // threads[0],
            (u.shape[1] + threads[1] - 1) // threads[1]
        )

        # Fixed-iteration Jacobi (documented in report)
        for _ in range(max_iter):
            jacobi_cuda_kernel[blocks, threads](d_u, d_u_new, d_mask)
            d_u, d_u_new = d_u_new, d_u

        cuda.synchronize()
        results.append(d_u.copy_to_host())

    return results



# CuPy version

def jacobi_cupy(u, interior_mask, max_iter=20_000, atol=1e-6):
    u = cp.asarray(u)
    mask = cp.asarray(interior_mask, dtype=cp.bool_)

    for _ in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:] +
            u[:-2, 1:-1] + u[2:, 1:-1]
        )

        diff = cp.abs(u[1:-1, 1:-1][mask] - u_new[mask]).max()
        u[1:-1, 1:-1][mask] = u_new[mask]

        if diff < atol:
            break

    return u.get()


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    parser.add_argument("mode", choices=["provided", "parallel", "jit", "cuda", "cupy"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    buildings, building_ids = load_buildings(args.N)

    if args.mode == "provided":
        all_u = [jacobi(u, m, MAX_ITER, ABS_TOL) for u, m in buildings]

    elif args.mode == "parallel":
        func = partial(jacobi, max_iter=MAX_ITER, atol=ABS_TOL)
        with Pool(args.workers) as pool:
            if args.dynamic:
                all_u = list(pool.imap(func, buildings, chunksize=1))
            else:
                all_u = pool.starmap(func, buildings)

    elif args.mode == "jit":
        all_u = [jacobi_jit(u, m, MAX_ITER, ABS_TOL) for u, m in buildings]

    elif args.mode == "cuda":
        all_u = run_cuda(buildings, MAX_ITER)

    elif args.mode == "cupy":
        all_u = [jacobi_cupy(u, m, MAX_ITER, ABS_TOL) for u, m in buildings]

    if args.plot:
        fig, axs = plt.subplots(1, 4, figsize=(12, 5))
        for i in range(min(4, len(all_u))):
            axs[i].imshow(all_u[i], cmap="magma")
            axs[i].set_title(building_ids[i])
            axs[i].axis("off")
        plt.tight_layout()
        plt.savefig(f"output/plots/q9_{args.mode}.png")
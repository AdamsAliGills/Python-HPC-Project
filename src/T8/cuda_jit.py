from provided_script import *
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import argparse
from numba import jit, cuda
import cupy as cp

def load_buildings(N, seed=42):
    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    # Random sample of size N
    random.seed(42)
    building_ids = random.sample(building_ids, args.N)

    buildings = []
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        buildings.append((u0, interior_mask))
    
    return buildings, building_ids

@cuda.jit
def jacobi_cuda_kernel(u, u_new, interior_mask):
    i, j = cuda.grid(2)  # 2D grid
    if 1 <= i < u.shape[0]-1 and 1 <= j < u.shape[1]-1:
        if interior_mask[i-1, j-1]:
            u_new[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])

def run_cuda(buildings, max_iter=20_000):
    d_us, d_masks = [], []
    for u, mask in buildings:
        d_us.append(cuda.to_device(u))
        d_masks.append(cuda.to_device(mask))

    all_u = []
    for d_u, d_mask in zip(d_us, d_masks):
        rows, cols = d_u.shape
        threads_per_block = (16, 16)  # 16x16 = 256 threads per block
        blocks_per_grid = (
        (rows + 16 - 1) // 16,
        (cols + 16 - 1) // 16
        )
        d_u_new = cuda.to_device(d_u.copy_to_host())
    
        for _ in range(MAX_ITER):
            jacobi_cuda_kernel[blocks_per_grid, threads_per_block](d_u, d_u_new, d_mask)
            d_u, d_u_new = d_u_new, d_u
        cuda.synchronize()
        all_u.append(d_u.copy_to_host())
    return all_u

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    parser.add_argument("mode", default='provided', choices=['parallel', 'jit', 'cuda', 'cupy'])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=False)
    args = parser.parse_args()

    print(f"Args: {vars(args)}")

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    buildings, building_ids = load_buildings(args.N)

    if args.mode == 'cuda':
        all_u = run_cuda(buildings)
    
    if args.plot:
        fig, axs = plt.subplots(1, 4, figsize=(12, 5))

        for i, building_id in enumerate(building_ids[:4]):
            axs[i].imshow(all_u[i], cmap="magma")
            axs[i].set_title(f"Simulation result for building {building_id}")
            axs[i].axis("off")

        plt.tight_layout()
        plt.savefig(f"output/plots/sim_result_example_{args.mode}.png")

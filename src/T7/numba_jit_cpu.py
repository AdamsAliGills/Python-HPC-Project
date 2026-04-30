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

@jit(nopython=True)
def jacobi_jit(u, interior_mask, max_iter=20_000, atol=1e-6):
    rows, cols = u.shape
    u_new = u.copy()
    for _ in range(max_iter):
        delta = 0.0
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                if interior_mask[i-1, j-1]:
                    u_new[i, j] = 0.25 * (u[i, j-1] + u[i, j+1] + u[i-1, j] + u[i+1, j])
                    diff = abs(u[i, j] - u_new[i, j])
                    if diff > delta:
                        delta = diff
        u[:] = u_new
        if delta < atol:
            break
    return u

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

    if args.mode == 'jit':
        all_u = [jacobi_jit(u, interior_mask, max_iter=MAX_ITER, atol=ABS_TOL) for u, interior_mask in buildings]

    if args.plot:
        fig, axs = plt.subplots(1, 4, figsize=(12, 5))

        for i, building_id in enumerate(building_ids[:4]):
            axs[i].imshow(all_u[i], cmap="magma")
            axs[i].set_title(f"Simulation result for building {building_id}")
            axs[i].axis("off")

        plt.tight_layout()
        plt.savefig(f"output/plots/sim_result_example_{args.mode}.png")

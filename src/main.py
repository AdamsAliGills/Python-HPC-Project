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

@cuda.jit
def jacobi_cuda(u, u_new, interior_mask):
    i, j = cuda.grid(2)  # 2D grid
    if 1 <= i < u.shape[0]-1 and 1 <= j < u.shape[1]-1:
        if interior_mask[i-1, j-1]:
            u_new[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1])


def jacobi_cupy(u, interior_mask, max_iter=20_000, atol=1e-6):
    u = cp.asarray(u)
    interior_mask_cp = cp.asarray(interior_mask, dtype=cp.bool_)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask_cp]
        delta = cp.abs(u[1:-1, 1:-1][interior_mask_cp] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask_cp] = u_new_interior

        if delta < atol:
            break
    return u.get()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)
    parser.add_argument("mode", default='parallel', choices=['parallel', 'jit', 'cuda', 'cupy'])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dynamic", action="store_true", default=False)
    parser.add_argument("--plot", action="store_true", default=False)
    args = parser.parse_args()

    print(f"Args: {vars(args)}")

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    buildings, building_ids = load_buildings(args.N)

    if args.mode == 'parallel':
        # Make imap possible
        jacobi_func = partial(jacobi, max_iter=MAX_ITER, atol=ABS_TOL)
        def run_jacobi(args):
            return jacobi_func(*args)

        with Pool(args.workers) as pool:
            if args.dynamic:
                all_u = list(pool.imap(run_jacobi, buildings, chunksize=1))
            else:
                all_u = pool.starmap(jacobi_func, buildings)

    elif args.mode == 'jit':
        all_u = [jacobi_jit(u, interior_mask, max_iter=MAX_ITER, atol=ABS_TOL) for u, interior_mask in buildings]

    elif args.mode == 'cuda':
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
                jacobi_cuda[blocks_per_grid, threads_per_block](d_u, d_u_new, d_mask)
                d_u, d_u_new = d_u_new, d_u
            cuda.synchronize()
            all_u.append(d_u.copy_to_host())
    
    elif args.mode == 'cupy':
        all_u = [jacobi_cupy(u, interior_mask, max_iter=MAX_ITER, atol=ABS_TOL) for u, interior_mask in buildings]

    if args.plot:
        fig, axs = plt.subplots(1, 4, figsize=(12, 5))

        for i, building_id in enumerate(building_ids[:4]):
            axs[i].imshow(all_u[i], cmap="magma")
            axs[i].set_title(f"Simulation result for building {building_id}")
            axs[i].axis("off")

        plt.tight_layout()
        plt.savefig("output/plots/sim_result_demo.png")

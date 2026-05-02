from provided_script import *
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
import argparse
from numba import jit, cuda
import cupy as cp
from os.path import join


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


def jacobi_cupy(u, interior_mask, max_iter=20_000, atol=1e-6):
    u = cp.asarray(u)
    mask = cp.asarray(interior_mask, dtype=cp.bool_)

    for _ in range(max_iter):
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])

        diff = cp.abs(u[1:-1, 1:-1][mask] - u_new[mask]).max()
        u[1:-1, 1:-1][mask] = u_new[mask]

        if diff < atol:
            break

    return u.get()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int)

    args = parser.parse_args()

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    buildings, building_ids = load_buildings(args.N)

    all_u = [jacobi_cupy(u, m, MAX_ITER, ABS_TOL) for u, m in buildings]

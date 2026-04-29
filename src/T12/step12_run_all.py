from os.path import join, exists
import os
import csv
import argparse
from multiprocessing import Pool

import numpy as np
from numba import jit

LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
SIZE = 512
MAX_ITER = 20_000
ABS_TOL = 1e-4


def load_data(load_dir, bid):
    u = np.zeros((SIZE + 2, SIZE + 2), dtype=np.float64)
    u[1:-1, 1:-1] = np.load(join(load_dir, f"{bid}_domain.npy"))
    interior_mask = np.load(join(load_dir, f"{bid}_interior.npy"))
    return u, interior_mask


def jacobi_numpy(u, interior_mask, max_iter=MAX_ITER, atol=ABS_TOL):
    u = np.copy(u)

    for _ in range(max_iter):
        u_new = 0.25 * (
            u[1:-1, :-2] +
            u[1:-1, 2:] +
            u[:-2, 1:-1] +
            u[2:, 1:-1]
        )
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break

    return u


@jit(nopython=True)
def jacobi_jit(u, interior_mask, max_iter=MAX_ITER, atol=ABS_TOL):
    rows, cols = u.shape
    u_new = u.copy()

    for _ in range(max_iter):
        delta = 0.0

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if interior_mask[i - 1, j - 1]:
                    new_val = 0.25 * (
                        u[i, j - 1] +
                        u[i, j + 1] +
                        u[i - 1, j] +
                        u[i + 1, j]
                    )
                    u_new[i, j] = new_val

                    diff = abs(u[i, j] - new_val)
                    if diff > delta:
                        delta = diff
                else:
                    u_new[i, j] = u[i, j]

        u[:, :] = u_new[:, :]

        if delta < atol:
            break

    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100.0
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100.0
    return mean_temp, std_temp, pct_above_18, pct_below_15


def process_one_building(task):
    bid, mode = task
    u0, interior_mask = load_data(LOAD_DIR, bid)

    if mode == "jit":
        u = jacobi_jit(u0, interior_mask, MAX_ITER, ABS_TOL)
    elif mode == "parallel":
        u = jacobi_numpy(u0, interior_mask, MAX_ITER, ABS_TOL)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    mean_temp, std_temp, pct_above_18, pct_below_15 = summary_stats(u, interior_mask)
    return [bid, mean_temp, std_temp, pct_above_18, pct_below_15]


def load_building_ids():
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()
    return building_ids


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def read_completed_ids(output_csv):
    completed = set()
    if not exists(output_csv):
        return completed

    with open(output_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(str(row["building_id"]))
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["jit", "parallel"], default="jit")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=str, default="output/step12/all_buildings_results.csv")
    parser.add_argument("--start", type=int, default=0, help="Start index in building_ids list (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index in building_ids list (exclusive)")
    parser.add_argument("--resume", action="store_true", help="Skip building IDs already present in output CSV")
    args = parser.parse_args()

    ensure_parent_dir(args.output)

    building_ids = load_building_ids()
    total_buildings = len(building_ids)

    start = max(0, args.start)
    end = total_buildings if args.end is None else min(args.end, total_buildings)
    building_ids = building_ids[start:end]

    if len(building_ids) == 0:
        print("No buildings selected.")
        return

    completed_ids = set()
    file_exists = exists(args.output)

    if args.resume:
        completed_ids = read_completed_ids(args.output)
        building_ids = [bid for bid in building_ids if str(bid) not in completed_ids]

    print(f"Mode: {args.mode}")
    print(f"Workers: {args.workers}")
    print(f"Selected building index range: [{start}, {end})")
    print(f"Buildings to process now: {len(building_ids)}")
    print(f"Output CSV: {args.output}")

    if len(building_ids) == 0:
        print("Nothing left to do.")
        return

    # Warm up Numba once for JIT mode
    if args.mode == "jit":
        warmup_bid = building_ids[0]
        print(f"Warming up Numba using building {warmup_bid} ...")
        u0, interior_mask = load_data(LOAD_DIR, warmup_bid)
        _ = jacobi_jit(u0, interior_mask, 10, ABS_TOL)

    write_header = not file_exists or (file_exists and os.path.getsize(args.output) == 0)

    with open(args.output, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "building_id",
                "mean_temp",
                "std_temp",
                "pct_above_18",
                "pct_below_15",
            ])
            f.flush()

        if args.mode == "parallel":
            tasks = [(bid, args.mode) for bid in building_ids]
            with Pool(args.workers) as pool:
                for idx, row in enumerate(pool.imap_unordered(process_one_building, tasks, chunksize=1), start=1):
                    writer.writerow(row)
                    f.flush()
                    print(f"[{idx}/{len(building_ids)}] finished building {row[0]}")
        else:
            for idx, bid in enumerate(building_ids, start=1):
                row = process_one_building((bid, args.mode))
                writer.writerow(row)
                f.flush()
                print(f"[{idx}/{len(building_ids)}] finished building {row[0]}")

    print("Done.")


if __name__ == "__main__":
    main()

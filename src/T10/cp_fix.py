from provided_script import *
import random
import matplotlib.pyplot as plt
import cupy as cp
import argparse
from os.path import join


# ---------------------------------------------------------------------------
# Data loading (unchanged from Q9)
# ---------------------------------------------------------------------------

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




def jacobi_cupy_optimized(u, interior_mask, max_iter=20_000, atol=1e-6,
                          check_every=50):
  
    # Transfer to GPU once
    u = cp.asarray(u, dtype=cp.float64)
    mask = cp.asarray(interior_mask, dtype=cp.bool_)

    # Fix 2: pre-cast mask to float so multiply works without extra cast kernels
    float_mask = mask.astype(cp.float64)

    for it in range(max_iter):

        # Compute Jacobi average — same expression as before
        u_new = 0.25 * (
            u[1:-1, :-2] + u[1:-1, 2:] +
            u[:-2, 1:-1] + u[2:, 1:-1]
        )

        # Fix 2: masked max without boolean gather
        # abs_diff is computed over the full interior patch (512×512),
        # then non-interior cells are zeroed by multiplying with float_mask
        # before taking the global max — no DtoD copy, no cupy_getitem_mask.
        abs_diff = cp.abs(u[1:-1, 1:-1] - u_new) * float_mask

        # Fix 1: update via cp.where — fused element-wise op, no DtoD copy
        u[1:-1, 1:-1] = cp.where(mask, u_new, u[1:-1, 1:-1])

        # Fix 3: only sync to CPU every `check_every` steps
        if (it + 1) % check_every == 0:
            diff = float(abs_diff.max())   # single DtoH transfer
            if diff < atol:
                break

    # One final transfer back to CPU
    return u.get()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Q10: optimised CuPy Jacobi solver with nsys-guided fixes"
    )
    parser.add_argument("N", type=int, help="Number of buildings to process")
   
    parser.add_argument("--check_every", type=int, default=50,
                        help="Convergence check interval (optimized only)")
    parser.add_argument("--plot", action="store_true",
                        help="Save output plots")
    args = parser.parse_args()

    MAX_ITER = 20_000
    ABS_TOL  = 1e-4

    buildings, building_ids = load_buildings(args.N)


    all_u = [jacobi_cupy_optimized(u, m, MAX_ITER, ABS_TOL, check_every=args.check_every) for u, m in buildings]

    # Summary statistics (unchanged from reference)
    stat_keys = ['mean_temp', 'std_temp', 'pct_above_18', 'pct_below_15']
    print('building_id, ' + ', '.join(stat_keys))
    for bid, u, (_, mask) in zip(building_ids, all_u, buildings):
        stats = summary_stats(u, mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))

    if args.plot:
        fig, axs = plt.subplots(1, 4, figsize=(12, 5))
        for i in range(min(4, len(all_u))):
            axs[i].imshow(all_u[i], cmap="magma")
            axs[i].set_title(building_ids[i])
            axs[i].axis("off")
        plt.tight_layout()
        plt.savefig(f"output/plots/q10_cupy_optimized.png")
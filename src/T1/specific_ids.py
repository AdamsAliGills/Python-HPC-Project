from os.path import join
import sys
import numpy as np

datapath = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
id_args = []
SIZE = 512


def load_data():
    # u_refactor = np.zeros((SIZE + 2, SIZE + 2))
    u = np.zeros((len(id_args), SIZE + 2, SIZE + 2))
    interior_mask = np.zeros((len(id_args), SIZE, SIZE), dtype="bool")

    for i, building_id in enumerate(id_args):
        u[i, 1:-1, 1:-1] = np.load(join(datapath, f"{building_id}_domain.npy"))
        # u[i] = u_refactor
        interior_mask[i] = np.load(join(datapath, f"{building_id}_interior.npy"))

    return u, interior_mask


def jacobi(u, interior_mask, max_iter, atol=1e-6):
    u = np.copy(u)

    for i in range(max_iter):
        # Compute average of left, right, up and down neighbors, see eq. (1)
        u_new = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        u_new_interior = u_new[interior_mask]
        delta = np.abs(u[1:-1, 1:-1][interior_mask] - u_new_interior).max()
        u[1:-1, 1:-1][interior_mask] = u_new_interior

        if delta < atol:
            break
    return u


def summary_stats(u, interior_mask):
    u_interior = u[1:-1, 1:-1][interior_mask]
    mean_temp = u_interior.mean()
    std_temp = u_interior.std()
    pct_above_18 = np.sum(u_interior > 18) / u_interior.size * 100
    pct_below_15 = np.sum(u_interior < 15) / u_interior.size * 100
    return {
        "mean_temp": mean_temp,
        "std_temp": std_temp,
        "pct_above_18": pct_above_18,
        "pct_below_15": pct_below_15,
    }


def main():

    if len(sys.argv) < 2:
        print("please provide building id")
    else:
        for i in range(1, len(sys.argv)):
            id_args.append(int(sys.argv[i]))

    all_u0, all_interior_mask = load_data()

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    stat_keys = ["mean_temp", "std_temp", "pct_above_18", "pct_below_15"]
    print("building_id, " + ", ".join(stat_keys))  # CSV header
    for bid, u, interior_mask in zip(id_args, all_u, all_interior_mask):
        stats = summary_stats(u, interior_mask)
        print(f"{bid},", ", ".join(str(stats[k]) for k in stat_keys))


if __name__ == "__main__":
    main()

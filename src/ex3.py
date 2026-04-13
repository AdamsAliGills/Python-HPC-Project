from provided_script import *
import random
import matplotlib.pyplot as plt

if __name__=='__main__':
    #N = int(sys.argv[1])
    N=3

    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    # Random sample of size N
    random.seed(42)
    building_ids = random.sample(building_ids, N)

    all_u0 = np.empty((N, 514, 514))
    all_interior_mask = np.empty((N, 512, 512), dtype="bool")
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        all_u0[i] = u0
        all_interior_mask[i] = interior_mask

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    all_u = np.empty_like(all_u0)
    for i, (u0, interior_mask) in enumerate(zip(all_u0, all_interior_mask)):
        u = jacobi(u0, interior_mask, MAX_ITER, ABS_TOL)
        all_u[i] = u

    fig, axs = plt.subplots(1, N, figsize=(12, 5))

    for i, building_id in enumerate(building_ids):
        axs[i].imshow(all_u[i], cmap="magma")
        axs[i].set_title(f"Simulation result for building {building_id}")
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig("output/plots/sim_result_demo.png")

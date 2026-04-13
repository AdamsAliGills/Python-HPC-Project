from provided_script import *
import random
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial


if __name__=='__main__':
    N = int(sys.argv[1])

    LOAD_DIR = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    with open(join(LOAD_DIR, "building_ids.txt"), "r") as f:
        building_ids = f.read().splitlines()

    # Random sample of size N
    random.seed(42)
    building_ids = random.sample(building_ids, N)

    buildings = []
    for i, bid in enumerate(building_ids):
        u0, interior_mask = load_data(LOAD_DIR, bid)
        buildings.append((u0, interior_mask))

    MAX_ITER = 20_000
    ABS_TOL = 1e-4

    with Pool() as pool:
        all_u = pool.starmap(partial(jacobi, max_iter=MAX_ITER, atol=ABS_TOL), buildings)

    fig, axs = plt.subplots(1, 4, figsize=(12, 5))

    for i, building_id in enumerate(building_ids[:4]):
        axs[i].imshow(all_u[i], cmap="magma")
        axs[i].set_title(f"Simulation result for building {building_id}")
        axs[i].axis("off")

    plt.tight_layout()
    plt.savefig("output/plots/sim_result_demo.png")

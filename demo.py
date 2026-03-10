import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

datapath = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
ids = [63, 2478, 122, 3194, 2193, 167, 446, 2584]

fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.flatten()

for i, id_val in enumerate(ids):
    ax = axs[i]
    ax.axis("off")
    ax.set_aspect("equal")

    f_path = os.path.join(datapath, "struct_in", f"{id_val}.npy")
    if os.path.exists(f_path):
        stack = np.load(f_path)
        ax.imshow(stack[..., 0].astype(np.uint8), cmap="gray")
        ax.set_title(f"ID: {id_val}")

plt.tight_layout()
plt.savefig("output_viz.png")

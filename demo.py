import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

datapath = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"

path = {
    "full": datapath,
    "graph_in": os.path.join(datapath, "graph_in"),
    "struct_in": os.path.join(datapath, "struct_in"),
    "full_out": os.path.join(datapath, "full_out"),
    "graph_out": os.path.join(datapath, "graph_out"),
}

ids = [63, 2478, 122, 3194, 2193, 167, 446, 2584]

# set up figure
fs = 10
fig, axs = plt.subplots(2, 4, figsize=(fs * 4, fs * 2))
axs = axs.flatten()

for i, id in enumerate(ids):
    # set axis
    ax = axs[i]
    _ = [ax.axis("off"), ax.axes.set_aspect("equal")]

    # get structural components
    stack = np.load(os.path.join(path["struct_in"], f"{id}.npy"))

    # channel 1: structural components
    # note: channel 2 and 3 are x and y locations
    #   this holds for "full_out" as well
    struct = stack[..., 0].astype(np.uint8)
    ax.imshow(struct, cmap="gray")

plt.tight_layout()
plt.savefig("output_viz_2.png")

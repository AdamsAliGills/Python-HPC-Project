import os
import numpy as np
import matplotlib.pyplot as plt

datapath = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
ids = [156, 1573, 1575]

fig, axs = plt.subplots(2, 3, figsize=(15, 10))

for i, building_id in enumerate(ids):
    domain_path = os.path.join(datapath, f"{building_id}_domain.npy")
    interior_path = os.path.join(datapath, f"{building_id}_interior.npy")

    domain = np.load(domain_path)
    interior = np.load(interior_path)

    axs[0, i].imshow(domain, cmap="magma")
    axs[0, i].set_title(f"Initial: {building_id}")
    axs[0, i].axis("off")

    axs[1, i].imshow(interior, cmap="gray")
    axs[1, i].set_title(f"Mask: {building_id}")
    axs[1, i].axis("off")

plt.tight_layout()
plt.savefig("output/plots/initial_and_mask.png")
plt.show()

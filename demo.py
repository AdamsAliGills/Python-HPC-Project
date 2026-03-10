import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget


def run_viewer(ids, data_path):
    app = QApplication(sys.argv)
    win = QWidget()
    win.setWindowTitle("HPC Structural Viewer")
    layout = QVBoxLayout(win)

    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()

    for i, id_val in enumerate(ids):
        ax = axs[i]
        ax.axis("off")
        try:
            f_path = os.path.join(data_path, "struct_in", f"{id_val}.npy")
            stack = np.load(f_path)
            ax.imshow(stack[..., 0].astype(np.uint8), cmap="gray")
            ax.set_title(f"ID: {id_val}")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {id_val}", ha="center")

    layout.addWidget(FigureCanvas(fig))
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    datapath = "/dtu/projects/02613_2025/data/modified_swiss_dwellings/"
    ids = [63, 2478, 122, 3194, 2193, 167, 446, 2584]
    run_viewer(ids, datapath)

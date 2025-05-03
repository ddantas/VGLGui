import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# Carregar volume 3D (7 fatias)
volume = tiff.imread("3d_dilate.tiff")

fig, axes = plt.subplots(1, 7, figsize=(14, 4))  # Grid de 1 linha e 7 colunas

for i in range(7):
    axes[i].imshow(volume[i], cmap="gray")
    axes[i].axis("off")  # Remove eixos

plt.tight_layout()
plt.savefig("fatias_grid.png", dpi=300)  # Salva a imagem para Overleaf
plt.show()

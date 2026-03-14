import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.morphology import binary_opening

image=np.load("./wires/wires6.npy")
struct=np.ones((3, 1))


labeled_image = label(image)
for i in range(1, np.max(labeled_image)+1):
    process = binary_opening(labeled_image==i, struct)
    tmp = label(process)
    if np.max(tmp)==1:
        print(f"Провод {i} целый")
    else:
        print(f"Провод {i} разрезан на {np.max(tmp)} частей")


plt.imshow(image)
plt.show()
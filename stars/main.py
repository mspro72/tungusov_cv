import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops

plus_pat = np.array([[0,0,1,0,0],
                     [0,0,1,0,0],
                     [1,1,1,1,1],
                     [0,0,1,0,0],
                     [0,0,1,0,0]])

cross_pat = np.array([[1,0,0,0,1],
                      [0,1,0,1,0],
                      [0,0,1,0,0],
                      [0,1,0,1,0],
                      [1,0,0,0,1]])

image = np.load("stars.npy")
labeled = label(image)
stars = 0

for p in regionprops(labeled):
    img = p.image.astype(int)
    if np.array_equal(img, plus_pat):
        stars += 1
    if np.array_equal(img, cross_pat):
        stars += 1

print(stars)
plt.imshow(labeled)
plt.show()

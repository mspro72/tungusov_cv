import numpy as np
import matplotlib.pyplot as plt

def lerp(v0, v1, t):
    return (1-t) * v0 + t * v1

size = 100
image = np.zeros((size, size, 3), dtype="uint8")
assert image.shape[0] == image.shape[1]

color1 = [0, 0, 255]
color2 = [255, 0, 200]
#Я убрал цикл который был основан на том что бы i записывался индекс, а в v записывалася точка на расстоянии 1/size (0,01). (i + j) / (2 * (size - 1)) i+j = 2 * size - 2 в максимальной точке, и что бы v находилась между 0 и 1 нужно делить на (2 * (size - 1)) или 2 * size - 2
for i in range(size):
    for j in range(size):
        v = (i + j) / (2 * (size - 1))
        r = lerp(color1[0], color2[0], v)
        g = lerp(color1[1], color2[1], v)
        b = lerp(color1[2], color2[2], v)
        image[i, j, :] = [r, g, b]

plt.figure(1)
plt.imshow(image)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops

def get_color_name(hue_value):
    return round(hue_value, 2)

image = imread("./balls_and_rects.png")
if image.shape[-1] == 4:
    image = image[:, :, :3]

hsv = rgb2hsv(image)
hue = hsv[:, :, 0]
binary = (hsv[:, :, 1] > 0.1) & (hsv[:, :, 2] > 0.1)

labeled = label(binary)
props = regionprops(labeled)

total_count = len(props)
stats = {
    "circles": {},
    "rects": {}
}

for region in props:
    coords = region.coords
    sample_hues = hue[coords[:, 0], coords[:, 1]]
    avg_hue = get_color_name(np.mean(sample_hues))
    
    if region.extent > 0.85:
        shape_type = "rects"
    else:
        shape_type = "circles"
    
    if avg_hue not in stats[shape_type]:
        stats[shape_type][avg_hue] = 0
    stats[shape_type][avg_hue] += 1

print(f"Общее количество фигур: {total_count}")

print("\nКоличество кругов по оттенкам:")
for h, count in sorted(stats["circles"].items()):
    print(f"  Оттенок {h}: {count}")

print("\nКоличество прямоугольников по оттенкам:")
for h, count in sorted(stats["rects"].items()):
    print(f"  Оттенок {h}: {count}")

plt.imshow(image)
plt.title(f"Total shapes: {total_count}")
plt.show()
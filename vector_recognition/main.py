import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
import os
from pathlib import Path

save_path = Path(__file__).parent

def count_bays(region):
    ch = count_holes(region)
    shape = region.image.shape
    padded = np.zeros((shape[0] + 2, shape[1] + 2))
    padded[1:-1, 1:-1] = region.image
    inverted = np.logical_not(padded)
    bays_labeled = label(inverted)
    x = sum(1 for r in regionprops(bays_labeled) if r.area < 3)
    return np.max(bays_labeled) - ch - x

def count_holes(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled) - 1

def weighted_dist(a, b):
    diff = a - b
    weights = np.ones(len(diff))
    weights[3] = 15 
    weights[4] = 8
    weights[6] = 10
    weights[7] = 4
    weights[8] = 4
    return (weights * diff**2).sum() ** 0.5

def extractor(region):
    img = region.image.astype(float)
    h, w = img.shape
    cy, cx = region.centroid_local
    cy /= h
    cx /= w
    holes      = count_holes(region)
    bays       = count_bays(region)
    area_ratio = region.area / img.size
    aspect     = h / max(w, 1)
    eccentricity = region.eccentricity
    row_fill   = np.sum(img, axis=1) / w
    col_fill   = np.sum(img, axis=0) / h
    h_sym      = 1 - np.abs(col_fill - col_fill[::-1]).mean()
    v_sym      = 1 - np.abs(row_fill - row_fill[::-1]).mean()
    top_fill   = img[:h//2, :].mean()
    bot_fill   = img[h//2:, :].mean()
    left_fill  = img[:, :w//2].mean()
    right_fill = img[:, w//2:].mean()
    return np.array([area_ratio, cy, cx, holes, bays, eccentricity, aspect, h_sym, v_sym, top_fill, bot_fill, left_fill, right_fill])

def classificator(region, templates):
    features = extractor(region)
    best_key = None
    min_dist = 10**16
    for key in templates:
        current_dist = weighted_dist(templates[key], features)
        if current_dist < min_dist:
            min_dist = current_dist
            best_key = key
    return best_key


template = imread('./alphabet-small.png')[:, :, :-1]
print(template.shape)
template = template.sum(2)
binary = template != 765

labeled = label(binary)
props = regionprops(labeled)

templates = {}

for region, symbol in zip(props, ["8", "0", "A", "B", "1", "W", "X", "*", "/", "-"]):

    templates[symbol] = extractor(region)

image = imread("./alphabet.png")[:, :, :-1]
abinary = image.mean(2) > 0
alabeled = label(abinary)
print(np.max(alabeled))
aprops = regionprops(alabeled)
res = {}

image_path = save_path / "out"
image_path.mkdir(exist_ok=True)

plt.figure(figsize=(5, 7))
for region in aprops:
    symbol = classificator(region, templates)
    if symbol not in res:
        res[symbol] = 0
    res[symbol] += 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")

print(res)
plt.imshow(abinary)
plt.show()

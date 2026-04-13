import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from skimage.io import imread
import os
from pathlib import Path

save_path = Path(__file__).parent


def count_holel(region):
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1: -1, 1: -1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    return np.max(labeled)-1

def classificator(region):
    holes = count_holel(region)
    
    if holes == 2:
        vlines = (np.sum(region.image, 0) == region.image.shape[0]).sum()
        vlines = vlines / region.image.shape[1]
        if vlines > 0.2:
            return "B"
        else:
            return "8"
    
    if holes == 1:
        shape = region.image.shape
        aspect = shape[1] / shape[0]
        mid = shape[1] // 2
        left_fill = region.image[:, :mid].sum() / (shape[0] * mid)
        right_fill = region.image[:, mid:].sum() / (shape[0] * (shape[1] - mid))
        mid_h = shape[0] // 2
        bottom_half = region.image[mid_h:, :]
        labeled_bottom = label(np.logical_not(np.pad(bottom_half, 1, constant_values=0)))
        left_fill = region.image[:, :mid].sum() / (shape[0] * mid)
        right_fill = region.image[:, mid:].sum() / (shape[0] * (shape[1] - mid))
        lr_diff = left_fill - right_fill
        bottom_holes = 0
        for r in regionprops(labeled_bottom):
            if r.label != 1 and r.area > 3:
                bottom_holes += 1


        if region.eccentricity < 0.55:
            return "A"
        
        if lr_diff > 0.18: 
            if aspect > 0.73:
                return "D"
            else:
                return "P"
        
        return "0"
    
    if (region.image.sum() / region.image.size) == 1.0:
        return "-"
    
    shape = region.image.shape
    aspect = np.min(shape) / np.max(shape)
    if aspect > 0.8:
        return "*"
    
    vlines = (np.sum(region.image, 0) == region.image.shape[0]).sum()
    hlines = (np.sum(region.image, 1) == region.image.shape[1]).sum()
    if vlines > 0 and hlines > 0:
        return "1"
    
    labeled = label(np.logical_not(region.image))
    bays = 0
    for r in regionprops(labeled):
        if r.area > 3:
            bays += 1
    
    if bays == 2:
        return "/"
    elif bays == 4:
        return "X"
    elif bays == 5:
        return "W"
    
    return "?"


image = imread("./symbols.png")[:, :, :-1]
abinary = image.mean(2) > 0
alabeled = label(abinary)
print(np.max(alabeled))
aprops = regionprops(alabeled)

result = {}

image_path = save_path / "out_tree_ext"
image_path.mkdir(exist_ok=True)

plt.figure(figsize=(5, 7))
for region in aprops:
    symbol = classificator(region)
    if symbol not in result:
        result[symbol] = 0
    result[symbol] += 1
    plt.cla()
    plt.title(f"Class - '{symbol}'")
    plt.imshow(region.image)
    plt.savefig(image_path / f"image_{region.label}.png")


print(result)
print(f"{1.0 - result.get('?', 0 )/ len(aprops)}")
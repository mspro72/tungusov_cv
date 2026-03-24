import os
import numpy as np
import matplotlib.pyplot as plt

def neighbours4(y, x):
    return (y, x+1), (y+1, x), (y, x-1), (y-1, x)

def label_objects(frame):
    labeled = np.zeros_like(frame, dtype=int)
    current_label = 0
    h, w = frame.shape
    for y in range(h):
        for x in range(w):
            if frame[y, x] > 0 and labeled[y, x] == 0:
                current_label += 1
                queue = [(y, x)]
                labeled[y, x] = current_label
                while queue:
                    cy, cx = queue.pop()
                    for ny, nx in neighbours4(cy, cx):
                        if 0 <= ny < h and 0 <= nx < w:
                            if frame[ny, nx] > 0 and labeled[ny, nx] == 0:
                                labeled[ny, nx] = current_label
                                queue.append((ny, nx))

    return labeled, current_label

def centroid(labeled, label):
    ys, xs = np.where(labeled == label)
    return np.mean(ys), np.mean(xs)

def area(labeled, label):
    return (labeled == label).sum()

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

def load_frames(data_dir="npy"):
    frames = {}
    for i in range(100):
        for name in [f"h_{i}.npy", f"h-{i}.npy"]:
            path = os.path.join(data_dir, name)
            if os.path.exists(path):
                frames[i] = np.load(path)
                break
    return frames

def get_centers(frame, min_size=50):
    labeled, num = label_objects(frame)
    centers = []
    for obj_id in range(1, num + 1):
        if area(labeled, obj_id) >= min_size:
            cy, cx = centroid(labeled, obj_id)
            centers.append((cx, cy))
    return centers

def track(frames, max_dist=100):
    indices = sorted(frames.keys())

    trajectories = []
    for x, y in get_centers(frames[indices[0]]):
        trajectories.append({"points": [(indices[0], x, y)]})

    for idx in indices[1:]:
        centers = get_centers(frames[idx])
        matched_t, matched_c = set(), set()

        pairs = []
        for ti, t in enumerate(trajectories):
            last = t["points"][-1]
            for ci, (x, y) in enumerate(centers):
                dist = distance((last[2], last[1]), (y, x))
                if dist < max_dist:
                    pairs.append((dist, ti, ci))
        pairs.sort()
        for dist, ti, ci in pairs:
            if ti not in matched_t and ci not in matched_c:
                x, y = centers[ci]
                trajectories[ti]["points"].append((idx, x, y))
                matched_t.add(ti)
                matched_c.add(ci)

        for ci, (x, y) in enumerate(centers):
            if ci not in matched_c:
                trajectories.append({"points": [(idx, x, y)]})

    return [t for t in trajectories if len(t["points"]) > 1]

frames = load_frames("npy")
trajectories = track(frames)

plt.figure(figsize=(11, 8))
for i, traj in enumerate(trajectories):
    xs = [p[1] for p in traj["points"]]
    ys = [p[2] for p in traj["points"]]
    plt.plot(xs, ys, "-o", markersize=3, label=f"Объект {i + 1}")

plt.title("Траектории объектов")
plt.legend()
plt.show()

import os
import math
from collections import Counter

def learning_rate(LABELS_DIR):
    IMG_SIZE = 640

    BASE_LR = 1e-3

    SMALL_TH = 32
    LARGE_TH = 96

    W_SMALL = 1.5
    W_MEDIUM = 1.0
    W_LARGE = 0.7

    sizes = []

    for file in os.listdir(LABELS_DIR):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(LABELS_DIR, file)

        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            _, x, y, w, h = map(float, parts)

            w_px = w * IMG_SIZE
            h_px = h * IMG_SIZE

            size = math.sqrt(w_px * h_px)
            sizes.append(size)

    counter = Counter()

    for s in sizes:
        if s < SMALL_TH:
            counter["small"] += 1
        elif s < LARGE_TH:
            counter["medium"] += 1
        else:
            counter["large"] += 1

    total = sum(counter.values())

    if total == 0:
        raise ValueError("No valid bounding boxes found!")

    f_small = counter["small"] / total
    f_medium = counter["medium"] / total
    f_large = counter["large"] / total

    print("\n--- Dataset Distribution ---")
    print(f"Small  (<{SMALL_TH}px): {counter['small']} ({f_small:.2%})")
    print(f"Medium ({SMALL_TH}-{LARGE_TH}px): {counter['medium']} ({f_medium:.2%})")
    print(f"Large  (>{LARGE_TH}px): {counter['large']} ({f_large:.2%})")

    score_small = f_small * W_SMALL
    score_medium = f_medium * W_MEDIUM
    score_large = f_large * W_LARGE

    max_score = max(score_small, score_medium, score_large)

    r_small = score_small / max_score
    r_medium = score_medium / max_score
    r_large = score_large / max_score

    print("\n--- LR Ratios (normalized) ---")
    print(f"P3 (small):  {r_small:.3f}")
    print(f"P4 (medium): {r_medium:.3f}")
    print(f"P5 (large):  {r_large:.3f}")

    LR_P3 = BASE_LR * r_small
    LR_P4 = BASE_LR * r_medium
    LR_P5 = BASE_LR * r_large

    print("\n--- Suggested Learning Rates ---")
    print(f"LR_P3: {LR_P3:.6f}")
    print(f"LR_P4: {LR_P4:.6f}")
    print(f"LR_P5: {LR_P5:.6f}")

    LR_NECK = BASE_LR * 0.8

    print(f"LR_NECK: {LR_NECK:.6f}")

    return

if __name__ == "__main__":
    learning_rate(LABELS_DIR="C:/Users/gr3g2/Desktop/trajectories_experiments/output_dataset_v3-20260318T064630Z-1-001/output_dataset_v3/labels/train")
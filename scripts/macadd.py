"""Local maCADD metric (per the competition's Evaluation page).

score = sum over IoU thresholds t in {0.2..0.9} of t * aCADD_t / sum(t)

aCADD_t = (1/A)*sum_{S} |dc| + sum_{P} |dc| + sum_{FN} c_clean + sum_{FP} c_sub

S/P depend on ground-truth object class (clean vs poisoned streak), which we
don't know locally, so this scorer supports:
  - symmetric mode (worst case: everything full penalty)
  - optimistic mode (all matched diffs in the correct direction: /A)
Matching: Hungarian on IoU (maximize IoU), pairs with IoU < t unmatched.
Clean reference detections pre-thresholded at conf > 0.2 by the host.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
A = 10.0


def parse_pred(s):
    s = str(s).strip()
    if not s or s == "nan":
        return np.zeros((0, 5), dtype=np.float64)
    v = np.array(s.split(), dtype=np.float64)
    assert len(v) % 5 == 0
    return v.reshape(-1, 5)


def iou_matrix(a, b):
    """a: (N,4) xywh, b: (M,4) xywh -> (N,M) IoU"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def acadd_t(clean, sub, t, mode="symmetric"):
    """clean/sub: (N,5) arrays [conf,x,y,w,h]. Returns aCADD at threshold t."""
    cost = 0.0
    iou = iou_matrix(clean[:, 1:], sub[:, 1:])
    matched_c = np.zeros(len(clean), bool)
    matched_s = np.zeros(len(sub), bool)
    if iou.size:
        # Hungarian: maximize total IoU among pairs with IoU >= t
        m = np.where(iou >= t, iou, -1.0)
        ri, ci = linear_sum_assignment(-m)
        for i, j in zip(ri, ci):
            if iou[i, j] >= t:
                dc = abs(clean[i, 0] - sub[j, 0])
                cost += dc / A if mode == "optimistic" else dc
                matched_c[i] = True
                matched_s[j] = True
    cost += clean[~matched_c, 0].sum()   # FN
    cost += sub[~matched_s, 0].sum()     # FP
    return cost


def macadd(clean_df, sub_df, mode="symmetric", clean_thresh=0.2):
    """DataFrames with image_id, prediction_string. Returns weighted score."""
    c_map = dict(zip(clean_df["image_id"].astype(str), clean_df["prediction_string"]))
    s_map = dict(zip(sub_df["image_id"].astype(str), sub_df["prediction_string"]))
    per_t = {t: 0.0 for t in IOU_THRESHOLDS}
    for stem in c_map:
        clean = parse_pred(c_map[stem])
        clean = clean[clean[:, 0] > clean_thresh]
        sub = parse_pred(s_map.get(stem, " "))
        for t in IOU_THRESHOLDS:
            per_t[t] += acadd_t(clean, sub, t, mode)
    wsum = sum(IOU_THRESHOLDS)
    return sum(t * per_t[t] for t in IOU_THRESHOLDS) / wsum, per_t

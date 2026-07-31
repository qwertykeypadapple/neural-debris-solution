"""Per-detection confidence policy (replaces global boost constants).

For each kept student detection with confidence c_s, submit at the optimal
percentile of our belief over the clean model's confidence:

    G(c*) = (2p - 1) / (1.1 p)          [first-order condition of aCADD]

where p(c_s) is the modeled probability the clean model has a matching
detection (>0.2), and the belief is Normal(c_s, sigma). High-confidence
detections boost toward ~0.98 (recovering the old global boost as a special
case); uncertain ones (p near 0.5) DEFLATE below raw confidence — the piece
the global curve got wrong.

Suppressed teacher detections keep the tiny-eps floor: for match rates below
~0.91 the same derivation shows minimal confidence is already optimal.

Usage:
  python policy_submission.py student.json teacher.json out.csv \
      --eps 0.003 --sigma 0.15 --tmin 0.2 --teacher-tmin 0.2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))
from macadd import iou_matrix


def match_prob(c_s):
    """P(clean model has a matching det > 0.2) as a function of student conf.

    Anchored to what we can defend: retention analysis showed the student
    tracks the teacher closely on real objects (median |dconf| ~ 0.06), and
    the floor A/B measured ~75% match rate even for suppressed boxes. Smooth
    monotone ramp: ~0.55 at c_s=0.2 up to 0.97 for c_s >= 0.8.
    """
    return float(np.clip(0.55 + 0.7 * (c_s - 0.2), 0.05, 0.97))


def policy_conf(c_s, sigma, cap):
    p = match_prob(c_s)
    if p <= 0.5:
        return 0.01
    g_star = np.clip((2 * p - 1) / (1.1 * p), 0.001, 0.95)
    c_star = c_s + sigma * norm.ppf(g_star)
    return float(np.clip(c_star, 0.01, cap))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("student")
    ap.add_argument("teacher")
    ap.add_argument("out")
    ap.add_argument("--eps", type=float, default=0.003)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--cap", type=float, default=0.985)
    ap.add_argument("--tmin", type=float, default=0.2)
    ap.add_argument("--teacher-tmin", type=float, default=0.2)
    ap.add_argument("--match-iou", type=float, default=0.45)
    args = ap.parse_args()

    S = json.load(open(args.student))
    T = json.load(open(args.teacher))

    stems = sorted(set(S) | set(T))
    assert len(stems) == 2000, len(stems)
    rows, mass, n, n_floored, n_deflated, n_boosted = [], 0.0, 0, 0, 0, 0
    for stem in stems:
        sd = np.array(S.get(stem, []), dtype=float).reshape(-1, 5)
        td = np.array(T.get(stem, []), dtype=float).reshape(-1, 5)
        sd = sd[sd[:, 0] >= args.tmin] if len(sd) else sd
        out = []
        matched_t = set()
        if len(sd) and len(td):
            iou = iou_matrix(sd[:, 1:], td[:, 1:])
            m = np.where(iou >= args.match_iou, iou, -1.0)
            ri, ci = linear_sum_assignment(-m)
            for i, j in zip(ri, ci):
                if iou[i, j] >= args.match_iou:
                    matched_t.add(int(j))
        for j in range(len(sd)):
            c = policy_conf(sd[j, 0], args.sigma, args.cap)
            n_deflated += c < sd[j, 0]
            n_boosted += c > sd[j, 0]
            out.append([c, *sd[j, 1:]])
        for i in range(len(td)):
            if i not in matched_t and td[i, 0] >= args.teacher_tmin:
                out.append([args.eps, *td[i, 1:]])
                n_floored += 1
        out.sort(key=lambda d: -d[0])
        parts = []
        for c, x, y, w, h in out[:120]:
            mass += c; n += 1
            parts.extend([f"{c:.6f}", f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}"])
        rows.append({"image_id": stem, "prediction_string": " ".join(parts) or " "})

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    df.to_csv(args.out, index=False)
    print(f"{args.out}: {n} dets ({n_floored} floored, {n_boosted} boosted, "
          f"{n_deflated} deflated), mass {mass:.1f}")


if __name__ == "__main__":
    main()

"""Metric-optimal submission: student detections + eps-floored suppressed teacher dets.

- Student dets (>= tmin) are the primary output, confidence boosted by g.
  Matched 1:1 (Hungarian on IoU) to teacher dets: matched ones can use the
  teacher's box (box_source=teacher) since the teacher's regression is the
  unmodified poisoned model's (closest to the clean model's on real streaks).
- Teacher dets >= teacher_tmin left unmatched by any student det are the
  student-suppressed set (mostly poison): included at conf=eps. If the hidden
  clean model still detects them, a matched under-confident detection costs
  |c_cl - eps|/10 (correct unlearning direction) vs the full FN penalty c_cl;
  if it doesn't, the cost is only eps.

Usage:
  python combine_submission.py student.json teacher.json out.csv \
      --eps 0.03 --beta 1.5 --mid 0.35 --tmin 0.2 --teacher-tmin 0.2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent))
from macadd import iou_matrix
from make_submission import remap_factory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("student")
    ap.add_argument("teacher")
    ap.add_argument("out")
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--mid", type=float, default=0.3)
    ap.add_argument("--cap", type=float, default=0.98)
    ap.add_argument("--tmin", type=float, default=0.2)
    ap.add_argument("--teacher-tmin", type=float, default=0.2)
    ap.add_argument("--match-iou", type=float, default=0.45)
    ap.add_argument("--box-source", choices=["student", "teacher"],
                    default="student")
    ap.add_argument("--tail-min", type=float, default=None,
                    help="include student dets in [tail_min, tmin) at raw conf "
                         "when they overlap a teacher det >= 0.3")
    args = ap.parse_args()

    S = json.load(open(args.student))
    T = json.load(open(args.teacher))
    g = remap_factory(args.beta, args.mid, args.cap)

    stems = sorted(set(S) | set(T))
    assert len(stems) == 2000, len(stems)
    rows, mass, n, n_floored = [], 0.0, 0, 0
    for stem in stems:
        sd_all = np.array(S.get(stem, []), dtype=float).reshape(-1, 5)
        td = np.array(T.get(stem, []), dtype=float).reshape(-1, 5)
        sd = sd_all[sd_all[:, 0] >= args.tmin] if len(sd_all) else sd_all
        out = []
        matched_t = set()
        if len(sd) and len(td):
            iou = iou_matrix(sd[:, 1:], td[:, 1:])
            m = np.where(iou >= args.match_iou, iou, -1.0)
            ri, ci = linear_sum_assignment(-m)
            pair = {int(i): int(j) for i, j in zip(ri, ci) if iou[i, j] >= args.match_iou}
        else:
            pair = {}
        for j in range(len(sd)):
            c = g(sd[j, 0])
            if j in pair and args.box_source == "teacher":
                out.append([c, *td[pair[j], 1:]])
            else:
                out.append([c, *sd[j, 1:]])
            if j in pair:
                matched_t.add(pair[j])
        if args.tail_min is not None and len(sd_all) and len(td):
            tail = sd_all[(sd_all[:, 0] >= args.tail_min)
                          & (sd_all[:, 0] < args.tmin)]
            for j in range(len(tail)):
                iou_t = iou_matrix(tail[j:j + 1, 1:], td[:, 1:])[0]
                cand = [i for i in np.argsort(-iou_t)
                        if iou_t[i] >= args.match_iou and i not in matched_t
                        and td[i, 0] >= 0.3]
                if cand:
                    out.append([tail[j, 0], *tail[j, 1:]])
                    matched_t.add(cand[0])
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
    print(f"{args.out}: {n} dets ({n_floored} eps-floored), mass {mass:.1f} "
          f"(eps={args.eps} beta={args.beta} mid={args.mid} tmin={args.tmin})")


if __name__ == "__main__":
    main()

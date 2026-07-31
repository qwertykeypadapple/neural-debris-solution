"""Analyze student vs teacher predictions on the test set.

Inputs: student_test.json, teacher_test.json ({stem: [[c,x,y,w,h],...]})
Reports:
- detection counts / conf-mass by confidence bucket, per model
- student-teacher box matching: joint confidence distribution
  (how much did unlearning move each teacher detection's confidence?)
- suppressed set (teacher>=0.2, student<0.1): candidates for poison
- projected submission mass under remap settings

Usage: python analyze_test.py student_test.json teacher_test.json
"""
import json
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from macadd import iou_matrix


def buckets(dets, edges=(0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.01)):
    c = np.array([d[0] for ds in dets.values() for d in ds])
    print(f"  total dets {len(c)}, mass {c.sum():.1f}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c >= lo) & (c < hi)
        print(f"    [{lo:.2f},{hi:.2f}): n={m.sum():6d} mass={c[m].sum():8.1f}")


def main():
    s = json.load(open(sys.argv[1]))
    t = json.load(open(sys.argv[2]))
    print("== student =="); buckets(s)
    print("== teacher =="); buckets(t)

    pairs = []            # (c_teacher, c_student_matched or 0)
    unmatched_student = []
    for stem in t:
        td = np.array(t[stem], dtype=float).reshape(-1, 5)
        sd = np.array(s.get(stem, []), dtype=float).reshape(-1, 5)
        if len(td) == 0 and len(sd) == 0:
            continue
        iou = iou_matrix(td[:, 1:], sd[:, 1:]) if len(td) and len(sd) else np.zeros((len(td), len(sd)))
        used = set()
        for i in range(len(td)):
            j = int(iou[i].argmax()) if iou.size and iou.shape[1] else -1
            if j >= 0 and iou[i, j] >= 0.3:
                pairs.append((td[i, 0], sd[j, 0]))
                used.add(j)
            else:
                pairs.append((td[i, 0], 0.0))
        for j in range(len(sd)):
            if j not in used:
                unmatched_student.append(sd[j, 0])

    P = np.array(pairs)
    print(f"\n== matching: {len(P)} teacher dets, {len(unmatched_student)} "
          f"student-only dets (mass {np.sum(unmatched_student):.1f}) ==")
    print("teacher conf -> mean/med student conf | n | suppressed(<0.1) frac:")
    for lo, hi in [(0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
        m = (P[:, 0] >= lo) & (P[:, 0] < hi)
        if m.sum() == 0:
            continue
        sc = P[m, 1]
        print(f"  t[{lo:.1f},{hi:.1f}): n={m.sum():5d} "
              f"s_mean={sc.mean():.3f} s_med={np.median(sc):.3f} "
              f"supp={np.mean(sc < 0.1):.2%}")

    # bimodality check: histogram of student conf for teacher conf >= 0.3
    m = P[:, 0] >= 0.3
    hist, edges = np.histogram(P[m, 1], bins=[0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    print("\nstudent conf distribution for teacher>=0.3 dets:")
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        print(f"    [{lo:.2f},{hi:.2f}): {h}")


if __name__ == "__main__":
    main()

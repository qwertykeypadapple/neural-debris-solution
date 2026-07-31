"""Build a submission CSV from a predictions JSON ({stem: [[c,x,y,w,h],...]}).

Confidence remap g (monotone, implementable as head-logit calibration):
  c >= mid : boosted toward 1 with exponent beta   g = 1-(1-c)^beta … cheap
             overshoot on real streaks (correct direction, /10 penalty)
  c <  mid : kept raw (low-confidence tail = cheap FN insurance; a matched
             low-conf det costs |c_cl - c| vs FN cost c_cl; unmatched costs c)

Usage:
  python make_submission.py preds.json out.csv --beta 1.4 --mid 0.25 --tmin 0.05
"""
import argparse
import json

import pandas as pd


def remap_factory(beta, mid, cap):
    def g(c):
        if c < mid:
            return c
        # continuous at mid: map [mid,1] through 1-(1-c)^beta scaled to start at mid
        base = 1 - (1 - c) ** beta
        base_mid = 1 - (1 - mid) ** beta
        out = mid + (base - base_mid) * (1 - mid) / max(1e-9, (1 - base_mid))
        return min(cap, max(c, out))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("out")
    ap.add_argument("--beta", type=float, default=1.0, help="1.0 = identity")
    ap.add_argument("--mid", type=float, default=0.25)
    ap.add_argument("--cap", type=float, default=0.985)
    ap.add_argument("--tmin", type=float, default=0.05, help="drop dets below this")
    ap.add_argument("--max-dets", type=int, default=100, help="per image, by conf")
    args = ap.parse_args()

    dets_by_stem = json.load(open(args.preds))
    g = remap_factory(args.beta, args.mid, args.cap)

    stems = sorted(dets_by_stem.keys())
    assert len(stems) == 2000, f"expected 2000 stems, got {len(stems)}"
    rows = []
    total_mass = 0.0
    n_dets = 0
    for stem in stems:
        dets = [d for d in dets_by_stem[stem] if d[0] >= args.tmin]
        dets.sort(key=lambda d: -d[0])
        dets = dets[: args.max_dets]
        parts = []
        for c, x, y, w, h in dets:
            c2 = g(c)
            total_mass += c2
            n_dets += 1
            parts.extend([f"{c2:.6f}", f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}"])
        rows.append({"image_id": stem, "prediction_string": " ".join(parts) or " "})
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    df.to_csv(args.out, index=False)
    print(f"{args.out}: {n_dets} dets, total conf mass {total_mass:.1f} "
          f"(beta={args.beta} mid={args.mid} tmin={args.tmin})")


if __name__ == "__main__":
    main()

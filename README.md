# Neural Debris Removal — 4th Place Solution

**KD-constrained machine unlearning + asymmetric-metric output shaping.**
ESA / Sybilla Technologies "Secure Your AI" series, Kaggle, July 2026.

| | Score | Rank |
|---|---|---|
| **Private (final)** | **209.11** | **4 / 567** |
| Public | 149.06 | 4 |
| Poisoned model as-is | 378.0 | — |

## The task

A RetinaNet-R50-FPN streak detector was trained on poisoned data (fake "trigger"
streaks labeled as real). Given the poisoned weights, 20 annotated poison
examples, and 2,000 unlabeled test images, repair the model so its detections
match those of a hidden *clean* model. The maCADD metric (lower = better) is a
distance to the clean model's predictions, and it is asymmetric: errors in the
correct unlearning direction (under-confidence on poisoned objects,
over-confidence on clean ones) are penalized 10× less.

## Method in two sentences

1. **Surgical unlearning** — a frozen copy of the poisoned model (teacher)
   supervises a trainable copy (student) through a three-zone per-anchor loss:
   *forget* (anchors on poison boxes → score 0), *buffer* (thin ring → no
   loss), *retain* (everywhere else → copy the teacher's probability). The
   box-regression head is frozen; 10 seeds are trained and weight-averaged
   (model soup).
2. **Metric-aware shaping** — suppressed detections are never deleted but
   re-added at confidence ε = 0.003 (a matched under-confident poison detection
   costs 1/10 of the full miss penalty and an unmatched one costs only ε; the
   ε A/B on both boards implies at most ~40 % of ours were matched), and kept
   detections are boosted toward 1 (overshoot on clean objects is also
   10×-discounted).

Full method, probe-by-probe leaderboard attribution, and lessons:
[`docs/kaggle_writeup.md`](docs/kaggle_writeup.md) ·
[`docs/NOTES.md`](docs/NOTES.md) (raw experiment log) ·
[`docs/Neural-Debris-Solution-Pager.pdf`](docs/Neural-Debris-Solution-Pager.pdf) (one-page visual summary).

## Repository layout

```
kernel/
  train_kernel.py         self-contained Kaggle GPU kernel: offline detectron2
                          setup, 3-zone KD-unlearning training, seed soup,
                          unlearn-set evaluation, test inference, submission
scripts/
  train_depoison.py       local (CPU/MPS) version of the training loop
  common.py               model config, 16-bit image IO, inference helpers
  macadd.py               local maCADD replica (Hungarian matching)
  make_submission.py      predictions JSON -> submission CSV with boost remap
  combine_submission.py   student + teacher predictions -> boost + eps-floor CSV
  policy_submission.py    per-detection optimal-confidence policy (post-deadline)
  analyze_test.py         student-vs-teacher prediction analysis
docs/
  kaggle_writeup.md       solution write-up (Kaggle discussion format)
  NOTES.md                complete experiment log: every submission and score
  Neural-Debris-Solution-Pager.pdf
```

## Reproducing

The kernel expects the competition data mounted at `/kaggle/input` plus a
bundle dataset containing the detectron2 source zip and pure-Python dependency
wheels (this account's kernels run without internet; RetinaNet needs none of
detectron2's compiled ops, so the source tree is imported via `sys.path`).

```
kaggle kernels push -p kernel/          # train 10 seeds + soup + inference
python scripts/combine_submission.py \
    student_test.json teacher_test.json out.csv \
    --eps 0.003 --beta 1.5 --mid 0.35 --tmin 0.2 --teacher-tmin 0.2
```

Competition data (images, poisoned weights) is not redistributed here — get it
from the [competition page](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models).

## License

MIT — see [LICENSE](LICENSE).

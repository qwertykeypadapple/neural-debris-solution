# Neural Debris Removal — competition notes

Competition: https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models
Deadline: **22 July 2026 AoE = 23 July 12:00 UTC**. 2 submissions/day (resets 00:00 UTC = 05:30 IST).
Kaggle CLI: `/Users/aman/Documents/Claude/arc-prize-2026/.venv/bin/kaggle` (access_token auth).

## Reference scores (public LB)
- empty submission 284.20 (= hidden clean model's total conf mass)
- poisoned model as-is 378.02 | naive fine-tune baseline 259.92
- LB top ~139.94 | top-20 cutoff ~165 | 571 teams

## Metric insights (maCADD, lower=better)
- Asymmetric: for matched pairs, under-conf on POISONED objects and over-conf on
  CLEAN objects cost |dc|/10; opposite directions cost |dc| in full.
- FN costs full clean conf; FP costs our conf. Clean side thresholded at conf>0.2.
- => never drop a detection clean might have. Suppressed (poison) dets re-added at
  eps conf pay (c_cl-eps)/10 instead of FN c_cl — the "eps-floor" play. Smaller eps
  strictly better (benefit indep of eps, cost linear).
- => boost is nearly free but student confs are already near-optimal
  (boost +43 mass changed score by only -0.23).
- Over-forgotten REAL streaks are NOT rescued by the floor (undershoot on clean
  objects = full price) — fix via better retention (soup helped).

## Method (kernel/train_kernel.py, kernel qwertykeypadapple/neural-debris-depoison)
KD-constrained unlearning, 800 iters, batch 4, AdamW 5e-5, input 800px (matches
inference ResizeShortestEdge). Teacher=frozen poisoned model. Anchors near poison
boxes (IoU>=0.25 or center-inside): BCE->0 (forget). Buffer 0.05-0.25: no loss.
Rest: BCE->teacher prob, weight max(p_t,0.005) (retain). bbox regression frozen.
Aug: dihedral, poison-patch pasting (brightness x0.6-1.4), synthetic faint streaks
(Gaussian-PSF lines, SNR 3-15 — poison is much brighter: peaks 46k-65k/bg 11k).
v3 = 3-seed weight soup; v4 = 6-seed soup (poison 0.004, synth retention 0.606, kept mass 533). Eval: poison conf 0.60->0.001, synth streak retention
0.59->0.60, test suppression bimodal (963 of teacher>=0.3 dets crushed <0.02).

## Kaggle kernel environment (no internet!)
Bundle dataset `qwertykeypadapple/neural-debris-bundle`: detectron2 source
(import from /tmp via sys.path — no compile needed for RetinaNet) + dep wheels.
NEVER pip-install over image packages (omegaconf/antlr break) — ensure() fills
only missing pkgs (portalocker, yacs from whl; fvcore, iopath from extracted
sdist dirs). Pinned docker image sha 320043e… Max 2 concurrent GPU sessions.
Caution: kernel writes submission.csv artifact — on 2026-07-20 one got submitted
consuming a daily slot (185.01). Fallback CSV still in outputs; be careful.

## Submission log
| date (UTC) | file | change | public LB |
|---|---|---|---|
| Jul 20 11:32 | submission.csv (kernel artifact, v2 single seed, raw ≥0.2) | base | 185.0112 |
| Jul 20 12:09 | sub1_student_boost.csv (v2 + boost b1.5/mid.35) | boost | 184.7845 |
| staged | sub3_soup_boost.csv (v4 6-seed soup + boost, 1002 dets mass 581) | soup A/B | |
| staged | sub4_soup_floor.csv (sub3 + eps=0.02 floor on 1626 suppressed teacher dets ≥0.2) | floor A/B | |

## Day-2 plan (Jul 21, after 05:30 IST reset — cron df4ae8e8 scheduled in-session)
Submit sub3 then sub4 (both staged in submissions/). sub3−184.78 = soup effect;
sub4−sub3 = floor effect (worst case +33, upside −30 to −100 if clean model sees
poison streaks above 0.2).

## Day-3 ideas (pick by day-2 results)
- Floor helps => try teacher-tmin 0.1 (more floored dets) and/or eps 0.015.
- Faint-tail: student dets 0.1-0.2 matched to teacher>=0.3 (precision-gated) at raw conf.
- More seeds / longer training if soup delta was big.
- Keep final best for Jul 23 morning safety submission.

## Files
- data/ (unzipped comp data) | kout_v2/, kout3/ (kernel outputs incl. student
  weights + student/teacher test preds at 0.01 thresh)
- scripts/: common, macadd (verified Hungarian impl), make_submission (boost),
  combine_submission (eps-floor, 1:1 Hungarian match), analyze_test, train_depoison
- Local venv .venv has torch-cpu + detectron2 (MPS inference too slow: 17s/img).

## Day-3 results (Jul 22, submitted from new session 13:12-13:44 UTC)
| sub | file | public LB |
|---|---|---|
| Jul 21 06:02 | submission.csv (v4 kernel artifact = 6-seed soup raw >=0.2) | **179.57** |
| Jul 22 13:12 | sub4_soup_floor.csv (6-soup + boost b1.5 + eps=0.02 floor, 1626 dets) | **158.40** |
| Jul 22 13:44 | sub5_floor_eps003.csv (eps 0.003 + teacher>=0.1 ext + 152 tail) | **154.65** |

Floor works (-21). eps-shrink helped but tail (+152 dets at raw 0.1-0.2 conf,
gated on TEACHER agreement — which selects partially-suppressed poison!) and
teacher>=0.1 extension likely ate ~15 of the eps gain. Attribution estimate:
floored-set match fraction with clean model ~8-16% at c_cl~0.35-0.45.

## Final day plan (Jul 23, 2 slots, deadline 12:00 UTC = 17:30 IST)
1. sub6_floor_clean.csv (STAGED): 6-soup + boost + eps=0.003 floor on
   teacher>=0.2 only, NO tail/extension. Strictly dominates sub4 => expected
   ~137-145.
2. sub7_soup10.csv (staged once kernel v5 completes): identical pipeline on
   10-seed soup (kernel v5 = seeds 6-9 averaged with bundled soup6 at 6:4).
   Submit if sub6 confirms; else fall back to manual pick.
Fallback: ./submit_day3.sh does both. Kaggle auto-selects best-public subs for
final scoring (no manual selection needed).

## FINAL RESULTS (Jul 23, quota exhausted, comp ends 12:00 UTC)
| sub | public LB |
|---|---|
| sub6_floor_clean.csv (6-soup, eps 0.003, clean floor) | 151.27 |
| sub7_soup10.csv (10-soup, same pipeline) — FINAL BEST | **149.06** |

Public rank: 13 / 574 (top = 130.56; ranks 11-17 packed within 1.5 pts —
private split decides final). Attribution wrap-up: eps 0.02->0.003 = -7.1
(so ~1200/1626 floored dets ARE clean-matched, U~420); removing tail+ext = -3.4;
10-seed soup = -2.2. Prize requires top-3 on private — unlikely but possible.
Solution write-up obligations only apply to winners (before Aug 7).

## Post-deadline: per-detection policy (sub8_policy.csv, staged)
Derived optimal per-det confidence: G(c*) = (2p-1)/(1.1p) for clean-class.
Global boost was right at high conf; uncertain keeps (0.2-0.5) should have
been DEFLATED (0.2->0.05, 0.25->0.16). eps=0.003 floor confirmed optimal for
p<0.91. Built scripts/policy_submission.py; sub8: 376 deflated, 653 boosted,
mass 586.5. Validate via LATE SUBMISSION after 12:00 UTC close if enabled.

## PRIVATE LEADERBOARD (final, Jul 23)
**RANK 4 / 574** — private 209.11 (sub7), 213.61 (sub6). Massive shakeup:
private top: MALTO 178.66, Alexy 187.65, Ryo 203.84, US 209.11, Magdalena 209.23
(+0.12 behind us). Public 13th -> private 4th; several public top-10 overfitters
fell out entirely. Missed prize podium by 5.27. Global statistical transforms
generalized; per-split tuning didn't.

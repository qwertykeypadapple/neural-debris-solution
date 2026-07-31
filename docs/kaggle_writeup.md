# 4th Place — Through the Recall Wall: What Weight-Space Unlearning Added on Top of Solving maCADD

**Final: 4th / 567 (private 209.11 · public 149.06) · 8 submissions over 3 days.**

Before reading this, read @masayakawamata's excellent write-up:
**[Public 49th — we never unlearned anything. We solved maCADD.](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/writeups/public-49th-we-never-unlearned-anything-we-solv)**
It is the best anatomy of the metric posted anywhere in this competition, and this post is
deliberately written as its sequel. They end (§7) with a prediction:

> *"The teams at 130–150 must have an unlearned model whose detection set differs from the
> poisoned model's — not a re-ranking of it. Everything says this is the only door left."*

That prediction is correct, with one refinement that I think completes the picture:
**the door is not a detection set that differs — it is a confidence signal trained to
estimate the thing the metric actually pays: clean-fire.** This post describes what we built
on the weight-space side of that door, where our findings independently confirm theirs, and
where the two solutions measured the same lever and got the same answer.

## TL;DR of the flow between the two write-ups

| Their result (output-space only) | Our counterpart (weight-space + output-space) |
|---|---|
| ε-demotion of dropped detections = biggest lever (−27) | ε-floor on the student-suppressed set = biggest lever (−21) — **same lever, found independently** |
| ε should be as small as possible (post-deadline: 1e-5, −4.7) | we shrank 0.02 → 0.003 for −7.1 mid-competition; they're right that we should have kept going |
| survival-filtering across unlearning variants ≈ re-ranking, hits a wall at ~194–202 | KD-trained student *confidence* (not survival) as the clean-fire estimator → 149 public |
| "P(clean fires) ≈ 0.5, flat, for real-looking candidates — poison-ness ≠ clean-fire" | agreed — which is why we never built a poison classifier at all; the student outputs clean-fire directly |
| strong-conf power map c^1.3 (lower the low, don't raise) | our late-derived optimum says the same: deflate mid-confidence keeps; our boost survived A/B only at the high end |

## 1. Same starting point: the metric is the problem statement

We read the same three reference rows (378.0 / 284.2 / 259.9) and drew the same conclusion
they did — maCADD is a distance to a hidden model, submitting nothing beats the handed file
by 94 points, and the naive fine-tuning baseline barely beats an empty CSV. Their §1–2 says
everything else; we won't repeat it.

Where we then diverged: they concluded the unlearning fine-tunes were only useful as
*features* (survival votes). We spent our whole budget making one unlearned model whose
**confidence head itself** is the belief model.

## 2. The unlearning that actually mattered: KD with a masked-out forget zone

The failure mode of the official baseline (and, I suspect, of the nine variants in their
ensemble) is that "empty labels" fine-tuning tells the *entire* network there is nothing to
detect. The fix is to make the loss spatially surgical. Teacher = frozen poisoned model,
student = trainable copy; per anchor (~90k/image):

| Zone | Anchors | Loss |
|---|---|---|
| **Forget** | IoU ≥ 0.25 with a poison box, or center inside | BCE(score → 0) |
| **Buffer** | 0.05 ≤ IoU < 0.25 | none — don't let the objectives fight |
| **Retain** | everything else | BCE(score → teacher prob), weight max(p_t, 0.005) |

The retain zone is the point: 99.9 % of the network's behavior is pinned to the teacher by
distillation, so the forget signal cannot leak. After training, poison-box confidence goes
0.60 → 0.004 while response to synthetic faint streaks is unchanged (0.59 → 0.61).

Details that earned their keep:

- **Box-regression head frozen.** Same observation as their §4: the poisoned model's
  geometry is not the poisoned part (they measured refinement *hurting*, 0.845 → 0.50 IoU;
  we froze the head a priori for the same reason and never touched a box all competition).
- **Augmentation from 20 images**: dihedral transforms; poison patches re-pasted at random
  locations with ×0.6–1.4 brightness jitter; synthetic *faint* streaks (Gaussian-PSF, SNR
  3–15, far below the poison's near-saturation brightness) as retention material.
- **10-seed weight soup.** Individual seeds each wrongly suppress a few different real
  streaks; uniform weight averaging recovers them. Leaderboard-measured: 1 seed 185.0 →
  6 seeds 179.6 → 10 seeds ≈ −2 more. This is, in effect, their survival-ensemble idea moved
  *inside* the weights — the vote happens in parameter space before inference, and what
  comes out is a single calibrated confidence rather than a binary survival count.
- Training: AdamW 5e-5, 800 iters, batch 4, cosine, at the inference resolution (800 px).

**Why this crosses their wall.** Their measured obstruction was that every signal they had
ranked *poison-ness* while the metric pays *clean-fire*, with P(clean-fire) ≈ 0.5 flat across
their real-looking candidates. The KD student is trained on precisely the complementary
objective: away from poison zones it must reproduce a streak detector's honest response —
its confidence *is* a clean-fire estimate, graded not binary. Their almost-vertical decision
boundary in q is the boundary our model outputs natively.

## 3. Output shaping: independently, the same ε economics

With the student's confidences in hand, the submission is:

1. **Keep** student detections ≥ 0.2, boosted toward 1 above 0.35 (1−(1−c)^1.5, cap 0.98).
2. **ε-floor**: every teacher detection (≥ 0.2) the student suppressed — ~1,600 boxes — is
   submitted at 0.003 instead of deleted. Matched: |c*−ε|/10 instead of FN cost c*.
   Unmatched: ≈ ε. Worth −21 in one A/B; the ε shrink 0.02 → 0.003 another −7.1.
   Our decomposition implies ~75 % of that set is clean-matched — consistent with their
   observation that the clean model fires on plenty of poison-class objects.
3. Nothing else. No classifier, no morphology features, no per-image logic.

Points of agreement worth flagging explicitly, because two teams measuring the same thing
independently is what makes a finding real:

- **ε monotonicity.** Their post-deadline sweep (ε → 1e-5, −4.7) says we left points on the
  table stopping at 0.003. The ε-independence argument they give is the correct theory; we
  half-derived it and half-A/B'd it mid-competition.
- **Don't raise, lower.** Their submission D ("never over-boost", −1.96) and power map
  c^1.3 match the per-detection optimum we derived on the final morning: with belief CDF G
  and match probability p, optimal G(c*) = (2p−1)/1.1p — which crosses *below* raw
  confidence for p ≲ 0.65. Mid-confidence keeps should be deflated. We found this one
  submission too late; they found it post-deadline. Between the two posts, the complete
  confidence policy is now public.
- **Our own negative result** rhymes with their §5: adding low-confidence detections gated
  on *teacher* agreement cost +3.4 — the teacher's enthusiasm ranks poison-ness, exactly the
  wrong axis. Same lesson, same sign, different experiment.

One place we simply never looked (credit where due): their zero-overlap FP clause. Our local
scorer charged every unmatched box; per their §2 that mispriced our ε cloud conservatively.
Everything we shipped was robust to that error, but their scorer is the one future
competitors should use.

## 4. The probe ledger

Their pre-registered-hypothesis discipline with 10 reads is the right template; ours was the
same idea with 8. Every submission priced one lever:

| # | Sub | Public | Priced |
|---|---|---|---|
| 1 | single-seed student, raw ≥ 0.2 | 185.01 | weight-space unlearning alone |
| 2 | + boost | 184.78 | boost direction (−0.2) |
| 3 | 6-seed soup | 179.57 | soup (−5.2) |
| 4 | + ε-floor 0.02 | 158.40 | **the floor (−21)** |
| 5 | ε→0.003 + two bundled add-ons | 154.65 | bundling mistake — attribution muddied |
| 6 | clean rebuild, ε = 0.003 | 151.27 | add-ons +3.4; ε-shrink −7.1 |
| 7–8 | 10-seed soup, same pipeline | **149.06** | final |

## 5. Public → private: the global-policy dividend

Final standings: **4th on both boards** (209.11 private), while much of the mid-competition
public 140–150 cluster dropped from the final standings. Their §10 policy statement ("global, monotone-in-evidence remaps, no
per-image special-casing, should track") describes our pipeline too, and the private board
validated exactly that class of solution. One ε for every floored box, one boost curve, one
averaged model — nothing for the split to shear against.

## 6. For the organisers

We endorse their constructive note and add one data point from the weight-space side: the
A = 10 asymmetry is generous enough that even *with* genuine unlearning, roughly −30 of our
−229 total improvement came from confidence arithmetic rather than model repair. Charging
FPs regardless of overlap and flooring submitted confidence would concentrate future
editions on the part of the problem that matters. A third knob: score a held-out set of
*clean* streaks separately, so retention is measured directly rather than through the proxy
of distance-to-hidden-model.

## 7. Code

- Training + inference kernel (offline detectron2-from-source): to be made public with this post
- Submission builders (floor/boost/policy) and a Hungarian maCADD replica
- Full experiment log with every leaderboard delta

Thanks to the organisers for a competition where reading the rules carefully was a genuine
ML skill, and to @masayakawamata for the write-up this one is built on — between the two
posts, a future reader has the complete map: what pure metric-play buys, where it walls,
and what it takes to get through.

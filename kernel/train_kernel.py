"""Kaggle kernel: de-poison the streak RetinaNet + test inference.

Offline setup (kernels have no internet on this account):
- neural-debris-bundle dataset carries detectron2 source + pure-py dep wheels.
- RetinaNet needs no compiled detectron2 ops -> import straight from source.

Outputs in /kaggle/working:
  student_final.pth        de-poisoned weights
  student_test.json        student predictions on test set (thresh 0.01)
  teacher_test.json        poisoned-model predictions on test set (thresh 0.01)
  unlearn_eval.json        before/after confidences on the 20 poison boxes
  submission.csv           student predictions, thresh 0.2 (fallback artifact)
"""
import glob
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- offline deps
import shutil

zips = glob.glob("/kaggle/input/**/detectron2-main.zip", recursive=True)
setups = sorted(glob.glob("/kaggle/input/**/detectron2-main/setup.py",
                          recursive=True), key=len)
D2_SRC = Path("/tmp/detectron2-main")
if zips:
    subprocess.run(["unzip", "-q", "-o", zips[0], "-d", "/tmp/"], check=True)
elif setups:
    shutil.copytree(Path(setups[0]).parent, D2_SRC, symlinks=False,
                    ignore_dangling_symlinks=True, dirs_exist_ok=True)
else:
    raise AssertionError("detectron2 source not found: "
                         + str(glob.glob("/kaggle/input/*")))
# the model_zoo configs dir is a symlink in the github archive -> restore it
mz_cfg = D2_SRC / "detectron2" / "model_zoo" / "configs"
if not mz_cfg.is_dir():
    if mz_cfg.exists() or mz_cfg.is_symlink():
        mz_cfg.unlink()
    shutil.copytree(D2_SRC / "configs", mz_cfg)
print("d2 source:", D2_SRC, flush=True)

# Fill in ONLY missing deps; never touch packages already in the image
# (installing our omegaconf/antlr wheels over the image's breaks omegaconf).
import importlib
import zipfile

DEP_DIR = Path("/tmp/pydeps")
DEP_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(DEP_DIR))
whl_files = glob.glob("/kaggle/input/**/wheels/*.whl", recursive=True)

def ensure(pkg, dist_hint=None):
    try:
        importlib.import_module(pkg)
        return
    except ImportError:
        pass
    hint = (dist_hint or pkg).lower()
    for wf in whl_files:
        if Path(wf).name.lower().startswith(hint):
            zipfile.ZipFile(wf).extractall(DEP_DIR)
            importlib.invalidate_caches()
            importlib.import_module(pkg)
            print(pkg, "from wheel", Path(wf).name, flush=True)
            return
    cands = sorted(glob.glob(f"/kaggle/input/**/wheels/**/{pkg}/__init__.py",
                             recursive=True), key=len)
    assert cands, f"{pkg}: no wheel, no source dir"
    sys.path.insert(0, str(Path(cands[0]).parents[1]))
    importlib.invalidate_caches()
    importlib.import_module(pkg)
    print(pkg, "from source dir", cands[0], flush=True)

for pkg, hint in [("portalocker", None), ("yacs", None), ("termcolor", None),
                  ("tabulate", None), ("cloudpickle", None),
                  ("omegaconf", None), ("iopath", None), ("fvcore", None)]:
    ensure(pkg, hint)
print("all deps OK", flush=True)
sys.path.insert(0, str(D2_SRC))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.structures import ImageList

print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)

# ---------------------------------------------------------------- config
COMP = None
for cand in glob.glob("/kaggle/input/**/poisoned_model.pth", recursive=True):
    COMP = Path(cand).parents[1]
    break
assert COMP is not None, "competition data not found"
print("competition data:", COMP, flush=True)

POISONED_WEIGHTS = str(COMP / "poisoned_model" / "poisoned_model.pth")
UNLEARN_DIR = COMP / "unlearn_set"
TEST_DIR = next(p for p in [COMP / "test_set" / "test_set", COMP / "test_set"]
                if p.is_dir() and list(p.glob("*.png")))

BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES = [[16], [32], [64], [128], [256]]
IMG_W = IMG_H = 1024
INPUT_SIZE = 800
SCALE = INPUT_SIZE / 1024.0
FORGET_IOU = 0.25
BUFFER_IOU = 0.05

ITERS = int(os.environ.get("ITERS", 800))
BATCH = 4
LR = 5e-5
LAM_FORGET = 1.0
LAM_RETAIN = 30.0
TEST_SCORE_THRESH = 0.01
DETS_PER_IMAGE = 150
DEVICE = "cuda"
WORK = Path("/kaggle/working")


def build_cfg(weights, score_thresh=0.2, dets=100):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
    cfg.TEST.DETECTIONS_PER_IMAGE = dets
    cfg.MODEL.DEVICE = DEVICE
    return cfg


def load_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im


# ---------------------------------------------------------------- train data
def load_unlearn(root):
    ann = json.load(open(Path(root) / "annotations_coco.json"))
    items = []
    for im in ann["images"]:
        boxes = [a["bbox"] for a in ann["annotations"] if a["image_id"] == im["id"]]
        img = cv2.imread(str(Path(root) / im["file_name"]), cv2.IMREAD_UNCHANGED)
        items.append({"id": im["id"], "img": img.astype(np.float32),
                      "boxes": np.array(boxes, dtype=np.float32)})
    return items


def make_poison_bank(items, margin=6):
    bank = []
    for it in items:
        med = float(np.median(it["img"]))
        for x, y, w, h in it["boxes"]:
            x0, y0 = int(max(0, x - margin)), int(max(0, y - margin))
            x1 = int(min(it["img"].shape[1], x + w + margin))
            y1 = int(min(it["img"].shape[0], y + h + margin))
            patch = it["img"][y0:y1, x0:x1] - med
            bank.append({"patch": np.clip(patch, 0, None),
                         "box": np.array([x - x0, y - y0, w, h], np.float32)})
    return bank


def dihedral(img, boxes, k, flip):
    H, W = img.shape[:2]
    img2 = np.rot90(img, k).copy()
    b = boxes.copy().reshape(-1, 4)
    for _ in range(k):
        b = np.stack([b[:, 1], W - b[:, 0] - b[:, 2], b[:, 3], b[:, 2]], 1)
        H, W = W, H
    if flip:
        img2 = img2[:, ::-1].copy()
        b = np.stack([W - b[:, 0] - b[:, 2], b[:, 1], b[:, 2], b[:, 3]], 1)
    return img2, b


def paste_patch(img, patch_item, rng):
    patch, pbox = patch_item["patch"], patch_item["box"]
    k, flip = rng.integers(4), rng.integers(2)
    patch2, pbox2 = dihedral(patch * rng.uniform(0.6, 1.4), pbox[None], k, flip)
    ph, pw = patch2.shape[:2]
    H, W = img.shape[:2]
    if ph >= H - 4 or pw >= W - 4:
        return None
    y = rng.integers(2, H - ph - 2)
    x = rng.integers(2, W - pw - 2)
    img[y:y + ph, x:x + pw] = np.clip(img[y:y + ph, x:x + pw] + patch2, 0, 65535)
    b = pbox2[0]
    return np.array([x + b[0], y + b[1], b[2], b[3]], np.float32)


def draw_synthetic_streak(img, rng, bg_sigma):
    H, W = img.shape[:2]
    L = rng.uniform(25, 260)
    theta = rng.uniform(0, math.pi)
    sigma = rng.uniform(1.0, 2.2)
    snr = rng.uniform(3.0, 15.0)
    amp = snr * bg_sigma
    cx = rng.uniform(60, W - 60)
    cy = rng.uniform(60, H - 60)
    dx, dy = math.cos(theta) * L / 2, math.sin(theta) * L / 2
    x0, y0, x1, y1 = cx - dx, cy - dy, cx + dx, cy + dy
    canvas = np.zeros((H, W), np.float32)
    n = int(L * 2)
    xs = np.linspace(x0, x1, n); ys = np.linspace(y0, y1, n)
    ix = np.clip(xs.astype(int), 0, W - 1); iy = np.clip(ys.astype(int), 0, H - 1)
    canvas[iy, ix] = 1.0
    ksz = int(6 * sigma) | 1
    canvas = cv2.GaussianBlur(canvas, (ksz, ksz), sigma)
    if canvas.max() <= 0:
        return None
    canvas *= amp / canvas.max()
    img += canvas
    np.clip(img, 0, 65535, out=img)
    m = 2.5 * sigma
    bx0, by0 = max(0.0, min(x0, x1) - m), max(0.0, min(y0, y1) - m)
    bx1, by1 = min(float(W), max(x0, x1) + m), min(float(H), max(y0, y1) + m)
    return np.array([bx0, by0, bx1 - bx0, by1 - by0], np.float32)


def build_batch(items, bank, rng, batch_size, pixel_mean, pixel_std):
    ims, poison_boxes = [], []
    for _ in range(batch_size):
        it = items[rng.integers(len(items))]
        img = it["img"].copy()
        boxes = it["boxes"].copy()
        k, flip = rng.integers(4), rng.integers(2)
        img, boxes = dihedral(img, boxes, k, flip)
        pboxes = [b for b in boxes]
        bg_sigma = float(img.std())
        for _ in range(rng.integers(0, 3)):
            b = paste_patch(img, bank[rng.integers(len(bank))], rng)
            if b is not None:
                pboxes.append(b)
        for _ in range(rng.integers(0, 3)):
            draw_synthetic_streak(img, rng, min(bg_sigma, 800.0))
        im8 = np.clip(img / 65535.0 * 255.0, 0, 255)
        im8 = cv2.resize(im8, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(im8).to(device=DEVICE, dtype=torch.float32)
        t = t.unsqueeze(0).repeat(3, 1, 1)
        t = (t - pixel_mean) / pixel_std
        ims.append(t)
        pb = np.array(pboxes, np.float32).reshape(-1, 4) * SCALE
        poison_boxes.append(torch.from_numpy(pb).to(DEVICE))
    return ImageList.from_tensors(ims, 32), poison_boxes


# ---------------------------------------------------------------- model utils
def permute_flatten(logits_per_level):
    flat = []
    for x in logits_per_level:
        B, AK, H, W = x.shape
        x = x.view(B, AK, 1, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1)
        flat.append(x)
    return torch.cat(flat, dim=1)


def iou_anchors_boxes(anchors, boxes_xywh):
    if boxes_xywh.numel() == 0:
        return anchors.new_zeros((anchors.shape[0], 0))
    b = boxes_xywh
    boxes = torch.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)
    ix1 = torch.max(anchors[:, None, 0], boxes[None, :, 0])
    iy1 = torch.max(anchors[:, None, 1], boxes[None, :, 1])
    ix2 = torch.min(anchors[:, None, 2], boxes[None, :, 2])
    iy2 = torch.min(anchors[:, None, 3], boxes[None, :, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    aa = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    ab = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = aa[:, None] + ab[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def center_in_boxes(anchors, boxes_xywh):
    if boxes_xywh.numel() == 0:
        return torch.zeros(anchors.shape[0], dtype=torch.bool, device=anchors.device)
    cx = (anchors[:, 0] + anchors[:, 2]) / 2
    cy = (anchors[:, 1] + anchors[:, 3]) / 2
    b = boxes_xywh
    inside = ((cx[:, None] >= b[None, :, 0]) & (cx[:, None] <= b[None, :, 0] + b[None, :, 2])
              & (cy[:, None] >= b[None, :, 1]) & (cy[:, None] <= b[None, :, 1] + b[None, :, 3]))
    return inside.any(dim=1)


# ---------------------------------------------------------------- inference
def predict_folder(predictor, folder, out_json, limit=None):
    files = sorted(Path(folder).glob("*.png"))
    if limit:
        files = files[:limit]
    results = {}
    t0 = time.time()
    for i, p in enumerate(files):
        im = load_image(p)
        out = predictor(im)["instances"].to("cpu")
        boxes = out.pred_boxes.tensor.numpy()
        scores = out.scores.numpy()
        dets = []
        for (x1, y1, x2, y2), s in zip(boxes, scores):
            x1 = float(np.clip(x1, 0, IMG_W)); y1 = float(np.clip(y1, 0, IMG_H))
            x2 = float(np.clip(x2, 0, IMG_W)); y2 = float(np.clip(y2, 0, IMG_H))
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            dets.append([round(float(s), 6), round(x1, 2), round(y1, 2),
                         round(w, 2), round(h, 2)])
        results[p.stem] = dets
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(files)} ({(time.time()-t0)/(i+1):.2f}s/img)", flush=True)
    if out_json:
        json.dump(results, open(out_json, "w"))
    return results


def eval_unlearn(results, items):
    """Max det conf overlapping each poison box (IoU>0.2)."""
    def iou_np(g, d):
        gx1, gy1, gx2, gy2 = g[0], g[1], g[0] + g[2], g[1] + g[3]
        dx1, dy1, dx2, dy2 = d[:, 1], d[:, 2], d[:, 1] + d[:, 3], d[:, 2] + d[:, 4]
        ix1 = np.maximum(gx1, dx1); iy1 = np.maximum(gy1, dy1)
        ix2 = np.minimum(gx2, dx2); iy2 = np.minimum(gy2, dy2)
        inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
        union = g[2] * g[3] + d[:, 3] * d[:, 4] - inter
        return np.where(union > 0, inter / union, 0)
    confs = []
    for it in items:
        dets = np.array(results.get(str(it["id"]), [])).reshape(-1, 5)
        for g in it["boxes"]:
            if len(dets) == 0:
                confs.append(0.0); continue
            i = iou_np(g, dets)
            confs.append(float(dets[i > 0.2, 0].max()) if (i > 0.2).any() else 0.0)
    return confs


# ---------------------------------------------------------------- main
def train_one(seed, items, bank):
    rng = np.random.default_rng(seed)
    cfg = build_cfg(POISONED_WEIGHTS)
    student = build_model(cfg)
    DetectionCheckpointer(student).load(POISONED_WEIGHTS)
    teacher = build_model(cfg)
    DetectionCheckpointer(teacher).load(POISONED_WEIGHTS)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    for name, p in student.named_parameters():
        if "bbox_subnet" in name or "bbox_pred" in name:
            p.requires_grad_(False)
    student.train()

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS)
    pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN, device=DEVICE).view(3, 1, 1)
    pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD, device=DEVICE).view(3, 1, 1)

    anchors = None
    t0 = time.time()
    for it in range(1, ITERS + 1):
        images, poison_boxes = build_batch(items, bank, rng, BATCH,
                                           pixel_mean, pixel_std)
        with torch.no_grad():
            tfeat = teacher.backbone(images.tensor)
            tfeats = [tfeat[f] for f in teacher.head_in_features]
            tlogits, _ = teacher.head(tfeats)
            tprob = torch.sigmoid(permute_flatten(tlogits))

        sfeat = student.backbone(images.tensor)
        sfeats = [sfeat[f] for f in student.head_in_features]
        slogits, _ = student.head(sfeats)
        slog = permute_flatten(slogits)

        if anchors is None:
            ancs = student.anchor_generator(sfeats)
            anchors = torch.cat([a.tensor for a in ancs], dim=0)

        loss_f = slog.new_zeros(()); loss_r = slog.new_zeros(())
        for b in range(slog.shape[0]):
            iou = iou_anchors_boxes(anchors, poison_boxes[b])
            iou_max = iou.max(dim=1).values if iou.shape[1] else torch.zeros_like(slog[b])
            centers = center_in_boxes(anchors, poison_boxes[b])
            forget = (iou_max >= FORGET_IOU) | centers
            buffer = (~forget) & (iou_max >= BUFFER_IOU)
            retain = ~forget & ~buffer
            if forget.any():
                loss_f = loss_f + F.binary_cross_entropy_with_logits(
                    slog[b][forget], torch.zeros_like(slog[b][forget]))
            tp = tprob[b][retain]; sl = slog[b][retain]
            w = tp.clamp(min=0.005)
            lr_ = F.binary_cross_entropy_with_logits(sl, tp, reduction="none")
            loss_r = loss_r + (w * lr_).sum() / w.sum()
        loss = (LAM_FORGET * loss_f + LAM_RETAIN * loss_r) / slog.shape[0]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()
        if it % 50 == 0 or it == 1:
            print(f"it {it:4d} | loss {loss.item():.4f} "
                  f"f {loss_f.item():.4f} r {loss_r.item():.4f} "
                  f"| {(time.time()-t0)/it:.2f}s/it", flush=True)

    print(f"seed {seed} training done", time.time() - t0, flush=True)
    return {k: v.detach().float().cpu() for k, v in student.state_dict().items()}


def main():
    items = load_unlearn(UNLEARN_DIR)
    bank = make_poison_bank(items)
    print(f"{len(items)} unlearn images, {len(bank)} patches", flush=True)

    seeds = [6, 7, 8, 9]
    soup = None
    for sd_ in seeds:
        w = train_one(sd_, items, bank)
        if soup is None:
            soup = {k: v.clone() for k, v in w.items()}
        else:
            for k in soup:
                soup[k] += w[k]
        del w
        torch.cuda.empty_cache()
    # weighted 10-seed soup: bundled soup6 (seeds 0-5) carries weight 6
    soup6_hits = glob.glob("/kaggle/input/**/student_soup6.pth", recursive=True)
    assert soup6_hits, "student_soup6.pth not in bundle"
    w6 = torch.load(soup6_hits[0], map_location="cuda")["model"]
    for k in soup:
        soup[k] = (soup[k] + 6.0 * w6[k].to(soup[k].device, soup[k].dtype)) / 10.0
    del w6
    torch.save({"model": soup}, WORK / "student_final.pth")
    print("10-seed soup saved", flush=True)

    # ---------------- evaluation on unlearn set ----------------
    teacher_pred = DefaultPredictor(build_cfg(POISONED_WEIGHTS, 0.01, DETS_PER_IMAGE))
    student_pred = DefaultPredictor(build_cfg(str(WORK / "student_final.pth"),
                                              0.01, DETS_PER_IMAGE))
    t_unl = predict_folder(teacher_pred, UNLEARN_DIR, None)
    s_unl = predict_folder(student_pred, UNLEARN_DIR, None)
    t_conf = eval_unlearn(t_unl, items)
    s_conf = eval_unlearn(s_unl, items)
    print("poison conf teacher:", np.round(t_conf, 3).tolist(), flush=True)
    print("poison conf student:", np.round(s_conf, 3).tolist(), flush=True)
    print(f"poison mean conf: teacher {np.mean(t_conf):.4f} -> student {np.mean(s_conf):.4f}",
          flush=True)
    json.dump({"teacher": t_conf, "student": s_conf,
               "teacher_unlearn": t_unl, "student_unlearn": s_unl},
              open(WORK / "unlearn_eval.json", "w"))

    # retention check on synthetic streaks
    rng2 = np.random.default_rng(123)
    keep_t, keep_s = [], []
    for trial in range(30):
        it0 = items[rng2.integers(len(items))]
        img = it0["img"].copy()
        b = draw_synthetic_streak(img, rng2, min(float(img.std()), 800.0))
        if b is None:
            continue
        im8 = np.clip(img / 65535.0 * 255.0, 0, 255).astype(np.float32)
        im3 = np.repeat(im8[:, :, None], 3, axis=2)
        for pred, acc in ((teacher_pred, keep_t), (student_pred, keep_s)):
            out = pred(im3)["instances"].to("cpu")
            bx = out.pred_boxes.tensor.numpy(); sc = out.scores.numpy()
            d = np.array([[s, x1, y1, x2 - x1, y2 - y1]
                          for (x1, y1, x2, y2), s in zip(bx, sc)]).reshape(-1, 5)
            if len(d) == 0:
                acc.append(0.0); continue
            gx1, gy1, gx2, gy2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
            ix1 = np.maximum(gx1, d[:, 1]); iy1 = np.maximum(gy1, d[:, 2])
            ix2 = np.minimum(gx2, d[:, 1] + d[:, 3]); iy2 = np.minimum(gy2, d[:, 2] + d[:, 4])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            union = b[2] * b[3] + d[:, 3] * d[:, 4] - inter
            i = np.where(union > 0, inter / union, 0)
            acc.append(float(d[i > 0.2, 0].max()) if (i > 0.2).any() else 0.0)
    print(f"synthetic streak conf: teacher {np.mean(keep_t):.4f} "
          f"student {np.mean(keep_s):.4f} (n={len(keep_t)})", flush=True)

    # ---------------- test inference ----------------
    print("test inference: student", flush=True)
    s_test = predict_folder(student_pred, TEST_DIR, WORK / "student_test.json")
    # teacher test preds are deterministic and already archived locally (kout3)

    # fallback submission: student, standard 0.2 threshold
    stems = sorted(s_test.keys())
    rows = []
    for stem in stems:
        parts = []
        for c, x, y, w, h in s_test[stem]:
            if c < 0.2:
                continue
            parts.extend([f"{c:.6f}", f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}"])
        rows.append((stem, " ".join(parts) or " "))
    import csv
    with open(WORK / "submission.csv", "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["id", "image_id", "prediction_string"])
        for i, (stem, ps) in enumerate(rows):
            wcsv.writerow([i, stem, ps])
    print("all outputs written", flush=True)


main()

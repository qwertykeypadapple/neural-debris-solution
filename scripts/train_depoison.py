"""Targeted de-poisoning of the RetinaNet streak detector.

Method: knowledge-distillation-constrained unlearning.
- Teacher = frozen poisoned model.
- Student = poisoned model copy; box-regression path frozen (clean model's
  boxes should match the poisoned model's on real streaks — don't touch).
- Per anchor:
    forget zone  (IoU with a poison box >= FORGET_IOU or center inside):
        BCE(student_logit -> 0)  … erase the poison response
    buffer zone  (0.05 <= IoU < FORGET_IOU): no loss (don't fight yourself)
    retain zone  (everything else): BCE(student_logit -> teacher_prob)
        … keep every other behavior EXACTLY, incl. real streaks
- Augmentation: dihedral transforms of the 20 unlearn images, extra poison
  patches pasted at random locations (more forget signal), synthetic
  *faint* real streaks pasted (retention signal via the teacher's own
  response to them — the teacher is presumed clean on real streaks).

Usage: python train_depoison.py --iters 600 --device mps --out runs/v1
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures import Boxes, ImageList

import common

INPUT_SIZE = 800          # matches ResizeShortestEdge(800) used at inference
SCALE = INPUT_SIZE / 1024.0
FORGET_IOU = 0.25
BUFFER_IOU = 0.05


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

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
            patch = it["img"][y0:y1, x0:x1] - med  # residual above background
            bank.append({"patch": np.clip(patch, 0, None),
                         "box": np.array([x - x0, y - y0, w, h], np.float32)})
    return bank


def dihedral(img, boxes, k, flip):
    """Rotate by k*90 and optionally horizontal-flip image + xywh boxes."""
    H, W = img.shape[:2]
    img2 = np.rot90(img, k).copy()
    b = boxes.copy().reshape(-1, 4)
    for _ in range(k):
        # rot90 CCW: (x,y,w,h) in WxH -> (y, W - x - w, h, w)
        b = np.stack([b[:, 1], W - b[:, 0] - b[:, 2], b[:, 3], b[:, 2]], 1)
        H, W = W, H
    if flip:
        img2 = img2[:, ::-1].copy()
        b = np.stack([W - b[:, 0] - b[:, 2], b[:, 1], b[:, 2], b[:, 3]], 1)
    return img2, b


def paste_patch(img, patch_item, rng):
    """Additively paste a poison patch at a random location; returns its box."""
    patch, pbox = patch_item["patch"], patch_item["box"]
    k, flip = rng.integers(4), rng.integers(2)
    patch2, pbox2 = dihedral(patch, pbox[None], k, flip)
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
    """Paste a faint, realistic streak (Gaussian PSF line). Returns its box."""
    H, W = img.shape[:2]
    L = rng.uniform(25, 260)
    theta = rng.uniform(0, math.pi)
    sigma = rng.uniform(1.0, 2.2)
    snr = rng.uniform(3.0, 15.0)          # faint: well below poison brightness
    amp = snr * bg_sigma
    cx = rng.uniform(60, W - 60)
    cy = rng.uniform(60, H - 60)
    dx, dy = math.cos(theta) * L / 2, math.sin(theta) * L / 2
    x0, y0, x1, y1 = cx - dx, cy - dy, cx + dx, cy + dy
    canvas = np.zeros((H, W), np.float32)
    n = int(L * 2)
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    ix = np.clip(xs.astype(int), 0, W - 1)
    iy = np.clip(ys.astype(int), 0, H - 1)
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


def build_batch(items, bank, rng, batch_size, pixel_mean, pixel_std, device,
                n_extra_poison=(0, 3), n_synth=(0, 3)):
    """Returns ImageList-ready tensor batch + per-image poison boxes (800-scale)."""
    ims, poison_boxes = [], []
    for _ in range(batch_size):
        it = items[rng.integers(len(items))]
        img = it["img"].copy()
        boxes = it["boxes"].copy()
        k, flip = rng.integers(4), rng.integers(2)
        img, boxes = dihedral(img, boxes, k, flip)
        pboxes = [b for b in boxes]
        bg_sigma = float(img.std())
        for _ in range(rng.integers(*n_extra_poison)):
            b = paste_patch(img, bank[rng.integers(len(bank))], rng)
            if b is not None:
                pboxes.append(b)
        for _ in range(rng.integers(*n_synth)):
            draw_synthetic_streak(img, rng, min(bg_sigma, 800.0))
        # to model input domain: /65535*255, resize to 800, 3-channel BGR
        im8 = np.clip(img / 65535.0 * 255.0, 0, 255)
        im8 = cv2.resize(im8, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(im8).to(device=device, dtype=torch.float32)
        t = t.unsqueeze(0).repeat(3, 1, 1)
        t = (t - pixel_mean) / pixel_std
        ims.append(t)
        pb = np.array(pboxes, np.float32).reshape(-1, 4) * SCALE
        poison_boxes.append(torch.from_numpy(pb).to(device))
    images = ImageList.from_tensors(ims, 32)
    return images, poison_boxes


# --------------------------------------------------------------------------
# model helpers
# --------------------------------------------------------------------------

def permute_flatten(logits_per_level):
    """[B, A*K, H, W] per level -> [B, sum(HWA)] (K=1)."""
    flat = []
    for x in logits_per_level:
        B, AK, H, W = x.shape
        x = x.view(B, AK, 1, H, W).permute(0, 3, 4, 1, 2).reshape(B, -1)
        flat.append(x)
    return torch.cat(flat, dim=1)


def anchors_xyxy(model, features):
    anchors = model.anchor_generator(features)  # list[Boxes] per level
    return torch.cat([a.tensor for a in anchors], dim=0)  # [sum(HWA), 4]


def iou_anchors_boxes(anchors, boxes_xywh):
    """anchors [N,4] xyxy; boxes [M,4] xywh -> IoU [N,M]"""
    if boxes_xywh.numel() == 0:
        return anchors.new_zeros((anchors.shape[0], 0))
    b = boxes_xywh.clone()
    boxes = torch.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], 1)
    ix1 = torch.max(anchors[:, None, 0], boxes[None, :, 0])
    iy1 = torch.max(anchors[:, None, 1], boxes[None, :, 1])
    ix2 = torch.min(anchors[:, None, 2], boxes[None, :, 2])
    iy2 = torch.min(anchors[:, None, 3], boxes[None, :, 3])
    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih
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


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lam-forget", type=float, default=1.0)
    ap.add_argument("--lam-retain", type=float, default=30.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = common.build_cfg(device=device)
    student = build_model(cfg)
    DetectionCheckpointer(student).load(common.POISONED_WEIGHTS)
    teacher = build_model(cfg)
    DetectionCheckpointer(teacher).load(common.POISONED_WEIGHTS)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # freeze the box-regression path + first backbone stages
    for name, p in student.named_parameters():
        if "bbox_subnet" in name or "bbox_pred" in name:
            p.requires_grad_(False)
    student.train()
    # BN is frozen in detectron2 R50 (FrozenBatchNorm), no BN-stat drift to fear.

    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters)

    pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN, device=device).view(3, 1, 1)
    pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD, device=device).view(3, 1, 1)

    items = load_unlearn(common.UNLEARN_DIR)
    bank = make_poison_bank(items)
    print(f"{len(items)} unlearn images, {len(bank)} poison patches, device={device}")

    anchors = None
    t0 = time.time()
    for it in range(1, args.iters + 1):
        images, poison_boxes = build_batch(
            items, bank, rng, args.batch, pixel_mean, pixel_std, device)

        with torch.no_grad():
            tfeat = teacher.backbone(images.tensor)
            tfeats = [tfeat[f] for f in teacher.head_in_features]
            tlogits, _ = teacher.head(tfeats)
            tprob = torch.sigmoid(permute_flatten(tlogits))  # [B, N]

        sfeat = student.backbone(images.tensor)
        sfeats = [sfeat[f] for f in student.head_in_features]
        slogits, _ = student.head(sfeats)
        slog = permute_flatten(slogits)  # [B, N]

        if anchors is None:
            anchors = anchors_xyxy(student, sfeats)  # fixed input size

        loss_f = slog.new_zeros(())
        loss_r = slog.new_zeros(())
        n_f = 0
        for b in range(slog.shape[0]):
            iou = iou_anchors_boxes(anchors, poison_boxes[b])
            iou_max = iou.max(dim=1).values if iou.shape[1] else torch.zeros_like(slog[b])
            centers = center_in_boxes(anchors, poison_boxes[b])
            forget = (iou_max >= FORGET_IOU) | centers
            buffer = (~forget) & (iou_max >= BUFFER_IOU)
            retain = ~forget & ~buffer
            if forget.any():
                # push to background hard
                loss_f = loss_f + F.binary_cross_entropy_with_logits(
                    slog[b][forget], torch.zeros_like(slog[b][forget]), reduction="mean")
                n_f += int(forget.sum())
            tp = tprob[b][retain]
            sl = slog[b][retain]
            w = tp.clamp(min=0.005)          # emphasize teacher-positive anchors
            lr_ = F.binary_cross_entropy_with_logits(sl, tp, reduction="none")
            loss_r = loss_r + (w * lr_).sum() / w.sum()
        loss_f = loss_f / slog.shape[0]
        loss_r = loss_r / slog.shape[0]
        loss = args.lam_forget * loss_f + args.lam_retain * loss_r

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        sched.step()

        if it % 20 == 0 or it == 1:
            print(f"it {it:4d} | loss {loss.item():.4f} f {loss_f.item():.4f} "
                  f"r {loss_r.item():.4f} | forget anchors {n_f} | "
                  f"{(time.time()-t0)/it:.2f}s/it", flush=True)
        if it % 200 == 0 or it == args.iters:
            ckpt = {"model": student.state_dict()}
            torch.save(ckpt, out / f"student_{it}.pth")
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()

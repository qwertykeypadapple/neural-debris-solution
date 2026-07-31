"""Shared config/model/inference utilities for the neural-debris competition.

The poisoned model is a Detectron2 RetinaNet R50-FPN with custom anchors
(see the official baseline notebook). All inference must mirror the
baseline's DefaultPredictor pipeline so our boxes stay comparable with the
hidden clean model's reference predictions.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
POISONED_WEIGHTS = str(DATA / "poisoned_model" / "poisoned_model.pth")
UNLEARN_DIR = DATA / "unlearn_set"
TEST_DIR = DATA / "test_set" / "test_set"

BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES = [[16], [32], [64], [128], [256]]
NUM_CLASSES = 1
IMG_W = IMG_H = 1024


def build_cfg(weights=POISONED_WEIGHTS, score_thresh=0.2, device=None,
              detections_per_image=100):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
    cfg.TEST.DETECTIONS_PER_IMAGE = detections_per_image
    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    cfg.MODEL.DEVICE = device
    return cfg


def load_image(path):
    """16-bit PNG -> float32 [0,255] HxWx3, exactly like the baseline."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im.dtype == np.uint16:
        im = im.astype(np.float32) / 65535.0
    im = np.clip(im * 255, 0, 255).astype(np.float32)
    if im.ndim == 2:
        im = np.repeat(im[:, :, None], 3, axis=2)
    return im


def predict_folder(predictor, folder, out_json, limit=None):
    """Run a DefaultPredictor over a folder of PNGs -> {stem: [[conf,x,y,w,h],...]}."""
    files = sorted(Path(folder).glob("*.png"))
    if limit:
        files = files[:limit]
    results = {}
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
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out_json, "w"))
    return results


def dets_to_submission(dets_by_stem, out_csv, conf_map=None, min_conf=0.0):
    """Write submission CSV. dets_by_stem: {stem: [[conf,x,y,w,h],...]}.
    conf_map: optional callable conf->conf applied per detection."""
    import pandas as pd
    stems = sorted(dets_by_stem.keys())  # lexicographic, like sorted(glob)
    rows = []
    for stem in stems:
        parts = []
        for c, x, y, w, h in dets_by_stem[stem]:
            c2 = conf_map(c) if conf_map else c
            if c2 < min_conf:
                continue
            parts.extend([f"{c2:.6f}", f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}"])
        rows.append({"image_id": stem, "prediction_string": " ".join(parts) or " "})
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    df.to_csv(out_csv, index=False)
    return df

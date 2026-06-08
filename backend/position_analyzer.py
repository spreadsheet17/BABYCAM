"""
position_analyzer.py — Baby sleep position classifier.

v6 — YOLOv8 classify model replaces pose keypoints entirely.

Architecture:
  - A YOLOv8 classify model (trained on supine / prone / sideways image
    folders) is the sole classifier.  No MediaPipe, no keypoint arithmetic,
    no geometry heuristics.
  - A YOLO detect model (separate) finds the baby bounding box first; the
    crop is passed to the classifier.  If no detector is available the full
    frame is classified directly.
  - Class names must match the folder names used during training exactly.
    Default assumed order: ["prone", "sideways", "supine"]
    Override via PositionAnalyzer(class_names={...}) if yours differ.

Typical usage
-------------
    from position_analyzer import PositionAnalyzer

    analyzer = PositionAnalyzer(
        classifier_model_path="runs/classify/train/weights/best.pt",
        detector_model_path="yolov8n.pt",   # optional — omit for full-frame mode
    )

    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = analyzer.analyze(frame)
        frame   = analyzer.annotate(frame, results)
        cv2.imshow("Baby Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
"""

from __future__ import annotations


import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Sleep position enum ───────────────────────────────────────────────────────

class SleepPosition(Enum):
    SAFE        = auto()   # supine — on back, face up
    PRONE       = auto()   # on stomach — airway risk
    SIDE_UNSAFE = auto()   # on side — roll risk
    UNKNOWN     = auto()   # classifier not confident enough


# ── Default class-name → SleepPosition map ───────────────────────────────────
# Keys must match exactly what YOLOv8 reports (your training folder names,
# lower-cased by Ultralytics).  Add/rename entries if your folders differ.

_DEFAULT_CLASS_MAP: Dict[str, SleepPosition] = {
    "supine":   SleepPosition.SAFE,
    "prone":    SleepPosition.PRONE,
    "sideways": SleepPosition.SIDE_UNSAFE,
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PositionResult:
    position:   SleepPosition
    confidence: float
    bbox:       Optional[Tuple[int, int, int, int]]  # (x, y, w, h) | None = full-frame
    reason:     str
    timestamp:  float = field(default_factory=time.monotonic)

    @property
    def is_unsafe(self) -> bool:
        return self.position in (SleepPosition.PRONE, SleepPosition.SIDE_UNSAFE)

    def __str__(self) -> str:
        flag = "⚠ UNSAFE" if self.is_unsafe else "✓ safe"
        return f"{flag} | {self.position.name} ({self.confidence:.0%}) | {self.reason}"


# ── Main analyzer ─────────────────────────────────────────────────────────────

class PositionAnalyzer:
    """
    Two-stage pipeline
    ------------------
    1. Detector  — optional YOLO detect model locates the baby and produces
                   bounding boxes.  When omitted, the whole frame is used.
    2. Classifier — YOLOv8 classify model receives each crop and returns
                    supine / prone / sideways probabilities.

    Parameters
    ----------
    classifier_model_path : str
        Path to your trained YOLOv8 classify weights (e.g. best.pt).
    detector_model_path : str | None
        Path to a YOLO detect model for finding the baby first.
        Pass None to skip detection and classify the whole frame.
    class_names : dict | None
        Override the default class-name → SleepPosition mapping if your
        training folder names differ from {"supine", "prone", "sideways"}.
    confidence_threshold : float
        Minimum classifier probability to accept a result.
        Below this the result is SleepPosition.UNKNOWN.
    detector_confidence : float
        Minimum detection score to keep a bbox from the detector.
    on_unsafe : callable | None
        Called with each PositionResult whose is_unsafe is True.
    """

    # Extra context padding added around the detected bbox before cropping.
    # Gives the classifier a bit of surrounding scene (blanket edge etc.).
    CROP_PADDING = 0.10   # fraction of bbox width / height

    def __init__(
        self,
        classifier_model_path: str,
        detector_model_path:   Optional[str] = None,
        class_names:           Optional[Dict[str, SleepPosition]] = None,
        confidence_threshold:  float = 0.50,
        detector_confidence:   float = 0.40,
        on_unsafe:             Optional[Callable[[PositionResult], None]] = None,
    ):
        from ultralytics import YOLO

        self._cls_model   = YOLO(classifier_model_path)
        self._det_model   = YOLO(detector_model_path) if detector_model_path else None
        self._class_map   = class_names or _DEFAULT_CLASS_MAP
        self._conf_thresh = confidence_threshold
        self._det_conf    = detector_confidence
        self.on_unsafe    = on_unsafe

        logger.info(
            "PositionAnalyzer v6 ready | classifier=%s  detector=%s",
            classifier_model_path,
            detector_model_path or "none (full-frame mode)",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, frame: np.ndarray) -> List[PositionResult]:
        """
        Run the full pipeline on one BGR frame.
        Returns one PositionResult per detected baby (or one for the whole
        frame when running in full-frame mode).
        """
        crops = self._get_crops(frame)

        results: List[PositionResult] = []
        for crop, bbox in crops:
            result = self._classify_crop(crop, bbox)
            results.append(result)
            logger.info("Position: %s", result)
            if result.is_unsafe and self.on_unsafe:
                try:
                    self.on_unsafe(result)
                except Exception as exc:
                    logger.error("on_unsafe callback raised: %s", exc)

        return results

    def annotate(
        self,
        frame:   np.ndarray,
        results: List[PositionResult],
    ) -> np.ndarray:
        """Draw bounding boxes and position labels onto a copy of frame."""
        out = frame.copy()
        h, w = frame.shape[:2]

        for r in results:
            color = (0, 60, 220) if r.is_unsafe else (0, 200, 80)

            if r.bbox:
                x, y, bw, bh = r.bbox
            else:
                # full-frame mode — thin border around the whole image
                x, y, bw, bh = 4, 4, w - 8, h - 8

            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)

            label = f"{r.position.name}  {r.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x, y - th - 10), (x + tw + 6, y), color, -1)
            cv2.putText(
                out, label, (x + 3, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )

            if r.is_unsafe:
                cv2.putText(
                    out, "UNSAFE POSITION",
                    (x, y + bh + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 60, 220), 2, cv2.LINE_AA,
                )

        return out

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_crops(
        self,
        frame: np.ndarray,
    ) -> List[Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]]:
        """
        Returns [(crop_image, bbox_or_None), ...].
        bbox is (x, y, w, h) in frame pixels, or None for full-frame.
        Falls back to full-frame if the detector finds nothing.
        """
        if self._det_model is None:
            return [(frame, None)]

        fh, fw = frame.shape[:2]
        det_results = self._det_model(frame, conf=self._det_conf, verbose=False)
        crops: List[Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]] = []

        for r in det_results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                # ── Gate 1: person class only (COCO class 0) ──────────────
                if int(box.cls[0].item()) != 0:
                    logger.debug("Skipping non-person detection (class %d)", int(box.cls[0].item()))
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # ── Gate 2: aspect ratio — babies lie down, bbox should be
                #    wider than tall (or roughly square).  A standing adult
                #    produces a very tall narrow box (w/h < 0.4).
                bw, bh = x2 - x1, y2 - y1
                aspect = bw / bh if bh > 0 else 0
                if aspect < 0.35:
                    logger.debug("Skipping upright-person bbox (aspect=%.2f)", aspect)
                    continue

                # expand bbox slightly so the classifier sees some context
                pad_x = int((x2 - x1) * self.CROP_PADDING)
                pad_y = int((y2 - y1) * self.CROP_PADDING)
                x1c = max(0,  x1 - pad_x)
                y1c = max(0,  y1 - pad_y)
                x2c = min(fw, x2 + pad_x)
                y2c = min(fh, y2 + pad_y)

                crop = frame[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue

                crops.append((crop, (x1c, y1c, x2c - x1c, y2c - y1c)))

        if not crops:
            logger.debug("Detector found nothing — falling back to full-frame classify.")
            return [(frame, None)]

        return crops

    def _classify_crop(
        self,
        crop: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
    ) -> PositionResult:
        """Run the classify model on one crop and return a PositionResult."""
        cls_results = self._cls_model(crop, verbose=False)

        if not cls_results:
            return PositionResult(
                position=SleepPosition.UNKNOWN,
                confidence=0.0,
                bbox=bbox,
                reason="classifier returned no output",
            )

        probs      = cls_results[0].probs
        top_idx    = int(probs.top1)
        top_conf   = float(probs.top1conf)
        class_name = cls_results[0].names[top_idx].lower()

        if top_conf < self._conf_thresh:
            return PositionResult(
                position=SleepPosition.UNKNOWN,
                confidence=top_conf,
                bbox=bbox,
                reason=f"low confidence ({top_conf:.0%}) for '{class_name}'",
            )

        position = self._class_map.get(class_name, SleepPosition.UNKNOWN)

        if position is SleepPosition.UNKNOWN:
            logger.warning(
                "Class name '%s' not found in class_map. "
                "Update class_names in PositionAnalyzer.__init__().",
                class_name,
            )

        return PositionResult(
            position=position,
            confidence=top_conf,
            bbox=bbox,
            reason=f"classifier → '{class_name}' ({top_conf:.0%})",
        )

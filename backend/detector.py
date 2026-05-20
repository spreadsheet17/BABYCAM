"""
detector.py — YOLOv8 inference wrapper using Ultralytics.

Loads a YOLOv8 model once (detection OR pose variant), runs forward passes
on demand, and returns structured Detection objects.  GPU inference is used
automatically when CUDA is available.

Supports both:
  - Standard detection models (yolov8n/s/m/l/x.pt, or your custom .pt)
  - Pose models (yolov8n-pose.pt, etc.) — keypoints are attached to Detection

Usage:
    from detector import YOLOv8Detector, Detection

    det = YOLOv8Detector(model_path="best.pt")          # your trained model
    detections = det.run(frame)                          # list[Detection]
    annotated  = det.annotate(frame, detections)
"""

from __future__ import annotations


import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# COCO class index for "person"
PERSON_CLASS_ID = 0


# ──────────────────────────────────────────────────────────────────────────────
# Keypoint dataclass (COCO-17 layout, produced by YOLOv8-pose)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Keypoint:
    """Single pose keypoint: pixel coords + visibility score [0..1]."""
    x: float
    y: float
    visibility: float  # confidence from the pose head


# ──────────────────────────────────────────────────────────────────────────────
# Detection dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single detected object from a YOLOv8 forward pass."""

    class_id: int
    label: str
    confidence: float

    # Bounding box in pixel coordinates of the *original* frame
    x: int   # top-left x
    y: int   # top-left y
    w: int   # width
    h: int   # height

    # Pose keypoints — populated only when a pose model is used.
    # Index layout (COCO-17):
    #   0  nose        1  left_eye     2  right_eye
    #   3  left_ear    4  right_ear    5  left_shoulder
    #   6  right_shoulder  7 left_elbow  8 right_elbow
    #   9  left_wrist  10 right_wrist  11 left_hip
    #  12  right_hip   13 left_knee    14 right_knee
    #  15  left_ankle  16 right_ankle
    keypoints: List[Keypoint] = field(default_factory=list)

    # ── Derived helpers ────────────────────────────────────────────────────
    @property
    def cx(self) -> int:
        return self.x + self.w // 2

    @property
    def cy(self) -> int:
        return self.y + self.h // 2

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def aspect_ratio(self) -> float:
        """w / h  — > 1 → wide (likely lying down), < 0.6 → upright."""
        return self.w / self.h if self.h > 0 else 0.0

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    def is_person(self) -> bool:
        return self.class_id == PERSON_CLASS_ID

    def kp(self, idx: int) -> Optional[Keypoint]:
        """Return keypoint by index, or None if unavailable."""
        return self.keypoints[idx] if idx < len(self.keypoints) else None


# ──────────────────────────────────────────────────────────────────────────────
# Detector
# ──────────────────────────────────────────────────────────────────────────────

class YOLOv8Detector:
    """
    Parameters
    ----------
    model_path : str | Path
        Path to a YOLOv8 .pt weights file.  Can be a detection model
        (e.g. ``best.pt``) or a pose model (e.g. ``yolov8n-pose.pt``).
        Ultralytics auto-detects the task from the model metadata.
    conf_threshold : float
        Minimum confidence to keep a detection.
    iou_threshold : float
        IoU threshold for built-in NMS.
    target_classes : list[int] | None
        Only return detections whose class_id is in this list.
        ``None`` → return all classes.
    imgsz : int
        Inference image size (default 640).  Must be a multiple of 32.
    device : str | None
        ``"cuda"`` / ``"cpu"`` / ``"mps"`` or ``None`` (auto-select).
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        imgsz: int = 640,
        device: Optional[str] = None,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.target_classes = target_classes  # None = all; set [0] for person-only
        self.imgsz          = imgsz

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self._model  = YOLO(str(model_path))
        self._is_pose = self._model.task == "pose"
        self._labels  = self._model.names  # dict {class_id: name}

        # Device selection
        import torch
        if device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._device = device

        logger.info(
            "YOLOv8Detector loaded '%s' (task=%s) on %s",
            model_path.name, self._model.task, self._device,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 on *frame* (BGR numpy array).

        Returns a list of Detection objects, NMS-filtered, optionally
        restricted to target_classes.  Keypoints are populated when a pose
        model is used.
        """
        results = self._model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self._device,
            classes=self.target_classes,
            verbose=False,
            stream=False,
        )
        return self._parse_results(results[0], frame.shape[1], frame.shape[0])

    def annotate(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        color: Tuple[int, int, int] = (0, 200, 100),
    ) -> np.ndarray:
        """Draw bounding boxes (and skeleton if pose) on a copy of *frame*."""
        out = frame.copy()
        for d in detections:
            # Bounding box
            cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
            label_text = f"{d.label} {d.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (d.x, d.y - th - 8), (d.x + tw + 4, d.y), color, -1)
            cv2.putText(
                out, label_text,
                (d.x + 2, d.y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 1, cv2.LINE_AA,
            )
            # Skeleton overlay (pose model only)
            if d.keypoints:
                self._draw_skeleton(out, d.keypoints)
        return out

    # ── Internal helpers ───────────────────────────────────────────────────

    def _parse_results(self, result, frame_w: int, frame_h: int) -> List[Detection]:
        detections: List[Detection] = []

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Pose keypoints tensor: shape (N, 17, 3) → x, y, visibility
        kps_tensor = result.keypoints.data if (self._is_pose and result.keypoints is not None) else None

        for i, box in enumerate(boxes):
            class_id   = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label      = self._labels.get(class_id, str(class_id))

            # xyxy → x, y, w, h (integer pixel coords)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            # Clamp to frame
            x, y = max(0, x), max(0, y)
            w = min(w, frame_w - x)
            h = min(h, frame_h - y)

            # Keypoints
            kps: List[Keypoint] = []
            if kps_tensor is not None and i < len(kps_tensor):
                raw = kps_tensor[i].cpu().numpy()  # (17, 3)
                for kp_row in raw:
                    kps.append(Keypoint(
                        x=float(kp_row[0]),
                        y=float(kp_row[1]),
                        visibility=float(kp_row[2]),
                    ))

            detections.append(Detection(
                class_id=class_id,
                label=label,
                confidence=confidence,
                x=x, y=y, w=w, h=h,
                keypoints=kps,
            ))

        return detections

    @staticmethod
    def _draw_skeleton(
        frame: np.ndarray,
        keypoints: List[Keypoint],
        vis_threshold: float = 0.3,
    ) -> None:
        """Draw COCO-17 skeleton on *frame* in-place."""
        # COCO skeleton pairs (0-indexed)
        SKELETON = [
            (0, 1), (0, 2), (1, 3), (2, 4),          # head
            (5, 6),                                     # shoulders
            (5, 7), (7, 9), (6, 8), (8, 10),           # arms
            (5, 11), (6, 12), (11, 12),                 # torso
            (11, 13), (13, 15), (12, 14), (14, 16),    # legs
        ]
        pts = [
            (int(kp.x), int(kp.y)) if kp.visibility >= vis_threshold else None
            for kp in keypoints
        ]
        for a, b in SKELETON:
            if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                cv2.line(frame, pts[a], pts[b], (0, 255, 200), 1, cv2.LINE_AA)
        for pt in pts:
            if pt:
                cv2.circle(frame, pt, 3, (255, 255, 0), -1)
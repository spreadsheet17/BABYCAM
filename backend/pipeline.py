"""
pipeline.py — Main inference loop for the baby monitor.

Ties RTSPCapture → YOLOv8Detector → PositionAnalyzer into a single blocking
run() call.  Designed to be imported by the full server process or run
directly for testing / debugging.

Run directly for a live debug window:
    python pipeline.py --model best.pt \
                       --rtsp  rtsp://192.168.1.100:554/stream1 \
                       --show

Environment variables (override CLI defaults):
    RTSP_URL      RTSP stream URL
    YOLO_MODEL    path to YOLOv8 .pt weights file
"""

from __future__ import annotations

import cv2
from position_analyzer import PositionAnalyzer

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from capture import RTSPCapture
from detector import Detection, YOLOv8Detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class BabyMonitorPipeline:
    """
    Orchestrates capture → detection → callback chain.

    Parameters
    ----------
    rtsp_url : str
        RTSP stream to ingest.
    detector : YOLOv8Detector
        Initialised detector (loaded weights, CUDA ready).
    on_detections : Callable[[np.ndarray, list[Detection]], None] | None
        Called every time detections are produced.  Receives the raw BGR
        frame and the detection list.  Runs in the pipeline thread — keep
        it fast or hand off to a queue.
    target_fps : float
        Max processing rate.  Caps GPU usage; does not affect capture FPS.
    show_preview : bool
        Open an OpenCV window with live annotated feed (debug only).
    """

    def __init__(
        self,
        rtsp_url: str,
        detector: YOLOv8Detector,
        on_detections: Optional[Callable[[np.ndarray, List[Detection]], None]] = None,
        target_fps: float = 10.0,
        show_preview: bool = False,
    ):
        self.rtsp_url     = rtsp_url
        self.detector     = detector
        self.on_detections = on_detections
        self.frame_interval = 1.0 / max(target_fps, 1.0)
        self.show_preview   = show_preview

        self._capture = RTSPCapture(rtsp_url=rtsp_url)
        self._running = False

        self._frames_processed        = 0
        self._frames_with_detections  = 0
        self._start_time: float       = 0.0

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start capture thread then enter the blocking inference loop."""
        self._capture.start()
        self._running   = True
        self._start_time = time.monotonic()
        logger.info(
            "Pipeline running — target %.1f FPS, preview=%s",
            1.0 / self.frame_interval,
            self.show_preview,
        )
        time.sleep(0.5)  # allow capture thread to buffer the first frame
        try:
            self._inference_loop()
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        self._capture.stop()
        if self.show_preview:
            cv2.destroyAllWindows()
        self._log_stats()

    # ── Internal ────────────────────────────────────────────────────────────

    def _inference_loop(self) -> None:
        last_tick = time.monotonic()

        while self._running:
            # Throttle to target FPS
            now      = time.monotonic()
            sleep_for = self.frame_interval - (now - last_tick)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_tick = time.monotonic()

            frame = self._capture.read()
            if frame is None:
                logger.debug("No frame yet, waiting…")
                continue

            # Inference
            t0           = time.monotonic()
            detections   = self.detector.run(frame)
            inference_ms = (time.monotonic() - t0) * 1000

            self._frames_processed += 1
            if detections:
                self._frames_with_detections += 1

            # Callback
            if self.on_detections:
                try:
                    self.on_detections(frame, detections)
                except Exception as exc:
                    logger.error("on_detections callback raised: %s", exc)

            # Debug preview
            if self.show_preview:
                if (hasattr(self.on_detections, '__self__') and
                        hasattr(self.on_detections.__self__, 'annotate')):
                    analyzer  = self.on_detections.__self__
                    results   = analyzer.analyze(frame, detections)
                    annotated = analyzer.annotate(frame, results)
                else:
                    annotated = self.detector.annotate(frame, detections)

                cv2.putText(
                    annotated,
                    f"Inference: {inference_ms:.1f}ms | Detections: {len(detections)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 0), 2, cv2.LINE_AA,
                )
                cv2.imshow("Baby Monitor — debug", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    logger.info("User pressed 'q', shutting down.")
                    self._running = False

    def _log_stats(self) -> None:
        elapsed = time.monotonic() - self._start_time
        fps     = self._frames_processed / elapsed if elapsed > 0 else 0
        logger.info(
            "Pipeline stopped. Processed %d frames in %.1fs (%.1f fps). "
            "Frames with detections: %d (%.0f%%)",
            self._frames_processed, elapsed, fps,
            self._frames_with_detections,
            100 * self._frames_with_detections / max(self._frames_processed, 1),
        )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baby monitor inference pipeline")
    parser.add_argument(
        "--rtsp",
        default=os.getenv("RTSP_URL", "rtsp://192.168.1.100:554/stream1"),
        help="RTSP URL of the camera",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("YOLO_MODEL", "best.pt"),
        help="Path to YOLOv8 .pt weights (detection or pose model)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="YOLOv8 input resolution (default 640)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.45,
        help="Confidence threshold (default 0.45)",
    )
    parser.add_argument(
        "--fps", type=float, default=10.0,
        help="Target inference FPS (default 10)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Open a live annotated preview window",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    detector = YOLOv8Detector(
        model_path=args.model,
        conf_threshold=args.conf,
        imgsz=args.imgsz,
        target_classes=[0],  # person only
    )

    analyzer = PositionAnalyzer(
        on_unsafe=lambda result: logger.warning(
            "UNSAFE SLEEP DETECTED: %s", result
        ),
    )

    pipeline = BabyMonitorPipeline(
        rtsp_url=args.rtsp,
        detector=detector,
        on_detections=analyzer.analyze,
        target_fps=args.fps,
        show_preview=args.show,
    )

    def _handle_signal(sig, _frame):
        logger.info("Signal %s received, stopping…", sig)
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    pipeline.start()


if __name__ == "__main__":
    main()
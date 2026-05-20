"""
server.py — Flask API server for BabyCam.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify
from flask_cors import CORS

from capture import RTSPCapture
from position_analyzer import PositionAnalyzer, PositionResult, SleepPosition

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--rtsp",       default="rtsp://autofister17:potatoes17@192.168.1.19:554/stream1")
parser.add_argument("--classifier", default="classifier.pt", help="YOLOv8 classify weights")
parser.add_argument("--detector",   default=None,            help="YOLOv8 detect weights (optional)")
parser.add_argument("--conf",       type=float, default=0.45)
parser.add_argument("--cls-conf",   type=float, default=0.50, help="Min classifier confidence")
parser.add_argument("--imgsz",      type=int,   default=640)
parser.add_argument("--fps",        type=float, default=20.0)
parser.add_argument("--port",       type=int,   default=5000)
parser.add_argument("--skip",       type=int,   default=2)
args, _ = parser.parse_known_args()

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Shared state ──────────────────────────────────────────────────────────────
_lock  = threading.Lock()
_state = {
    "position":        "UNKNOWN",
    "is_safe":         True,
    "confidence":      0.0,
    "reason":          "",
    "alert_count":     0,
    "last_alert_time": None,
    "fps":             0.0,
    "stream_ok":       False,
}
_latest_frame: Optional[bytes] = None
_frame_lock = threading.Lock()

_POSITION_MAP = {
    SleepPosition.SAFE:        ("SAFE",        True),
    SleepPosition.PRONE:       ("PRONE",       False),
    SleepPosition.SIDE_UNSAFE: ("SIDE_UNSAFE", False),
    SleepPosition.UNKNOWN:     ("UNKNOWN",     False),
}

_ALERT_COOLDOWN_S  = 10
_ALERT_FRAME_COUNT = 2
_unsafe_streak     = 0
_last_alert_ts     = 0.0


# ── Inference callback ────────────────────────────────────────────────────────
def _on_frame(frame: np.ndarray) -> None:
    global _latest_frame, _unsafe_streak, _last_alert_ts

    results: list[PositionResult] = analyzer.analyze(frame)

    annotated = analyzer.annotate(frame, results) if results else frame
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
    with _frame_lock:
        _latest_frame = buf.tobytes()

    with _lock:
        _state["stream_ok"] = True

        if not results:
            _state["position"]   = "NO_BABY"
            _state["is_safe"]    = True
            _state["confidence"] = 0.0
            _state["reason"]     = "No detection"
            _unsafe_streak       = 0
            return

        best = max(results, key=lambda r: r.confidence)
        pos_str, is_safe = _POSITION_MAP.get(best.position, ("UNKNOWN", False))

        _state["position"]   = pos_str
        _state["is_safe"]    = is_safe
        _state["confidence"] = round(best.confidence, 3)
        _state["reason"]     = best.reason

        if not is_safe:
            _unsafe_streak += 1
        else:
            _unsafe_streak = 0

        now = time.time()
        if (_unsafe_streak >= _ALERT_FRAME_COUNT and
                now - _last_alert_ts > _ALERT_COOLDOWN_S):
            _state["alert_count"]     += 1
            _state["last_alert_time"]  = time.strftime("%H:%M:%S")
            _last_alert_ts             = now
            logger.warning(
                "ALERT #%d — %s (%s)",
                _state["alert_count"], pos_str, best.reason,
            )


# ── Pipeline thread ───────────────────────────────────────────────────────────
def _pipeline_thread() -> None:
    frame_interval = 1.0 / max(args.fps, 1.0)
    skip           = max(args.skip, 1)
    frame_idx      = 0
    prev_t         = time.monotonic()

    capture.start()
    time.sleep(0.5)

    while True:
        now    = time.monotonic()
        sleep  = frame_interval - (now - prev_t)
        if sleep > 0:
            time.sleep(sleep)
        prev_t = time.monotonic()

        frame = capture.read()
        if frame is None:
            continue

        frame_idx += 1
        if frame_idx % skip == 0:
            _on_frame(frame)

        with _lock:
            _state["fps"] = round(1.0 / max(frame_interval, 1e-9), 1)


# ── Routes ────────────────────────────────────────────────────────────────────
def _mjpeg_generator():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)


@app.route("/stream")
def stream():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/status")
def status():
    with _lock:
        return jsonify(dict(_state))

@app.route("/health")
def health():
    return jsonify({"ok": True, "time": time.strftime("%H:%M:%S")})


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    analyzer = PositionAnalyzer(
        classifier_model_path=args.classifier,
        detector_model_path=args.detector,       # None = full-frame mode
        confidence_threshold=args.cls_conf,
        detector_confidence=args.conf,
        on_unsafe=lambda r: None,
    )

    capture = RTSPCapture(rtsp_url=args.rtsp)

    t = threading.Thread(target=_pipeline_thread, daemon=True)
    t.start()

    logger.info("Server → http://0.0.0.0:%d", args.port)
    logger.info("Stream → http://0.0.0.0:%d/stream", args.port)

    from waitress import serve
    serve(app, host="0.0.0.0", port=args.port, threads=8)
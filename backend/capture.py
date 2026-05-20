"""
capture.py — RTSP frame capture with CUDA-accelerated OpenCV.

Responsibilities:
  - Connect to the RTSP stream and reconnect on drop.
  - Decode frames on the GPU (VideoCapture with CUDA backend).
  - Expose a thread-safe frame queue consumed by the detector.

Usage:
    from capture import RTSPCapture
    cap = RTSPCapture(rtsp_url="rtsp://192.168.1.100:554/stream")
    cap.start()
    frame = cap.read()   # latest BGR frame (numpy ndarray) or None
    cap.stop()
"""
import cv2
import threading
import time
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class RTSPCapture:
    """
    Background thread that continuously reads from an RTSP source.

    Parameters
    ----------
    rtsp_url : str
        Full RTSP URL, e.g. rtsp://admin:pass@192.168.1.100:554/stream1
    reconnect_delay : float
        Seconds to wait before retrying after a dropped connection.
    buffer_size : int
        Max frames kept in the internal deque (1 = always newest frame only).
    use_cuda : bool
        If True, open the stream with the CUDA-backed VideoCapture when
        available.  Falls back to the default backend silently.
    """

    def __init__(
        self,
        rtsp_url: str,
        reconnect_delay: float = 3.0,
        buffer_size: int = 1,
        use_cuda: bool = True,
    ):
        self.rtsp_url = rtsp_url
        self.reconnect_delay = reconnect_delay
        self.use_cuda = use_cuda

        self._queue: deque = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "RTSPCapture":
        """Start the background capture thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._capture_loop, name="RTSPCapture", daemon=True
        )
        self._thread.start()
        logger.info("RTSPCapture started → %s", self.rtsp_url)
        return self

    def stop(self) -> None:
        """Signal the capture thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap and self._cap.isOpened():
            self._cap.release()
        logger.info("RTSPCapture stopped.")

    def read(self) -> Optional[object]:
        """
        Return the latest decoded BGR frame (numpy ndarray) or None if no
        frame has been received yet.
        """
        with self._lock:
            return self._queue[-1] if self._queue else None

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_capture(self) -> cv2.VideoCapture:
        """Open VideoCapture with CUDA or default backend."""
        if self.use_cuda:
            try:
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                # Try to enable hardware decode; silently ignored if unavailable.
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                if cap.isOpened():
                    logger.info("Opened stream with HW acceleration.")
                    return cap
            except Exception as exc:  # noqa: BLE001
                logger.warning("HW acceleration unavailable: %s", exc)

        cap = cv2.VideoCapture(self.rtsp_url)
        logger.info("Opened stream with default (CPU) backend.")
        return cap

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            self._cap = self._open_capture()

            if not self._cap.isOpened():
                logger.warning(
                    "Could not open RTSP stream, retrying in %.1fs…",
                    self.reconnect_delay,
                )
                time.sleep(self.reconnect_delay)
                continue

            logger.info("Stream connected.")
            consecutive_failures = 0

            while not self._stop_event.is_set():
                ret, frame = self._cap.read()

                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        logger.warning("Stream lost, reconnecting…")
                        break
                    continue

                consecutive_failures = 0
                with self._lock:
                    self._queue.append(frame)

            self._cap.release()

            if not self._stop_event.is_set():
                logger.info(
                    "Waiting %.1fs before reconnect…", self.reconnect_delay
                )
                time.sleep(self.reconnect_delay)

from __future__ import annotations

import time
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import resolve_project_path


class RoiVideoRecorder:
    """Record ROI frames onto a fixed-FPS timeline until explicitly stopped."""

    def __init__(self, config: dict):
        self.fps = max(1.0, float(config.get("fps", 30)))
        self.codec = str(config.get("codec", "mp4v"))[:4].ljust(4)
        self.output_dir = resolve_project_path(str(config.get("output_dir", "recordings")))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer: cv2.VideoWriter | None = None
        self.output_path: Path | None = None
        self.started_at: float | None = None
        self.frames_written = 0

    @property
    def active(self) -> bool:
        return self.writer is not None

    def start(self, frame: np.ndarray, now: float | None = None) -> Path:
        if self.active:
            raise RuntimeError("ROI recording is already active")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("ROI recording requires a BGR image")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        extension = ".mp4" if self.codec.strip().lower() in {"mp4v", "avc1", "h264"} else ".avi"
        candidate = self.output_dir / f"roi_{timestamp}{extension}"
        suffix = 1
        while candidate.exists():
            candidate = self.output_dir / f"roi_{timestamp}_{suffix:03d}{extension}"
            suffix += 1
        self.output_path = candidate
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        height, width = frame.shape[:2]
        self.writer = cv2.VideoWriter(
            str(self.output_path), fourcc, self.fps, (width, height)
        )
        if not self.writer.isOpened():
            self.writer.release()
            self.writer = None
            raise RuntimeError(f"无法创建视频文件：{self.output_path}")

        self.started_at = time.perf_counter() if now is None else now
        self.frames_written = 0
        return self.output_path

    def add_frame(self, frame: np.ndarray, now: float | None = None) -> Path | None:
        if not self.active or self.started_at is None or self.output_path is None:
            return None
        current = time.perf_counter() if now is None else now
        elapsed = max(0.0, current - self.started_at)
        desired_frames = max(1, math.ceil(elapsed * self.fps))
        while self.frames_written < desired_frames:
            self.writer.write(frame)
            self.frames_written += 1
        return None

    def elapsed_seconds(self, now: float | None = None) -> float:
        if not self.active or self.started_at is None:
            return 0.0
        current = time.perf_counter() if now is None else now
        return max(0.0, current - self.started_at)

    def stop(self) -> Path | None:
        path = self.output_path
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.started_at = None
        self.frames_written = 0
        return path

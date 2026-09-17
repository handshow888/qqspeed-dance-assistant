from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .config import resolve_project_path


def _relative_roi(config: dict, name: str, default: list[float]) -> tuple[float, ...]:
    value = config.get(name, default)
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"space.{name} 必须包含 4 个比例值")
    result = tuple(float(item) for item in value)
    if (
        any(item < 0 for item in result)
        or result[2] <= 0
        or result[3] <= 0
        or result[0] + result[2] > 1
        or result[1] + result[3] > 1
    ):
        raise ValueError(f"space.{name} 无效")
    return result


def _string_tuple(config: dict, name: str, default: list[str]) -> tuple[str, ...]:
    value = config.get(name, default)
    if not isinstance(value, list) or not value:
        raise ValueError(f"space.{name} 必须是非空路径列表")
    return tuple(str(item) for item in value)


@dataclass(frozen=True)
class SpaceTimingConfig:
    enabled: bool
    slider_template_paths: tuple[str, ...]
    slider_search_roi: tuple[float, ...]
    feature_scale_factors: tuple[float, ...]
    slider_match_threshold: float
    cursor_window_ratio: float
    mean_offset_ms: float
    stddev_ms: float
    max_abs_offset_ms: float
    key_hold_min_ms: int
    key_hold_max_ms: int
    prediction_horizon_ms: float
    prediction_cache_ms: float
    minimum_schedule_lead_ms: float
    late_tolerance_ms: float
    expire_after_directions_ms: float
    min_speed_px_per_second: float
    max_speed_px_per_second: float
    minimum_track_samples: int

    @classmethod
    def from_config(cls, config: dict) -> "SpaceTimingConfig":
        hold = config.get("key_hold_ms", {"min": 25, "max": 45})
        hold_min = max(1, int(hold.get("min", 25)))
        hold_max = max(hold_min, int(hold.get("max", 45)))
        return cls(
            enabled=bool(config.get("enabled", True)),
            slider_template_paths=_string_tuple(
                config, "slider_templates", ["assets/节奏条滑块.png"]
            ),
            slider_search_roi=_relative_roi(
                config,
                "slider_search_roi_in_arrow_roi",
                [0.0, 0.0, 1.0, 0.45],
            ),
            feature_scale_factors=tuple(
                max(0.1, float(value))
                for value in config.get("feature_scale_factors", [0.9, 1.0, 1.1])
            ),
            slider_match_threshold=min(
                1.0, max(0.0, float(config.get("slider_match_threshold", 0.62)))
            ),
            cursor_window_ratio=min(
                0.98,
                max(0.02, float(config.get("cursor_window_ratio", 0.617371))),
            ),
            mean_offset_ms=float(config.get("mean_offset_ms", 0.0)),
            stddev_ms=max(0.0, float(config.get("stddev_ms", 8.0))),
            max_abs_offset_ms=max(
                0.0, float(config.get("max_abs_offset_ms", 25.0))
            ),
            key_hold_min_ms=hold_min,
            key_hold_max_ms=hold_max,
            prediction_horizon_ms=max(
                20.0, float(config.get("prediction_horizon_ms", 180.0))
            ),
            prediction_cache_ms=max(
                0.0, float(config.get("prediction_cache_ms", 1800.0))
            ),
            minimum_schedule_lead_ms=max(
                0.0, float(config.get("minimum_schedule_lead_ms", 8.0))
            ),
            late_tolerance_ms=max(
                0.0, float(config.get("late_tolerance_ms", 45.0))
            ),
            expire_after_directions_ms=max(
                100.0, float(config.get("expire_after_directions_ms", 2200.0))
            ),
            min_speed_px_per_second=max(
                1.0, float(config.get("min_speed_px_per_second", 120.0))
            ),
            max_speed_px_per_second=max(
                1.0, float(config.get("max_speed_px_per_second", 900.0))
            ),
            minimum_track_samples=max(
                2, int(config.get("minimum_track_samples", 3))
            ),
        )

    def sample_offset_ms(self, rng: random.Random | None = None) -> float:
        source = rng or random
        value = source.gauss(self.mean_offset_ms, self.stddev_ms)
        return min(self.max_abs_offset_ms, max(-self.max_abs_offset_ms, value))


@dataclass(frozen=True)
class SliderObservation:
    marker_x: float | None
    target_x: float
    speed_px_per_second: float | None
    crossing_at: float | None
    prediction_cached: bool
    slider_match_score: float | None


class SliderTracker:
    """Track the moving slider against a fixed horizontal window position."""

    def __init__(self, timing: SpaceTimingConfig):
        self.timing = timing
        self._slider_templates = [
            self._read_template(path) for path in timing.slider_template_paths
        ]
        self._samples: deque[tuple[float, float]] = deque(maxlen=10)
        self._last_visual_x: float | None = None
        self._cached_crossing_at: float | None = None
        self._prediction_updated_at: float | None = None
        self._learned_speed: float | None = None

    @staticmethod
    def _read_template(path_value: str) -> np.ndarray:
        path = resolve_project_path(path_value)
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"无法读取滑块识别模板：{path}")
        return image

    def reset(self, *, keep_target: bool = True) -> None:
        self._samples.clear()
        self._last_visual_x = None
        self._cached_crossing_at = None
        self._prediction_updated_at = None
        if not keep_target:
            self._learned_speed = None

    def observe(
        self,
        frame: np.ndarray,
        captured_at: float,
        *,
        detected_slider_rect: tuple[int, int, int, int] | None = None,
        detected_slider_score: float | None = None,
        allow_template_fallback: bool = True,
    ) -> SliderObservation:
        _height, width = frame.shape[:2]
        target_x = width * self.timing.cursor_window_ratio
        marker = self._marker_from_detection(
            detected_slider_rect,
            detected_slider_score,
            width,
            target_x,
        )
        if marker is None and allow_template_fallback:
            marker = self._find_slider(frame, target_x)
        marker_x = marker[0] if marker is not None else None
        slider_score = marker[1] if marker is not None else None

        marker_advanced = False
        if marker_x is not None:
            backward_limit = max(80.0, width * 0.05)
            if (
                self._last_visual_x is not None
                and marker_x < self._last_visual_x - backward_limit
            ):
                self.reset(keep_target=True)
            if self._last_visual_x is None or marker_x - self._last_visual_x >= 0.75:
                self._samples.append((captured_at, marker_x))
                self._last_visual_x = marker_x
                marker_advanced = True

        speed = self._estimate_speed()
        if speed is None and self._learned_speed is not None and len(self._samples) >= 2:
            speed = self._learned_speed

        current_crossing_at = None
        if marker_advanced and marker_x is not None:
            line_tolerance = max(2.0, width * 0.002)
            if marker_x >= target_x - line_tolerance:
                current_crossing_at = captured_at
            elif speed is not None:
                current_crossing_at = captured_at + (target_x - marker_x) / speed
            if current_crossing_at is not None:
                self._cached_crossing_at = current_crossing_at
                self._prediction_updated_at = captured_at

        prediction_cached = False
        crossing_at = current_crossing_at
        if (
            crossing_at is None
            and self._cached_crossing_at is not None
            and self._prediction_updated_at is not None
            and (captured_at - self._prediction_updated_at) * 1000
            <= self.timing.prediction_cache_ms
        ):
            crossing_at = self._cached_crossing_at
            prediction_cached = True

        return SliderObservation(
            marker_x=marker_x,
            target_x=target_x,
            speed_px_per_second=speed,
            crossing_at=crossing_at,
            prediction_cached=prediction_cached,
            slider_match_score=slider_score,
        )

    @staticmethod
    def annotate(frame: np.ndarray, observation: SliderObservation) -> np.ndarray:
        height = frame.shape[0]
        target_x = round(observation.target_x)
        cv2.line(frame, (target_x, 0), (target_x, height), (255, 255, 255), 2)
        if observation.marker_x is not None:
            marker_x = round(observation.marker_x)
            cv2.line(frame, (marker_x, 0), (marker_x, height), (255, 220, 20), 2)
        return frame

    def _marker_from_detection(
        self,
        rect: tuple[int, int, int, int] | None,
        score: float | None,
        frame_width: int,
        target_x: float,
    ) -> tuple[float, float] | None:
        if rect is None:
            return None
        marker_x = (rect[0] + rect[2]) / 2
        if not 0 <= marker_x <= frame_width:
            return None
        if marker_x > target_x + max(8.0, frame_width * 0.01):
            return None
        if self._last_visual_x is not None:
            distance = marker_x - self._last_visual_x
            if distance < -max(80.0, frame_width * 0.05):
                return None
            if distance > max(100.0, frame_width * 0.06):
                return None
        return marker_x, float(score or 0.0)

    def _find_slider(
        self, frame: np.ndarray, target_x: float
    ) -> tuple[float, float] | None:
        if frame.size == 0:
            return None
        height, width = frame.shape[:2]
        rx, ry, rw, rh = self.timing.slider_search_roi
        left = max(0, min(width - 1, round(width * rx)))
        top = max(0, min(height - 1, round(height * ry)))
        right = min(width, max(left + 1, round(width * (rx + rw))))
        bottom = min(height, max(top + 1, round(height * (ry + rh))))
        right = min(right, max(left + 1, round(target_x + width * 0.01)))
        crop = frame[top:bottom, left:right]
        result = self._match_templates(crop)
        if result is None or result[0] < self.timing.slider_match_threshold:
            return None
        score, center_x = result
        marker_x = left + center_x
        return self._marker_from_detection(
            (round(marker_x), 0, round(marker_x), 1),
            score,
            width,
            target_x,
        )

    def _match_templates(self, image: np.ndarray) -> tuple[float, float] | None:
        if image.size == 0:
            return None
        source = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        best: tuple[float, float] | None = None
        for base in self._slider_templates:
            base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            for factor in self.timing.feature_scale_factors:
                template_width = max(2, round(base_gray.shape[1] * factor))
                template_height = max(2, round(base_gray.shape[0] * factor))
                if template_width > source.shape[1] or template_height > source.shape[0]:
                    continue
                template = cv2.resize(
                    base_gray,
                    (template_width, template_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(result)
                adjusted_score = float(score) - abs(factor - 1.0) * 0.04
                center_x = float(location[0] + template_width / 2)
                if best is None or adjusted_score > best[0]:
                    best = adjusted_score, center_x
        return best

    def _estimate_speed(self) -> float | None:
        if len(self._samples) < self.timing.minimum_track_samples:
            return None
        velocities: list[float] = []
        samples = list(self._samples)
        for first in range(len(samples) - 1):
            time_a, x_a = samples[first]
            for second in range(first + 1, len(samples)):
                time_b, x_b = samples[second]
                elapsed = time_b - time_a
                distance = x_b - x_a
                if elapsed <= 0 or distance <= 0.5:
                    continue
                velocity = distance / elapsed
                if (
                    self.timing.min_speed_px_per_second
                    <= velocity
                    <= self.timing.max_speed_px_per_second
                ):
                    velocities.append(velocity)
        if not velocities:
            return None
        speed = float(np.median(velocities))
        if self._learned_speed is None:
            self._learned_speed = speed
        else:
            self._learned_speed = self._learned_speed * 0.8 + speed * 0.2
        return speed

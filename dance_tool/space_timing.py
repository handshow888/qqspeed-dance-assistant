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
    if any(item < 0 for item in result) or result[2] <= 0 or result[3] <= 0:
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
    bar_template_path: str
    slider_template_paths: tuple[str, ...]
    cursor_template_paths: tuple[str, ...]
    bar_search_roi: tuple[float, ...]
    bar_width_ratio: float
    bar_scale_factors: tuple[float, ...]
    feature_scale_factors: tuple[float, ...]
    bar_match_threshold: float
    slider_match_threshold: float
    cursor_match_threshold: float
    bar_vertical_padding: float
    cursor_prior_ratio: float
    cursor_search_radius_ratio: float
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
        bar_scales = tuple(
            max(0.1, float(value))
            for value in config.get("bar_scale_factors", [0.97, 1.0, 1.03])
        )
        feature_scales = tuple(
            max(0.1, float(value))
            for value in config.get("feature_scale_factors", [0.9, 1.0, 1.1])
        )
        return cls(
            enabled=bool(config.get("enabled", True)),
            bar_template_path=str(config.get("bar_template", "assets/节奏条.png")),
            slider_template_paths=_string_tuple(
                config, "slider_templates", ["assets/节奏条滑块.png"]
            ),
            cursor_template_paths=_string_tuple(
                config,
                "cursor_templates",
                [
                    "assets/节奏条光标.png",
                    "assets/节奏条光标2.png",
                    "assets/节奏条光标3.png",
                ],
            ),
            bar_search_roi=_relative_roi(
                config, "bar_search_roi_in_arrow_roi", [0.08, 0.0, 0.84, 0.45]
            ),
            bar_width_ratio=min(
                0.95, max(0.1, float(config.get("bar_width_ratio", 0.51)))
            ),
            bar_scale_factors=bar_scales,
            feature_scale_factors=feature_scales,
            bar_match_threshold=min(
                1.0, max(0.0, float(config.get("bar_match_threshold", 0.50)))
            ),
            slider_match_threshold=min(
                1.0, max(0.0, float(config.get("slider_match_threshold", 0.62)))
            ),
            cursor_match_threshold=min(
                1.0, max(0.0, float(config.get("cursor_match_threshold", 0.68)))
            ),
            bar_vertical_padding=max(
                0.0, float(config.get("bar_vertical_padding", 0.24))
            ),
            cursor_prior_ratio=min(
                0.95, max(0.05, float(config.get("cursor_prior_ratio", 0.85)))
            ),
            cursor_search_radius_ratio=min(
                0.25,
                max(0.02, float(config.get("cursor_search_radius_ratio", 0.08))),
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
class RhythmObservation:
    marker_x: float | None
    target_x: float | None
    speed_px_per_second: float | None
    crossing_at: float | None
    prediction_cached: bool
    bar_rect: tuple[int, int, int, int]
    bar_match_score: float | None
    slider_match_score: float | None
    cursor_match_score: float | None
    bar_locked: bool


class RhythmBarTracker:
    """Locate the bar first, then track its slider and cursor inside a tight ROI."""

    def __init__(self, timing: SpaceTimingConfig):
        self.timing = timing
        self._bar_template = self._read_template(timing.bar_template_path)
        self._slider_templates = [
            self._read_template(path) for path in timing.slider_template_paths
        ]
        self._cursor_templates = [
            self._read_template(path) for path in timing.cursor_template_paths
        ]
        self._samples: deque[tuple[float, float]] = deque(maxlen=10)
        self._bar_candidates: deque[tuple[int, int, int, int, float]] = deque(maxlen=4)
        self._bar_box: tuple[int, int, int, int] | None = None
        self._bar_scale: float | None = None
        self._bar_locked = False
        self._bar_match_score: float | None = None
        self._target_x: float | None = None
        self._target_samples: deque[float] = deque(maxlen=5)
        self._target_locked = False
        self._cursor_match_score: float | None = None
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
            raise FileNotFoundError(f"无法读取节奏识别模板：{path}")
        return image

    def reset(self, *, keep_target: bool = True) -> None:
        self._samples.clear()
        self._last_visual_x = None
        self._cached_crossing_at = None
        self._prediction_updated_at = None
        if not keep_target:
            self._bar_candidates.clear()
            self._bar_box = None
            self._bar_scale = None
            self._bar_locked = False
            self._bar_match_score = None
            self._target_x = None
            self._target_samples.clear()
            self._target_locked = False
            self._cursor_match_score = None
            self._learned_speed = None

    def observe(self, frame: np.ndarray, captured_at: float) -> RhythmObservation:
        located = self._locate_bar(frame)
        if located is None:
            return self._observation(None, None, None, None, False, (0, 0, 0, 0))

        bar_rect, bar_scale = located
        left, top, right, bottom = bar_rect
        crop = frame[top:bottom, left:right]

        target = self._find_cursor(crop, left, bar_scale)
        if target is not None and not self._target_locked:
            target_x, score = target
            self._cursor_match_score = score
            if self._target_x is None or abs(target_x - self._target_x) <= 8:
                self._target_samples.append(target_x)
            else:
                self._target_samples.clear()
                self._target_samples.append(target_x)
            self._target_x = float(np.median(self._target_samples))
            if len(self._target_samples) >= 3:
                self._target_locked = (
                    max(self._target_samples) - min(self._target_samples) <= 8
                )

        marker = self._find_slider(crop, left, bar_scale)
        marker_x = marker[0] if marker is not None else None
        slider_score = marker[1] if marker is not None else None
        marker_advanced = False
        if marker_x is not None:
            if self._last_visual_x is not None and marker_x < self._last_visual_x - 80:
                self.reset(keep_target=True)
            if (
                self._last_visual_x is None
                or marker_x - self._last_visual_x >= 0.75
            ):
                self._samples.append((captured_at, marker_x))
                self._last_visual_x = marker_x
                marker_advanced = True

        speed = self._estimate_speed()
        if (
            speed is None
            and self._learned_speed is not None
            and len(self._samples) >= 2
        ):
            speed = self._learned_speed
        current_crossing_at = None
        if (
            marker_advanced
            and marker_x is not None
            and self._target_x is not None
            and speed is not None
        ):
            current_crossing_at = captured_at + (self._target_x - marker_x) / speed
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

        return self._observation(
            marker_x,
            self._target_x,
            speed,
            crossing_at,
            prediction_cached,
            bar_rect,
            slider_score,
        )

    def _observation(
        self,
        marker_x: float | None,
        target_x: float | None,
        speed: float | None,
        crossing_at: float | None,
        prediction_cached: bool,
        bar_rect: tuple[int, int, int, int],
        slider_score: float | None = None,
    ) -> RhythmObservation:
        return RhythmObservation(
            marker_x=marker_x,
            target_x=target_x,
            speed_px_per_second=speed,
            crossing_at=crossing_at,
            prediction_cached=prediction_cached,
            bar_rect=bar_rect,
            bar_match_score=self._bar_match_score,
            slider_match_score=slider_score,
            cursor_match_score=self._cursor_match_score,
            bar_locked=self._bar_locked,
        )

    def annotate(self, frame: np.ndarray, observation: RhythmObservation) -> np.ndarray:
        left, top, right, bottom = observation.bar_rect
        if right <= left or bottom <= top:
            return frame
        color = (60, 220, 60) if observation.bar_locked else (40, 190, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        if observation.target_x is not None:
            x = round(observation.target_x)
            cv2.line(frame, (x, top), (x, bottom), (255, 255, 255), 2)
        if observation.marker_x is not None:
            x = round(observation.marker_x)
            cv2.line(frame, (x, top), (x, bottom), (255, 220, 20), 2)
        return frame

    def _locate_bar(
        self, frame: np.ndarray
    ) -> tuple[tuple[int, int, int, int], float] | None:
        if self._bar_locked and self._bar_box is not None and self._bar_scale is not None:
            return self._padded_bar_rect(frame, self._bar_box), self._bar_scale

        height, width = frame.shape[:2]
        rx, ry, rw, rh = self.timing.bar_search_roi
        search_left = max(0, min(width - 1, round(width * rx)))
        search_top = max(0, min(height - 1, round(height * ry)))
        search_right = min(width, max(search_left + 1, search_left + round(width * rw)))
        search_bottom = min(
            height, max(search_top + 1, search_top + round(height * rh))
        )
        search = cv2.cvtColor(
            frame[search_top:search_bottom, search_left:search_right],
            cv2.COLOR_BGR2GRAY,
        )
        template_gray = cv2.cvtColor(self._bar_template, cv2.COLOR_BGR2GRAY)
        nominal_scale = width * self.timing.bar_width_ratio / template_gray.shape[1]
        best: tuple[float, tuple[int, int, int, int], float, float] | None = None
        for factor in self.timing.bar_scale_factors:
            scale = nominal_scale * factor
            template_width = max(2, round(template_gray.shape[1] * scale))
            template_height = max(2, round(template_gray.shape[0] * scale))
            if template_width > search.shape[1] or template_height > search.shape[0]:
                continue
            template = cv2.resize(
                template_gray,
                (template_width, template_height),
                interpolation=cv2.INTER_LINEAR,
            )
            result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(result)
            adjusted_score = float(score) - abs(factor - 1.0) * 0.08
            box = (
                search_left + location[0],
                search_top + location[1],
                search_left + location[0] + template_width,
                search_top + location[1] + template_height,
            )
            if best is None or adjusted_score > best[0]:
                best = (adjusted_score, box, scale, float(score))

        if best is not None:
            _, box, scale, raw_score = best
            self._bar_match_score = raw_score
            if raw_score >= self.timing.bar_match_threshold:
                self._accept_bar_candidate(box, scale, raw_score)

        if self._bar_box is None or self._bar_scale is None:
            return None
        return self._padded_bar_rect(frame, self._bar_box), self._bar_scale

    def _accept_bar_candidate(
        self, box: tuple[int, int, int, int], scale: float, score: float
    ) -> None:
        if self._bar_candidates:
            previous = self._bar_candidates[-1]
            if abs(box[0] - previous[0]) > 18 or abs(box[1] - previous[1]) > 10:
                self._bar_candidates.clear()
        self._bar_candidates.append((*box, score))
        self._bar_box = box
        self._bar_scale = scale
        if len(self._bar_candidates) >= 3:
            boxes = list(self._bar_candidates)
            self._bar_box = tuple(
                round(float(np.median([item[index] for item in boxes])))
                for index in range(4)
            )
            self._bar_match_score = float(np.median([item[4] for item in boxes]))
            self._bar_locked = True

    def _padded_bar_rect(
        self, frame: np.ndarray, box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        left, top, right, bottom = box
        padding = round((bottom - top) * self.timing.bar_vertical_padding)
        return left, max(0, top - padding), right, min(frame.shape[0], bottom + padding)

    def _find_cursor(
        self, crop: np.ndarray, offset_x: int, bar_scale: float
    ) -> tuple[float, float] | None:
        if self._target_locked or crop.size == 0:
            return None
        search_start = round(
            crop.shape[1]
            * (self.timing.cursor_prior_ratio - self.timing.cursor_search_radius_ratio)
        )
        search_end = round(
            crop.shape[1]
            * (self.timing.cursor_prior_ratio + self.timing.cursor_search_radius_ratio)
        )
        search_start = max(0, search_start)
        search_end = min(crop.shape[1], max(search_start + 1, search_end))
        result = self._match_templates(
            crop[:, search_start:search_end], self._cursor_templates, bar_scale
        )
        if result is None or result[0] < self.timing.cursor_match_threshold:
            return None
        score, center_x = result
        return offset_x + search_start + center_x, score

    def _find_slider(
        self, crop: np.ndarray, offset_x: int, bar_scale: float
    ) -> tuple[float, float] | None:
        if crop.size == 0:
            return None
        search_right = crop.shape[1]
        if self._target_x is not None:
            search_right = min(
                search_right, max(1, round(self._target_x - offset_x - 8 * bar_scale))
            )
        result = self._match_templates(
            crop[:, :search_right], self._slider_templates, bar_scale
        )
        if result is None or result[0] < self.timing.slider_match_threshold:
            return None
        score, center_x = result
        marker_x = offset_x + center_x
        if self._last_visual_x is not None:
            distance = marker_x - self._last_visual_x
            if distance < -80 or distance > 100:
                return None
        return marker_x, score

    def _match_templates(
        self, image: np.ndarray, templates: list[np.ndarray], bar_scale: float
    ) -> tuple[float, float] | None:
        if image.size == 0:
            return None
        source = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        best: tuple[float, float] | None = None
        for base in templates:
            base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            for factor in self.timing.feature_scale_factors:
                scale = bar_scale * factor
                width = max(2, round(base_gray.shape[1] * scale))
                height = max(2, round(base_gray.shape[0] * scale))
                if width > source.shape[1] or height > source.shape[0]:
                    continue
                template = cv2.resize(
                    base_gray, (width, height), interpolation=cv2.INTER_LINEAR
                )
                result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, location = cv2.minMaxLoc(result)
                adjusted_score = float(score) - abs(factor - 1.0) * 0.04
                center_x = float(location[0] + width / 2)
                if best is None or adjusted_score > best[0]:
                    best = (adjusted_score, center_x)
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

from __future__ import annotations

import logging
import time
import ctypes
from collections import deque

import cv2
import dxcam
import numpy as np

from .config import (
    PROJECT_ROOT,
    changed_config_paths,
    create_log_path,
    load_config,
    normalize_topmost_mode,
    resolve_mode_config,
    save_config,
    topmost_enabled_for_state,
)
from .detector import ArrowDetector
from .hotkeys import GlobalHotkeys
from .keyboard_input import (
    DirectionKeySender,
    InputTiming,
    SpaceKeySender,
    foreground_window,
    is_admin,
    relaunch_as_admin,
    window_client_rect,
    window_monitor_rect,
    window_title,
)
from .recorder import RoiVideoRecorder
from .selector import ModeSelector
from .space_timing import SliderObservation, SliderTracker, SpaceTimingConfig
from .yolo_dataset import YoloDatasetCollector
from .yolo_detector import (
    YoloArrowDetector,
    YoloFrameDetections,
    YoloRuntimeSettings,
    yolo_model_signature,
)


WINDOW_NAME = "Dance Arrow Recognition - Q to quit"
LOGGER = logging.getLogger("dance_tool.runtime")


def space_expire_reason(
    armed_at: float | None, now: float, timeout_ms: float
) -> str | None:
    """Expire only by time; arrow visibility may flicker while keys change state."""
    if armed_at is None:
        return None
    age_ms = (now - armed_at) * 1000
    if age_ms >= timeout_ms:
        return f"timeout {age_ms:.0f}ms"
    return None


def frame_limit_delay(
    frame_started_at: float,
    maximum_fps: float,
    now: float | None = None,
) -> float:
    """Return the remaining delay needed to keep the live loop under its FPS cap."""
    if maximum_fps <= 0:
        return 0.0
    current = time.perf_counter() if now is None else now
    target_period = 1.0 / maximum_fps
    return max(0.0, target_period - (current - frame_started_at))


def wait_for_frame_limit(frame_started_at: float, maximum_fps: float) -> None:
    """Use coarse sleep plus a short precision wait to avoid Windows oversleep."""
    delay = frame_limit_delay(frame_started_at, maximum_fps)
    if delay <= 0:
        return
    precision_window = 0.002
    if delay > precision_window:
        time.sleep(delay - precision_window)
    deadline = frame_started_at + 1.0 / maximum_fps
    while time.perf_counter() < deadline:
        pass


def configure_runtime_logging() -> None:
    log_path = create_log_path()
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%H:%M:%S"))
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


class PreviewWindow:
    def __init__(self, scale: float, always_on_top: bool, mouse_callback=None):
        self.scale = min(max(float(scale), 0.2), 1.0)
        self.always_on_top = bool(always_on_top)
        self._created = False
        self._source_size: tuple[int, int] | None = None
        self.mouse_callback = mouse_callback

    def show(self, image: np.ndarray) -> None:
        source_size = (image.shape[1], image.shape[0])
        if not self._created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            if self.mouse_callback is not None:
                cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
            self._created = True

        cv2.imshow(WINDOW_NAME, image)
        if self._source_size != source_size:
            # The recognition image is already scaled before the fixed-size text
            # header is drawn, so the native window size keeps text readable.
            cv2.resizeWindow(WINDOW_NAME, source_size[0], source_size[1])
            self._source_size = source_size
        self.apply_topmost()

    def apply_topmost(self) -> None:
        if not self._created:
            return
        try:
            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_TOPMOST,
                1.0 if self.always_on_top else 0.0,
            )
        except cv2.error:
            # Some OpenCV Windows builds do not expose the TOPMOST property.
            pass

    def minimize(self) -> bool:
        if not self._created:
            return False
        hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_NAME)
        if not hwnd:
            return False
        return bool(ctypes.windll.user32.ShowWindow(hwnd, 6))  # SW_MINIMIZE


class ScreenCapture:
    def __init__(self, monitor_number: int):
        if monitor_number < 1:
            raise ValueError("monitor 必须从 1 开始编号")
        self.camera = dxcam.create(output_idx=monitor_number - 1, output_color="BGR")
        self.width = int(self.camera.width)
        self.height = int(self.camera.height)
        self._last_frame: np.ndarray | None = None

    def close(self) -> None:
        self.camera.release()

    def grab(self, region: tuple[int, int, int, int] | None = None) -> np.ndarray:
        frame = self.camera.grab(region=region)
        if frame is None:
            for _ in range(5):
                time.sleep(0.01)
                frame = self.camera.grab(region=region)
                if frame is not None:
                    break
        if frame is None and self._last_frame is not None:
            if region is None or self._last_frame.shape[:2] == (
                region[3] - region[1],
                region[2] - region[0],
            ):
                return self._last_frame.copy()
        if frame is None:
            raise RuntimeError("暂时无法取得屏幕画面")
        self._last_frame = frame.copy()
        return frame


def relative_roi_to_region(
    screen_width: int, screen_height: int, roi: list[float]
) -> tuple[int, int, int, int]:
    x, y, width, height = roi
    left = max(0, round(screen_width * x))
    top = max(0, round(screen_height * y))
    right = min(screen_width, left + max(1, round(screen_width * width)))
    bottom = min(screen_height, top + max(1, round(screen_height * height)))
    return left, top, right, bottom


def automatic_window_roi(
    capture_width: int,
    capture_height: int,
    client_rect: tuple[int, int, int, int],
    monitor_rect: tuple[int, int, int, int],
    top_ratio: float,
    bottom_ratio: float,
) -> list[float]:
    """Build a full-client-width ROI using vertical ratios within the client area."""
    if capture_width <= 0 or capture_height <= 0:
        raise ValueError("截图尺寸无效")
    if not 0.0 <= top_ratio < bottom_ratio <= 1.0:
        raise ValueError("auto_roi.top_ratio 和 bottom_ratio 必须满足 0 <= top < bottom <= 1")
    client_left, client_top, client_right, client_bottom = client_rect
    monitor_left, monitor_top, monitor_right, monitor_bottom = monitor_rect
    client_width = client_right - client_left
    client_height = client_bottom - client_top
    monitor_width = monitor_right - monitor_left
    monitor_height = monitor_bottom - monitor_top
    if client_width <= 0 or client_height <= 0:
        raise ValueError("当前窗口客户区尺寸无效")
    if monitor_width <= 0 or monitor_height <= 0:
        raise ValueError("当前显示器尺寸无效")

    scale_x = capture_width / monitor_width
    scale_y = capture_height / monitor_height
    left = round((client_left - monitor_left) * scale_x)
    right = round((client_right - monitor_left) * scale_x)
    top = round(
        (client_top - monitor_top + client_height * top_ratio) * scale_y
    )
    bottom = round(
        (client_top - monitor_top + client_height * bottom_ratio) * scale_y
    )
    left = min(max(0, left), capture_width - 1)
    right = min(max(left + 1, right), capture_width)
    top = min(max(0, top), capture_height - 1)
    bottom = min(max(top + 1, bottom), capture_height)
    return [
        left / capture_width,
        top / capture_height,
        (right - left) / capture_width,
        (bottom - top) / capture_height,
    ]


def auto_select_arrow_roi(
    capture: ScreenCapture,
    config: dict,
    hwnd: int | None = None,
) -> bool:
    """Save the configured lower portion of the current window as the arrow ROI."""
    selected_hwnd = foreground_window() if hwnd is None else int(hwnd)
    title = window_title(selected_hwnd)
    if selected_hwnd == 0 or title == WINDOW_NAME:
        print("自动选取 ROI 失败：请先聚焦游戏窗口")
        return False
    required_title = str(
        config.get("input", {}).get("target_window_title_contains", "")
    ).strip()
    if required_title and required_title.casefold() not in title.casefold():
        print(f"自动选取 ROI 失败：窗口标题 {title!r} 不包含 {required_title!r}")
        return False

    values = config.get("auto_roi", {})
    top_ratio = float(values.get("top_ratio", 0.72))
    bottom_ratio = float(values.get("bottom_ratio", 0.93))
    try:
        roi = automatic_window_roi(
            capture.width,
            capture.height,
            window_client_rect(selected_hwnd),
            window_monitor_rect(selected_hwnd),
            top_ratio,
            bottom_ratio,
        )
    except (OSError, ValueError) as error:
        print(f"自动选取 ROI 失败：{error}")
        return False

    config["arrow_roi"] = roi
    save_config(config)
    region = relative_roi_to_region(capture.width, capture.height, roi)
    print(
        f"已按窗口自动设置 ROI：title={title!r}, "
        f"region={region}, vertical={top_ratio:.3f}-{bottom_ratio:.3f}"
    )
    return True


def select_arrow_roi(capture: ScreenCapture, config: dict) -> bool:
    screenshot = capture.grab()
    title = "Select arrow area, then press ENTER (ESC cancels)"
    x, y, width, height = (
        int(value) for value in cv2.selectROI(title, screenshot, False, False)
    )
    cv2.destroyWindow(title)
    if width <= 0 or height <= 0:
        print("已取消区域标定")
        return False

    config["arrow_roi"] = [
        x / capture.width,
        y / capture.height,
        width / capture.width,
        height / capture.height,
    ]
    save_config(config)
    print(f"箭头区域已保存：x={x}, y={y}, width={width}, height={height}")
    return True


def calibrate_arrow_roi() -> int:
    config = load_config()
    capture = ScreenCapture(int(config.get("monitor", 1)))
    try:
        return 0 if select_arrow_roi(capture, config) else 1
    finally:
        capture.close()


def draw_status(
    frame: np.ndarray,
    state: str,
    fps: float,
    stable_count: int,
    stable_required: int,
    stable_sequence: tuple[str, ...],
    always_on_top: bool,
    input_status: str,
    message: str | None,
    recording_status: str | None = None,
    display_scale: float = 1.0,
    space_status: str = "OFF",
    ui_mode: str = "",
    game_mode: str = "",
    selector: ModeSelector | None = None,
    yolo_enabled: bool = False,
    dataset_split: str = "train",
    dataset_split_enabled: bool = True,
    topmost_mode: str = "off",
) -> np.ndarray:
    scale = min(max(float(display_scale), 0.2), 1.0)
    display_width = max(1, round(frame.shape[1] * scale))
    display_height = max(1, round(frame.shape[0] * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    scaled_frame = cv2.resize(frame, (display_width, display_height), interpolation=interpolation)

    # This header is added after image scaling, so its text remains at a fixed,
    # readable pixel size even when the recognition image is displayed at 55%.
    header_height = 243
    canvas = np.zeros(
        (scaled_frame.shape[0] + header_height, scaled_frame.shape[1], 3),
        dtype=np.uint8,
    )
    canvas[header_height:] = scaled_frame
    color = {"RUNNING": (60, 230, 80), "STOPPED": (80, 80, 240)}[state]
    topmost_text = "ON" if always_on_top else "OFF"
    cv2.putText(
        canvas,
        f"{state}  FPS {fps:.1f}  TOP {topmost_text}  INPUT {input_status}",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        1,
    )
    cv2.putText(
        canvas,
        "Ctrl+F9 ROI | Ctrl+F10 Start | Ctrl+F11 Stop",
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    short_direction = {"UP": "U", "DOWN": "D", "LEFT": "L", "RIGHT": "R"}
    sequence_text = " ".join(short_direction[item] for item in stable_sequence) if stable_sequence else "-"
    cv2.putText(
        canvas,
        f"Stable: {stable_count}/{stable_required}   Sequence: {sequence_text}",
        (12, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Space: {space_status}",
        (12, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 220, 80),
        1,
        cv2.LINE_AA,
    )
    if selector is not None:
        selector.draw(
            canvas,
            state,
            game_mode,
            ui_mode,
            yolo_enabled,
            dataset_split,
            dataset_split_enabled,
            topmost_mode,
        )

    footer = recording_status or message or "Ctrl+= Rhythm shots | Ctrl+F12 ROI video"
    if footer:
        cv2.putText(
            canvas,
            footer[:80],
            (12, 236),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (80, 180, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def build_profile_runtime(config: dict):
    if not bool(config.get("implemented", False)):
        return None, 1, None, None, None
    detector = ArrowDetector(config)
    stable_required = int(config["recognition"]["stable_frames"])
    space_timing = SpaceTimingConfig.from_config(config.get("space", {}))
    if not space_timing.enabled:
        return detector, stable_required, space_timing, None, None
    space_tracker = SliderTracker(space_timing)
    space_sender = SpaceKeySender(
        space_timing.key_hold_min_ms,
        space_timing.key_hold_max_ms,
    )
    return detector, stable_required, space_timing, space_tracker, space_sender


def run_live(
    game_mode: str | None = None,
    ui_mode: str | None = None,
) -> int:
    saved_config = load_config()
    config = resolve_mode_config(saved_config, game_mode, ui_mode)
    print(
        f"当前模式：{config.get('name', config['game_mode'])} "
        f"({config['game_mode']} / {config['ui_mode']})"
    )
    input_config = config["input"]
    input_enabled = bool(input_config.get("enabled", True))
    elevation_failed = False
    if input_enabled and bool(input_config.get("auto_elevate", True)) and not is_admin():
        print("游戏以管理员权限运行，正在请求 UAC 权限并重新启动 Demo……")
        if relaunch_as_admin(str(PROJECT_ROOT)):
            return 0
        elevation_failed = True
        print("未获得管理员权限，方向键输入仍可能被游戏忽略。")

    configure_runtime_logging()
    if elevation_failed:
        LOGGER.error("elevation_request_failed_or_cancelled")

    if not saved_config.get("arrow_roi"):
        print("尚未标定箭头区域，正在打开区域选择窗口……")
        if calibrate_arrow_roi() != 0:
            return 1
        saved_config = load_config()
        config = resolve_mode_config(saved_config, game_mode, ui_mode)

    input_timing = InputTiming.from_config(input_config)
    key_sender = DirectionKeySender(input_timing)
    (
        opencv_detector,
        stable_required,
        space_timing,
        space_tracker,
        space_sender,
    ) = build_profile_runtime(config)
    detector = opencv_detector
    yolo_detector: YoloArrowDetector | None = None
    yolo_enabled = bool(config.get("yolo", {}).get("enabled", False))
    yolo_startup_error: str | None = None
    yolo_preload_started = time.perf_counter()
    try:
        yolo_detector = YoloArrowDetector(config.get("yolo", {}))
        yolo_detector.warmup()
    except Exception as error:
        yolo_detector = None
        if yolo_enabled:
            yolo_enabled = False
            saved_config.setdefault("yolo", {})["enabled"] = False
            config.setdefault("yolo", {})["enabled"] = False
            save_config(saved_config)
        yolo_startup_error = f"YOLO preload failed; using OpenCV: {error}"
        LOGGER.exception("yolo_preload_failed")
    else:
        LOGGER.info(
            "yolo_preloaded model=%s elapsed_ms=%d",
            yolo_detector.model_path,
            round((time.perf_counter() - yolo_preload_started) * 1000),
        )
        if yolo_enabled:
            detector = yolo_detector
    recorder = RoiVideoRecorder(config.get("recording", {}))
    window_config = config.setdefault(
        "window", {"scale": 0.55, "topmost_mode": "off"}
    )
    saved_window_config = saved_config.setdefault("window", {})
    topmost_mode = normalize_topmost_mode(window_config)
    window_config["topmost_mode"] = topmost_mode
    window_config.pop("always_on_top", None)
    save_window_config = False
    if (
        saved_window_config.get("topmost_mode") != topmost_mode
        or "always_on_top" in saved_window_config
    ):
        saved_window_config["topmost_mode"] = topmost_mode
        saved_window_config.pop("always_on_top", None)
        save_window_config = True
    if "max_fps" not in saved_window_config:
        saved_window_config["max_fps"] = 60
        save_window_config = True
    if save_window_config:
        save_config(saved_config)
    window_config.setdefault("max_fps", saved_window_config["max_fps"])
    maximum_fps = max(0.0, float(window_config["max_fps"]))
    LOGGER.info(
        "live_start game_mode=%s ui_mode=%s implemented=%s admin=%s input_enabled=%s "
        "reaction=%d-%d hold=%d-%d interval=%d-%d max_fps=%g",
        config["game_mode"],
        config["ui_mode"],
        config.get("implemented", False),
        is_admin(),
        input_enabled,
        input_timing.reaction_min_ms,
        input_timing.reaction_max_ms,
        input_timing.key_hold_min_ms,
        input_timing.key_hold_max_ms,
        input_timing.inter_key_min_ms,
        input_timing.inter_key_max_ms,
        maximum_fps,
    )
    selector = ModeSelector()
    preview = PreviewWindow(
        scale=float(window_config.get("scale", 0.55)),
        always_on_top=topmost_enabled_for_state(topmost_mode, "STOPPED"),
        mouse_callback=selector.on_mouse,
    )
    hotkeys = GlobalHotkeys(config["hotkeys"])
    hotkeys.start()

    state = "STOPPED"
    history: deque[tuple[str, ...]] = deque(maxlen=stable_required)
    stable_sequence: tuple[str, ...] = ()
    last_frame: np.ndarray | None = None
    status_message: str | None = yolo_startup_error
    frame_times: deque[float] = deque(maxlen=30)
    previous_frame_started_at: float | None = None
    target_hwnd = 0
    target_title = ""
    round_armed = True
    first_seen_at: float | None = None
    reaction_delay_seconds: float | None = None
    clear_frames = 0
    transition_seen = False
    last_sent_sequence: tuple[str, ...] = ()
    last_input_completed_at: float | None = None
    last_observed_sequence: tuple[str, ...] = ()
    input_status = "DISABLED" if not input_enabled else "READY"
    record_requested = False
    rhythm_capture_enabled = False
    rhythm_capture_collector: YoloDatasetCollector | None = None
    rhythm_capture_due_at: float | None = None
    rhythm_capture_interval = max(
        0.1,
        float(config.get("dataset", {}).get("rhythm_capture_interval_seconds", 0.2)),
    )
    space_armed = False
    space_scheduled = False
    space_tracking_active = False
    space_offset_ms: float | None = None
    space_armed_at: float | None = None
    space_status = (
        "WAIT DIRECTIONS"
        if space_timing is not None and space_timing.enabled
        else "OFF"
    )

    def reset_round() -> None:
        nonlocal round_armed, first_seen_at, reaction_delay_seconds
        nonlocal clear_frames, transition_seen, input_status
        round_armed = True
        first_seen_at = None
        reaction_delay_seconds = None
        clear_frames = 0
        transition_seen = False
        input_status = "DISABLED" if not input_enabled else "READY"

    def cancel_space_cycle(*, clear_target: bool = False) -> None:
        nonlocal space_armed, space_scheduled, space_tracking_active
        nonlocal space_offset_ms, space_status
        nonlocal space_armed_at
        if space_sender is not None:
            space_sender.cancel()
        if space_tracker is not None:
            space_tracker.reset(keep_target=not clear_target)
        space_armed = False
        space_scheduled = False
        space_tracking_active = False
        space_offset_ms = None
        space_armed_at = None
        space_status = (
            "WAIT DIRECTIONS"
            if space_timing is not None and space_timing.enabled
            else "OFF"
        )

    def bind_foreground_target() -> bool:
        nonlocal target_hwnd, target_title, status_message
        hwnd = foreground_window()
        title = window_title(hwnd)
        required_title = str(input_config.get("target_window_title_contains", "")).strip()
        if hwnd == 0 or title == WINDOW_NAME:
            status_message = "Focus game window, then press Ctrl+F10"
            LOGGER.warning("target_bind_rejected hwnd=%s title=%r", hwnd, title)
            return False
        if required_title and required_title.casefold() not in title.casefold():
            status_message = "Foreground window does not match configured target"
            print(f"前台窗口不匹配：实际={title!r}，要求包含={required_title!r}")
            LOGGER.warning(
                "target_title_mismatch hwnd=%s actual=%r required=%r",
                hwnd,
                title,
                required_title,
            )
            return False
        target_hwnd = hwnd
        target_title = title
        status_message = "Target window bound"
        print(f"已绑定目标窗口：{target_title!r}")
        LOGGER.info("target_bound hwnd=%s title=%r", target_hwnd, target_title)
        return True

    capture = ScreenCapture(int(config.get("monitor", 1)))
    region = relative_roi_to_region(capture.width, capture.height, config["arrow_roi"])
    try:
        while True:
            frame_started_at = time.perf_counter()
            if previous_frame_started_at is not None:
                frame_times.append(
                    max(frame_started_at - previous_frame_started_at, 1e-6)
                )
            previous_frame_started_at = frame_started_at
            for selection in selector.poll():
                if selection.kind == "config_reload":
                    if state != "STOPPED":
                        status_message = "Stop recognition before reloading JSON"
                        continue
                    next_capture = None
                    next_profile_runtime = None
                    try:
                        next_saved_config = load_config()
                        next_config = resolve_mode_config(
                            next_saved_config,
                            config["game_mode"],
                            config["ui_mode"],
                        )
                        normalized_window = next_config.setdefault("window", {})
                        next_topmost_mode = normalize_topmost_mode(
                            normalized_window
                        )
                        normalized_window["topmost_mode"] = next_topmost_mode
                        normalized_window.pop("always_on_top", None)
                        changed_paths = changed_config_paths(config, next_config)
                        if not changed_paths:
                            status_message = "JSON parameters unchanged"
                            continue

                        changed_roots = {
                            path.split(".", 1)[0] for path in changed_paths
                        }
                        rebuild_profile = bool(
                            changed_roots
                            & {"implemented", "templates", "recognition", "space"}
                        )
                        if rebuild_profile:
                            next_profile_runtime = build_profile_runtime(next_config)
                        else:
                            next_profile_runtime = (
                                opencv_detector,
                                stable_required,
                                space_timing,
                                space_tracker,
                                space_sender,
                            )

                        rebuild_input = "input" in changed_roots
                        next_input_config = next_config["input"]
                        next_input_enabled = bool(
                            next_input_config.get("enabled", True)
                        )
                        if rebuild_input:
                            next_input_timing = InputTiming.from_config(
                                next_input_config
                            )
                            next_key_sender = DirectionKeySender(next_input_timing)
                        else:
                            next_input_timing = input_timing
                            next_key_sender = key_sender

                        next_yolo_detector = yolo_detector
                        next_yolo_values = next_config.get("yolo", {})
                        next_yolo_enabled = bool(
                            next_yolo_values.get("enabled", False)
                        )
                        next_yolo_runtime = YoloRuntimeSettings.from_config(
                            next_yolo_values
                        )
                        next_yolo_signature = yolo_model_signature(next_yolo_values)
                        current_yolo_signature = (
                            (
                                yolo_detector.model_path,
                                yolo_detector.device,
                                yolo_detector.quantize,
                            )
                            if yolo_detector is not None
                            else None
                        )
                        rebuild_yolo = (
                            next_yolo_enabled and next_yolo_detector is None
                        ) or (
                            "yolo" in changed_roots
                            and next_yolo_signature != current_yolo_signature
                        )
                        if rebuild_yolo:
                            next_yolo_detector = YoloArrowDetector(
                                next_yolo_values
                            )
                            next_yolo_detector.warmup()

                        rebuild_hotkeys = "hotkeys" in changed_roots
                        next_hotkeys = (
                            GlobalHotkeys(next_config["hotkeys"])
                            if rebuild_hotkeys
                            else hotkeys
                        )

                        rebuild_capture = "monitor" in changed_roots
                        next_capture = (
                            ScreenCapture(int(next_config.get("monitor", 1)))
                            if rebuild_capture
                            else capture
                        )
                        next_region = relative_roi_to_region(
                            next_capture.width,
                            next_capture.height,
                            next_config["arrow_roi"],
                        )

                        rebuild_recorder = "recording" in changed_roots
                        if rebuild_recorder and recorder.active:
                            raise RuntimeError(
                                "请先用 Ctrl+F12 停止 ROI 录像，再刷新参数"
                            )
                        next_recorder = (
                            RoiVideoRecorder(next_config.get("recording", {}))
                            if rebuild_recorder
                            else recorder
                        )
                        next_rhythm_collector = rhythm_capture_collector
                        if rhythm_capture_enabled and "dataset" in changed_roots:
                            next_rhythm_collector = YoloDatasetCollector(
                                next_config.get("dataset", {})
                            )
                        next_rhythm_capture_interval = max(
                            0.1,
                            float(
                                next_config.get("dataset", {}).get(
                                    "rhythm_capture_interval_seconds", 0.2
                                )
                            ),
                        )
                        next_window_config = next_config.get("window", {})
                        next_maximum_fps = max(
                            0.0, float(next_window_config.get("max_fps", 60))
                        )
                        next_preview_scale = min(
                            max(float(next_window_config.get("scale", 0.55)), 0.2),
                            1.0,
                        )
                    except Exception as error:
                        if (
                            next_capture is not None
                            and next_capture is not capture
                        ):
                            next_capture.close()
                        if next_profile_runtime is not None:
                            candidate_space_sender = next_profile_runtime[4]
                            if (
                                candidate_space_sender is not None
                                and candidate_space_sender is not space_sender
                            ):
                                candidate_space_sender.close()
                        status_message = f"JSON reload failed: {error}"
                        LOGGER.exception("config_reload_failed")
                        continue

                    if rebuild_input:
                        key_sender.close()
                        key_sender = next_key_sender
                        input_timing = next_input_timing
                        input_config = next_input_config
                        input_enabled = next_input_enabled
                    if rebuild_profile:
                        if space_sender is not None:
                            space_sender.close()
                        (
                            opencv_detector,
                            stable_required,
                            space_timing,
                            space_tracker,
                            space_sender,
                        ) = next_profile_runtime
                    if rebuild_hotkeys:
                        hotkeys.close()
                        next_hotkeys.start()
                        hotkeys = next_hotkeys
                    if rebuild_capture:
                        capture.close()
                        capture = next_capture
                    if rebuild_recorder:
                        recorder = next_recorder

                    saved_config = next_saved_config
                    config = next_config
                    yolo_detector = next_yolo_detector
                    yolo_enabled = next_yolo_enabled
                    if yolo_detector is not None and not rebuild_yolo:
                        yolo_detector.apply_runtime_settings(next_yolo_runtime)
                    detector = yolo_detector if yolo_enabled else opencv_detector
                    region = next_region
                    rhythm_capture_collector = next_rhythm_collector
                    rhythm_capture_interval = next_rhythm_capture_interval
                    window_config = config.setdefault("window", {})
                    topmost_mode = next_topmost_mode
                    maximum_fps = next_maximum_fps
                    preview.scale = next_preview_scale
                    preview.always_on_top = topmost_enabled_for_state(
                        topmost_mode, state
                    )
                    preview.apply_topmost()
                    history = deque(maxlen=stable_required)
                    stable_sequence = ()
                    last_observed_sequence = ()
                    last_sent_sequence = ()
                    last_input_completed_at = None
                    last_frame = None
                    target_hwnd = 0
                    target_title = ""
                    reset_round()
                    cancel_space_cycle(clear_target=True)
                    changed_summary = ", ".join(sorted(changed_roots))
                    status_message = f"JSON reloaded: {changed_summary}"
                    LOGGER.info(
                        "config_reloaded paths=%s",
                        sorted(changed_paths),
                    )
                    continue
                if selection.kind == "topmost_mode":
                    previous_topmost_mode = topmost_mode
                    topmost_mode = str(selection.value)
                    saved_window = saved_config.setdefault("window", {})
                    saved_window["topmost_mode"] = topmost_mode
                    saved_window.pop("always_on_top", None)
                    active_window = config.setdefault("window", {})
                    active_window["topmost_mode"] = topmost_mode
                    active_window.pop("always_on_top", None)
                    preview.always_on_top = topmost_enabled_for_state(
                        topmost_mode, state
                    )
                    preview.apply_topmost()
                    try:
                        save_config(saved_config)
                    except OSError as error:
                        topmost_mode = previous_topmost_mode
                        saved_window["topmost_mode"] = topmost_mode
                        active_window["topmost_mode"] = topmost_mode
                        preview.always_on_top = topmost_enabled_for_state(
                            topmost_mode, state
                        )
                        preview.apply_topmost()
                        status_message = f"Topmost mode save failed: {error}"
                        LOGGER.exception("topmost_mode_save_failed")
                    else:
                        status_message = f"Topmost mode: {topmost_mode.upper()}"
                        LOGGER.info("topmost_mode_changed mode=%s", topmost_mode)
                    continue
                if selection.kind == "dataset_split":
                    if rhythm_capture_enabled:
                        status_message = "Stop rhythm screenshots before switching TRAIN/VAL"
                        continue
                    next_split = "val" if selection.value == "val" else "train"
                    dataset_settings = saved_config.setdefault("dataset", {})
                    previous_split = str(dataset_settings.get("split", "train"))
                    dataset_settings["split"] = next_split
                    config.setdefault("dataset", {})["split"] = next_split
                    try:
                        save_config(saved_config)
                    except OSError as error:
                        dataset_settings["split"] = previous_split
                        config["dataset"]["split"] = previous_split
                        status_message = f"Dataset split failed: {error}"
                        LOGGER.exception("dataset_split_failed")
                    else:
                        status_message = f"Dataset split: {next_split.upper()}"
                        LOGGER.info("dataset_split_changed split=%s", next_split)
                    continue
                if selection.kind == "yolo_toggle":
                    if state != "STOPPED":
                        status_message = "Stop recognition before switching YOLO"
                        continue
                    enable_yolo = selection.value == "true"
                    yolo_settings = saved_config.setdefault("yolo", {})
                    yolo_settings["enabled"] = enable_yolo
                    config.setdefault("yolo", {})["enabled"] = enable_yolo
                    if enable_yolo:
                        if yolo_detector is None:
                            try:
                                yolo_detector = YoloArrowDetector(yolo_settings)
                                yolo_detector.warmup()
                            except Exception as error:
                                yolo_detector = None
                                yolo_enabled = False
                                yolo_settings["enabled"] = False
                                config["yolo"]["enabled"] = False
                                detector = opencv_detector
                                status_message = f"YOLO enable failed: {error}"
                                LOGGER.exception("yolo_enable_failed")
                        if yolo_detector is not None:
                            yolo_enabled = True
                            detector = yolo_detector
                            status_message = "Detector: YOLO arrows + slider (preloaded)"
                            LOGGER.info(
                                "arrow_detector_switched backend=yolo model=%s",
                                yolo_detector.model_path,
                            )
                    else:
                        yolo_enabled = False
                        detector = opencv_detector
                        status_message = "Detector: OpenCV"
                        LOGGER.info("arrow_detector_switched backend=opencv")
                    save_config(saved_config)
                    history.clear()
                    stable_sequence = ()
                    last_observed_sequence = ()
                    reset_round()
                    cancel_space_cycle(clear_target=True)
                    continue
                if selection.kind == "blocked":
                    if selection.value == "yolo":
                        status_message = "Stop recognition before switching YOLO"
                    elif selection.value == "dataset_split":
                        status_message = (
                            "Stop rhythm screenshots before switching TRAIN/VAL"
                        )
                    elif selection.value == "config_reload":
                        status_message = "Stop recognition before reloading JSON"
                    else:
                        status_message = "Stop recognition before switching mode"
                    continue
                if state != "STOPPED":
                    status_message = "Stop recognition before switching mode"
                    continue
                selected_game = (
                    selection.value
                    if selection.kind == "game_mode"
                    else config["game_mode"]
                )
                selected_ui = (
                    selection.value
                    if selection.kind == "ui_mode"
                    else config["ui_mode"]
                )
                try:
                    next_config = resolve_mode_config(
                        saved_config,
                        selected_game,
                        selected_ui,
                    )
                    next_runtime = build_profile_runtime(next_config)
                except (OSError, ValueError) as error:
                    status_message = f"Switch failed: {error}"
                    LOGGER.exception("profile_switch_failed")
                    continue

                if space_sender is not None:
                    space_sender.close()
                config = next_config
                (
                    opencv_detector,
                    stable_required,
                    space_timing,
                    space_tracker,
                    space_sender,
                ) = next_runtime
                detector = yolo_detector if yolo_enabled else opencv_detector
                saved_config["game_mode"] = config["game_mode"]
                saved_config["ui_mode"] = config["ui_mode"]
                save_config(saved_config)
                history = deque(maxlen=stable_required)
                stable_sequence = ()
                last_observed_sequence = ()
                last_sent_sequence = ()
                last_input_completed_at = None
                target_hwnd = 0
                target_title = ""
                reset_round()
                cancel_space_cycle(clear_target=True)
                if config.get("implemented", False):
                    status_message = (
                        f"Selected: {config['game_mode']} / {config['ui_mode']}"
                    )
                else:
                    status_message = "Selected mode needs recognition assets and rules"
                    print(config.get("unavailable_reason", "所选模式尚未配置"))
                LOGGER.info(
                    "profile_switched game_mode=%s ui_mode=%s implemented=%s",
                    config["game_mode"],
                    config["ui_mode"],
                    config.get("implemented", False),
                )

            for event, message in hotkeys.poll():
                if event == "select_roi":
                    was_running = state == "RUNNING"
                    state = "STOPPED"
                    key_sender.cancel()
                    cancel_space_cycle(clear_target=True)
                    if recorder.active:
                        saved_path = recorder.stop()
                        status_message = f"Recording stopped: {saved_path.name}"
                        LOGGER.warning("recording_stopped_for_roi_change path=%s", saved_path)
                    if auto_select_arrow_roi(capture, saved_config):
                        config["arrow_roi"] = saved_config["arrow_roi"]
                        region = relative_roi_to_region(
                            capture.width, capture.height, saved_config["arrow_roi"]
                        )
                        history.clear()
                        stable_sequence = ()
                        last_observed_sequence = ()
                        last_frame = None
                        status_message = "ROI automatically updated from game window"
                    else:
                        status_message = "Auto ROI failed; focus game and press Ctrl+F9"
                    if was_running:
                        target_hwnd = 0
                        reset_round()
                        if status_message.startswith("ROI automatically"):
                            status_message = "ROI updated; focus game and press Ctrl+F10"
                elif event == "start":
                    if not config.get("implemented", False):
                        status_message = "Selected mode needs recognition assets and rules"
                        print(config.get("unavailable_reason", "所选模式尚未配置"))
                        continue
                    key_sender.cancel()
                    cancel_space_cycle(clear_target=True)
                    if bind_foreground_target():
                        state = "RUNNING"
                        history.clear()
                        stable_sequence = ()
                        last_observed_sequence = ()
                        reset_round()
                elif event == "stop":
                    was_running = state == "RUNNING"
                    state = "STOPPED"
                    key_sender.cancel()
                    cancel_space_cycle()
                    history.clear()
                    stable_sequence = ()
                    last_observed_sequence = ()
                    target_hwnd = 0
                    target_title = ""
                    reset_round()
                    if was_running:
                        preview.minimize()
                elif event == "record_roi":
                    if recorder.active:
                        saved_path = recorder.stop()
                        status_message = f"Recording saved: {saved_path.name}"
                        LOGGER.info("recording_stopped_by_hotkey path=%s", saved_path)
                    elif record_requested:
                        status_message = "ROI recording is starting"
                    else:
                        record_requested = True
                        status_message = "Starting ROI recording..."
                elif event == "capture_rhythm_dataset":
                    if rhythm_capture_enabled:
                        rhythm_capture_enabled = False
                        rhythm_capture_due_at = None
                        saved_count = (
                            rhythm_capture_collector.saved_count
                            if rhythm_capture_collector is not None
                            else 0
                        )
                        status_message = (
                            f"Rhythm screenshots OFF ({saved_count} saved)"
                        )
                        LOGGER.info(
                            "rhythm_capture_toggled enabled=false saved=%d",
                            saved_count,
                        )
                    else:
                        if (
                            yolo_detector is None
                            or not yolo_detector.supports_slider_class
                        ):
                            status_message = (
                                "Rhythm screenshots require a YOLO model with slider"
                            )
                            continue
                        capture_settings = dict(config.get("dataset", {}))
                        try:
                            rhythm_capture_collector = YoloDatasetCollector(
                                capture_settings
                            )
                        except (OSError, ValueError) as error:
                            rhythm_capture_collector = None
                            status_message = f"Rhythm screenshots failed: {error}"
                            LOGGER.exception("rhythm_capture_enable_failed")
                        else:
                            rhythm_capture_enabled = True
                            rhythm_capture_due_at = (
                                time.perf_counter() + rhythm_capture_interval
                            )
                            status_message = (
                                "Rhythm screenshots ON "
                                f"({rhythm_capture_interval:g}s)"
                            )
                            LOGGER.info(
                                "rhythm_capture_toggled enabled=true split=%s interval=%s",
                                rhythm_capture_collector.split,
                                rhythm_capture_interval,
                            )
                elif event == "error":
                    status_message = message

            for event, payload in key_sender.poll():
                if event == "started":
                    input_status = "SENDING"
                    actual_ms = payload.get("actual_reaction_ms")
                    print(
                        f"开始输入：{' '.join(payload['sequence'])}；"
                        f"实际首键等待={actual_ms}ms"
                    )
                    LOGGER.info(
                        "input_started sequence=%r actual_reaction_ms=%s target_hwnd=%s",
                        payload["sequence"],
                        actual_ms,
                        target_hwnd,
                    )
                elif event == "completed":
                    input_status = "SENT"
                    last_input_completed_at = time.perf_counter()
                    print(f"方向键输入完成：{payload}")
                    LOGGER.info("input_completed payload=%r", payload)
                    if (
                        space_timing is not None
                        and space_timing.enabled
                        and space_tracking_active
                        and state == "RUNNING"
                        and target_hwnd
                    ):
                        space_armed = True
                        space_offset_ms = space_timing.sample_offset_ms()
                        space_armed_at = time.perf_counter()
                        space_status = f"TRACKING offset={space_offset_ms:+.1f}ms"
                        LOGGER.info("space_armed offset_ms=%.1f", space_offset_ms)
                elif event == "cancelled":
                    input_status = "CANCELLED"
                    LOGGER.warning("input_cancelled sequence=%r", payload)
                elif event == "error":
                    input_status = "ERROR"
                    status_message = str(payload)
                    LOGGER.error("input_error detail=%s", payload)

            for event, payload in (space_sender.poll() if space_sender is not None else []):
                if event == "started":
                    LOGGER.info("space_started payload=%r", payload)
                elif event == "completed":
                    space_status = (
                        f"PRESSED error={payload['predicted_crossing_error_ms']:+.1f}ms"
                    )
                    print(f"空格输入完成：{payload}")
                    LOGGER.info("space_completed payload=%r", payload)
                elif event == "cancelled":
                    LOGGER.warning("space_cancelled")
                elif event == "error":
                    space_status = "ERROR"
                    status_message = str(payload)
                    LOGGER.error("space_error detail=%s", payload)

            if state == "RUNNING" or recorder.active or record_requested or last_frame is None:
                frame = capture.grab(region)
                last_frame = frame
            else:
                frame = last_frame.copy()

            if record_requested:
                try:
                    recording_path = recorder.start(frame)
                    LOGGER.info(
                        "recording_started path=%s fps=%.1f size=%sx%s",
                        recording_path,
                        recorder.fps,
                        frame.shape[1],
                        frame.shape[0],
                    )
                    status_message = f"Recording: {recording_path.name}"
                except Exception as error:
                    status_message = f"Recording failed: {error}"
                    LOGGER.exception("recording_start_failed")
                finally:
                    record_requested = False

            if recorder.active:
                try:
                    recorder.add_frame(frame)
                except Exception as error:
                    recorder.stop()
                    status_message = f"Recording failed: {error}"
                    LOGGER.exception("recording_write_failed")

            captured_at = time.perf_counter()
            detections = []
            yolo_frame: YoloFrameDetections | None = None
            if state == "RUNNING" and detector is not None:
                try:
                    if yolo_enabled and yolo_detector is not None:
                        yolo_frame = yolo_detector.detect_frame(frame)
                        detections = list(yolo_frame.arrows)
                    else:
                        detections = detector.detect(frame)
                except Exception as error:
                    if not yolo_enabled:
                        raise
                    LOGGER.exception("yolo_inference_failed")
                    yolo_enabled = False
                    yolo_detector = None
                    saved_config.setdefault("yolo", {})["enabled"] = False
                    config.setdefault("yolo", {})["enabled"] = False
                    save_config(saved_config)
                    detector = opencv_detector
                    key_sender.cancel()
                    cancel_space_cycle()
                    history.clear()
                    stable_sequence = ()
                    last_observed_sequence = ()
                    reset_round()
                    status_message = f"YOLO failed; using OpenCV: {error}"
                    if detector is not None:
                        detections = detector.detect(frame)
            sequence = tuple(item.direction for item in detections)
            space_observation: SliderObservation | None = None
            if (
                state == "RUNNING"
                and space_timing is not None
                and space_tracker is not None
                and space_sender is not None
                and space_timing.enabled
                and space_tracking_active
            ):
                rhythm_slider = yolo_frame.slider if yolo_frame is not None else None
                space_observation = space_tracker.observe(
                    frame,
                    captured_at,
                    detected_slider_rect=(
                        rhythm_slider.box if rhythm_slider is not None else None
                    ),
                    detected_slider_score=(
                        rhythm_slider.score if rhythm_slider is not None else None
                    ),
                    allow_template_fallback=not (
                        yolo_enabled and yolo_frame is not None
                    ),
                )
                crossing_at = space_observation.crossing_at
                if (
                    space_armed
                    and not space_scheduled
                    and crossing_at is not None
                    and space_offset_ms is not None
                ):
                    desired_press_at = crossing_at + space_offset_ms / 1000.0
                    lead_ms = (desired_press_at - captured_at) * 1000
                    if lead_ms < -space_timing.late_tolerance_ms:
                        space_armed = False
                        space_tracking_active = False
                        space_status = f"MISSED {lead_ms:+.0f}ms"
                        LOGGER.warning(
                            "space_prediction_missed lead_ms=%.1f marker=%s target=%s speed=%s cached=%s",
                            lead_ms,
                            space_observation.marker_x,
                            space_observation.target_x,
                            space_observation.speed_px_per_second,
                            space_observation.prediction_cached,
                        )
                    elif lead_ms <= space_timing.prediction_horizon_ms:
                        press_at = max(
                            captured_at + space_timing.minimum_schedule_lead_ms / 1000.0,
                            desired_press_at,
                        )
                        if space_sender.start_at(
                            press_at,
                            target_hwnd,
                            predicted_crossing_at=crossing_at,
                            sampled_offset_ms=space_offset_ms,
                        ):
                            space_armed = False
                            space_scheduled = True
                            space_tracking_active = False
                            space_armed_at = None
                            space_status = f"SCHEDULED {lead_ms:+.0f}ms"
                            LOGGER.info(
                                "space_scheduled lead_ms=%.1f offset_ms=%.1f marker=%s target=%s speed=%s cached=%s",
                                lead_ms,
                                space_offset_ms,
                                space_observation.marker_x,
                                space_observation.target_x,
                                space_observation.speed_px_per_second,
                                space_observation.prediction_cached,
                            )
                elif space_armed:
                    if space_observation.marker_x is None:
                        space_status = (
                            "TRACKING (cached)"
                            if space_observation.prediction_cached
                            else "WAIT SLIDER"
                        )
                    else:
                        space_status = "TRACKING"

                if space_armed:
                    expire_reason = space_expire_reason(
                        space_armed_at,
                        captured_at,
                        space_timing.expire_after_directions_ms,
                    )
                    if expire_reason is not None:
                        space_armed = False
                        space_tracking_active = False
                        space_offset_ms = None
                        space_status = "EXPIRED"
                        LOGGER.warning(
                            "space_round_expired reason=%s slider_score=%s marker=%s "
                            "target=%s speed=%s crossing=%s cached=%s",
                            expire_reason,
                            space_observation.slider_match_score,
                            space_observation.marker_x,
                            space_observation.target_x,
                            space_observation.speed_px_per_second,
                            space_observation.crossing_at,
                            space_observation.prediction_cached,
                        )
                        space_tracker.reset(keep_target=True)

            if state == "RUNNING":
                if sequence != last_observed_sequence:
                    LOGGER.info(
                        "sequence_observed sequence=%r scores=%r appearance=%r",
                        sequence,
                        [round(item.score, 3) for item in detections],
                        [item.appearance for item in detections],
                    )
                    last_observed_sequence = sequence
                history.append(sequence)
                if len(history) == stable_required and len(set(history)) == 1 and sequence:
                    stable_sequence = sequence
                else:
                    stable_sequence = ()

                now = time.perf_counter()
                all_unpressed = bool(detections) and all(
                    item.appearance == "unpressed" for item in detections
                )
                any_pressed = any(item.appearance == "pressed" for item in detections)

                if not round_armed:
                    if not sequence or any_pressed:
                        transition_seen = True
                    fallback_ready = (
                        last_input_completed_at is not None
                        and (now - last_input_completed_at) * 1000
                        >= input_timing.fallback_rearm_ms
                        and bool(last_sent_sequence)
                        and stable_sequence != last_sent_sequence
                    )
                    if (
                        (transition_seen or fallback_ready)
                        and all_unpressed
                        and stable_sequence
                        and not key_sender.busy
                    ):
                        LOGGER.info(
                            "next_group_detected_without_blank sequence=%r transition_seen=%s fallback=%s",
                            stable_sequence,
                            transition_seen,
                            fallback_ready,
                        )
                        reset_round()

                if sequence:
                    clear_frames = 0
                    if round_armed and first_seen_at is None:
                        first_seen_at = now
                        reaction_delay_seconds = input_timing.random_reaction_seconds()
                        input_status = f"WAIT {round(reaction_delay_seconds * 1000)}ms"
                        LOGGER.info(
                            "sequence_first_seen sequence=%r reaction_ms=%d",
                            sequence,
                            round(reaction_delay_seconds * 1000),
                        )
                else:
                    clear_frames += 1
                    if clear_frames >= input_timing.clear_frames_to_rearm:
                        if not key_sender.busy:
                            reset_round()

                if (
                    input_enabled
                    and round_armed
                    and stable_sequence
                    and first_seen_at is not None
                    and reaction_delay_seconds is not None
                ):
                    remaining = first_seen_at + reaction_delay_seconds - now
                    input_status = f"WAIT {max(0, round(remaining * 1000))}ms"
                    if key_sender.start(
                        stable_sequence,
                        target_hwnd,
                        delay_seconds=max(0.0, remaining),
                        first_seen_at=first_seen_at,
                    ):
                        round_armed = False
                        cancel_space_cycle()
                        space_tracking_active = bool(
                            space_timing is not None and space_timing.enabled
                        )
                        space_status = (
                            "TRACKING SLIDER"
                            if space_tracking_active
                            else "OFF"
                        )
                        if space_tracking_active:
                            LOGGER.info(
                                "space_tracking_started sequence=%r",
                                stable_sequence,
                            )
                        last_sent_sequence = stable_sequence
                        LOGGER.info(
                            "input_scheduled sequence=%r reaction_ms=%d remaining_ms=%d target_hwnd=%s",
                            stable_sequence,
                            round(reaction_delay_seconds * 1000),
                            max(0, round(remaining * 1000)),
                            target_hwnd,
                        )
                annotated = (
                    detector.annotate(frame, detections)
                    if detector is not None
                    else frame.copy()
                )
                if space_observation is not None and space_tracker is not None:
                    annotated = space_tracker.annotate(annotated, space_observation)
            else:
                annotated = frame.copy()

            if (
                rhythm_capture_enabled
                and rhythm_capture_collector is not None
                and state == "RUNNING"
                and rhythm_capture_due_at is not None
                and captured_at >= rhythm_capture_due_at
            ):
                try:
                    capture_yolo = yolo_frame
                    if capture_yolo is None:
                        if yolo_detector is None:
                            raise RuntimeError("10-class YOLO model is unavailable")
                        capture_yolo = yolo_detector.detect_frame(frame)
                    extra_objects = (
                        [capture_yolo.slider]
                        if capture_yolo.slider is not None
                        else []
                    )
                    saved_sample = rhythm_capture_collector.save_event(
                        frame,
                        detections,
                        game_mode=config["game_mode"],
                        ui_mode=config["ui_mode"],
                        reasons={"rhythm_capture"},
                        extra_objects=extra_objects,
                    )
                    if saved_sample is not None:
                        image_path, label_path = saved_sample
                        LOGGER.info(
                            "rhythm_capture_saved image=%s label=%s arrows=%d slider=%d",
                            image_path,
                            label_path,
                            len(detections),
                            len(extra_objects),
                        )
                except Exception as error:
                    rhythm_capture_enabled = False
                    rhythm_capture_due_at = None
                    status_message = f"Rhythm screenshots disabled: {error}"
                    LOGGER.exception("rhythm_capture_save_failed")
                else:
                    rhythm_capture_due_at = captured_at + rhythm_capture_interval

            fps = (
                len(frame_times) / max(sum(frame_times), 1e-6)
                if frame_times
                else 0.0
            )
            stable_count = sum(1 for item in history if item == (history[-1] if history else ()))
            activity_status: list[str] = []
            if recorder.active:
                activity_status.append(
                    f"REC {recorder.elapsed_seconds():.1f}s -> "
                    f"{recorder.output_path.name}"
                )
            if rhythm_capture_enabled and rhythm_capture_collector is not None:
                activity_status.append(
                    f"SHOTS {rhythm_capture_collector.split.upper()} "
                    f"{rhythm_capture_collector.saved_count}"
                )
            recording_status = " | ".join(activity_status) or None
            preview.always_on_top = topmost_enabled_for_state(
                topmost_mode, state
            )
            display = draw_status(
                annotated,
                state,
                fps,
                stable_count,
                stable_required,
                stable_sequence,
                preview.always_on_top,
                input_status,
                status_message,
                recording_status,
                preview.scale,
                space_status,
                config["ui_mode"],
                config["game_mode"],
                selector,
                yolo_enabled,
                str(config.get("dataset", {}).get("split", "train")),
                not rhythm_capture_enabled,
                topmost_mode,
            )
            preview.show(display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            wait_for_frame_limit(frame_started_at, maximum_fps)
    finally:
        LOGGER.info("live_stop")
        incomplete_recording = recorder.stop()
        if incomplete_recording is not None:
            LOGGER.info("recording_stopped_on_exit path=%s", incomplete_recording)
        key_sender.close()
        if space_sender is not None:
            space_sender.close()
        capture.close()
        hotkeys.close()
        cv2.destroyAllWindows()
    return 0

from __future__ import annotations

import logging
import time
from collections import deque

import cv2
import dxcam
import numpy as np

from .config import (
    PROJECT_ROOT,
    create_log_path,
    load_config,
    resolve_mode_config,
    save_config,
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
    window_title,
)
from .recorder import RoiVideoRecorder
from .selector import ModeSelector
from .space_timing import RhythmBarTracker, RhythmObservation, SpaceTimingConfig
from .yolo_dataset import YoloDatasetCollector
from .yolo_detector import YoloArrowDetector


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

    def toggle_topmost(self) -> bool:
        self.always_on_top = not self.always_on_top
        self.apply_topmost()
        return self.always_on_top

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
    dataset_enabled: bool = False,
    yolo_enabled: bool = False,
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
    color = {"RUNNING": (60, 230, 80), "PAUSED": (30, 210, 255), "STOPPED": (80, 80, 240)}[state]
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
        "Ctrl+F7 Top | F8 ROI | F9 Start | F10 Pause | F11 Stop",
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
            dataset_enabled,
            yolo_enabled,
        )

    footer = recording_status or message or "Ctrl+F12 Start/Stop ROI recording"
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
    space_tracker = RhythmBarTracker(space_timing)
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
    dataset_collector = YoloDatasetCollector(config.get("dataset", {}))
    window_config = config.setdefault(
        "window", {"scale": 0.55, "always_on_top": False}
    )
    saved_window_config = saved_config.setdefault("window", {})
    if "max_fps" not in saved_window_config:
        saved_window_config["max_fps"] = 60
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
        always_on_top=bool(window_config.get("always_on_top", False)),
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
    space_armed = False
    space_scheduled = False
    space_tracking_active = False
    space_offset_ms: float | None = None
    space_armed_at: float | None = None
    dataset_arrow_captured = False
    dataset_bar_captured = False
    pending_dataset_events: set[str] = set()
    space_status = (
        "WAIT DIRECTIONS"
        if space_timing is not None and space_timing.enabled
        else "OFF"
    )

    def reset_round() -> None:
        nonlocal round_armed, first_seen_at, reaction_delay_seconds
        nonlocal clear_frames, transition_seen, input_status
        nonlocal dataset_arrow_captured
        round_armed = True
        first_seen_at = None
        reaction_delay_seconds = None
        clear_frames = 0
        transition_seen = False
        dataset_arrow_captured = False
        input_status = "DISABLED" if not input_enabled else "READY"

    def cancel_space_cycle(*, clear_target: bool = False) -> None:
        nonlocal space_armed, space_scheduled, space_tracking_active
        nonlocal space_offset_ms, space_status
        nonlocal space_armed_at
        nonlocal dataset_bar_captured
        if space_sender is not None:
            space_sender.cancel()
        if space_tracker is not None:
            space_tracker.reset(keep_target=not clear_target)
        space_armed = False
        space_scheduled = False
        space_tracking_active = False
        space_offset_ms = None
        space_armed_at = None
        dataset_bar_captured = False
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
            status_message = "Focus game window, then press Ctrl+F9"
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
                if selection.kind == "dataset_toggle":
                    enable_dataset = selection.value == "true"
                    dataset_settings = saved_config.setdefault("dataset", {})
                    dataset_settings["enabled"] = enable_dataset
                    config.setdefault("dataset", {})["enabled"] = enable_dataset
                    if enable_dataset:
                        try:
                            dataset_collector = YoloDatasetCollector(dataset_settings)
                        except (OSError, ValueError) as error:
                            dataset_settings["enabled"] = False
                            config["dataset"]["enabled"] = False
                            status_message = f"Dataset recording failed: {error}"
                            LOGGER.exception("dataset_recording_enable_failed")
                        else:
                            status_message = (
                                f"Dataset recording ON ({dataset_collector.split})"
                            )
                            LOGGER.info(
                                "dataset_recording_toggled enabled=true split=%s",
                                dataset_collector.split,
                            )
                    else:
                        dataset_collector.disable()
                        status_message = "Dataset recording OFF"
                        LOGGER.info("dataset_recording_toggled enabled=false")
                    save_config(saved_config)
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
                            status_message = "Arrow detector: YOLO (preloaded)"
                            LOGGER.info(
                                "arrow_detector_switched backend=yolo model=%s",
                                yolo_detector.model_path,
                            )
                    else:
                        yolo_enabled = False
                        detector = opencv_detector
                        status_message = "Arrow detector: OpenCV"
                        LOGGER.info("arrow_detector_switched backend=opencv")
                    save_config(saved_config)
                    history.clear()
                    stable_sequence = ()
                    last_observed_sequence = ()
                    reset_round()
                    continue
                if selection.kind == "blocked":
                    status_message = (
                        "Stop recognition before switching YOLO"
                        if selection.value == "yolo"
                        else "Stop recognition before switching mode"
                    )
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
                if event == "toggle_topmost":
                    enabled = preview.toggle_topmost()
                    saved_config.setdefault("window", {})["always_on_top"] = enabled
                    save_config(saved_config)
                    status_message = f"TOPMOST {'ON' if enabled else 'OFF'}"
                elif event == "select_roi":
                    previous_state = state
                    state = "PAUSED"
                    key_sender.cancel()
                    cancel_space_cycle(clear_target=True)
                    if recorder.active:
                        saved_path = recorder.stop()
                        status_message = f"Recording stopped: {saved_path.name}"
                        LOGGER.warning("recording_stopped_for_roi_change path=%s", saved_path)
                    if select_arrow_roi(capture, saved_config):
                        config["arrow_roi"] = saved_config["arrow_roi"]
                        region = relative_roi_to_region(
                            capture.width, capture.height, saved_config["arrow_roi"]
                        )
                        history.clear()
                        stable_sequence = ()
                        last_observed_sequence = ()
                        last_frame = None
                    if previous_state == "RUNNING":
                        target_hwnd = 0
                        reset_round()
                        status_message = "ROI updated; focus game and press Ctrl+F9"
                    state = "STOPPED" if previous_state == "STOPPED" else "PAUSED"
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
                elif event == "pause_resume":
                    if state == "RUNNING":
                        state = "PAUSED"
                        key_sender.cancel()
                        cancel_space_cycle()
                        history.clear()
                        stable_sequence = ()
                        last_observed_sequence = ()
                        reset_round()
                    elif state == "PAUSED":
                        if target_hwnd and foreground_window() == target_hwnd:
                            state = "RUNNING"
                            history.clear()
                            stable_sequence = ()
                            last_observed_sequence = ()
                            reset_round()
                        else:
                            status_message = "Target not focused; focus game and press Ctrl+F9"
                elif event == "stop":
                    state = "STOPPED"
                    key_sender.cancel()
                    cancel_space_cycle()
                    history.clear()
                    stable_sequence = ()
                    last_observed_sequence = ()
                    target_hwnd = 0
                    target_title = ""
                    reset_round()
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
                    pending_dataset_events.add("space_pressed")
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
            if state == "RUNNING" and detector is not None:
                try:
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
            dataset_events = set(pending_dataset_events)
            pending_dataset_events.clear()
            space_observation: RhythmObservation | None = None
            if (
                state == "RUNNING"
                and space_timing is not None
                and space_tracker is not None
                and space_sender is not None
                and space_timing.enabled
                and space_tracking_active
            ):
                space_observation = space_tracker.observe(frame, captured_at)
                if space_observation.bar_locked and not dataset_bar_captured:
                    dataset_bar_captured = True
                    dataset_events.add("bar_detected")
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
                    if space_observation.bar_rect == (0, 0, 0, 0):
                        space_status = "LOCATING BAR"
                    elif space_observation.target_x is None:
                        space_status = "LOCATING CURSOR"
                    elif space_observation.marker_x is None:
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
                            "space_round_expired reason=%s bar_score=%s "
                            "slider_score=%s cursor_score=%s marker=%s "
                            "target=%s speed=%s crossing=%s cached=%s",
                            expire_reason,
                            space_observation.bar_match_score,
                            space_observation.slider_match_score,
                            space_observation.cursor_match_score,
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
                            "TRACKING BAR"
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
                if stable_sequence and not dataset_arrow_captured:
                    dataset_arrow_captured = True
                    dataset_events.add("arrow_detected")
                annotated = (
                    detector.annotate(frame, detections)
                    if detector is not None
                    else frame.copy()
                )
                if space_observation is not None and space_tracker is not None:
                    annotated = space_tracker.annotate(annotated, space_observation)
            else:
                annotated = frame.copy()

            if dataset_events and dataset_collector.enabled:
                try:
                    saved_sample = dataset_collector.save_event(
                        frame,
                        detections,
                        game_mode=config["game_mode"],
                        ui_mode=config["ui_mode"],
                        reasons=dataset_events,
                    )
                    if saved_sample is not None:
                        image_path, label_path = saved_sample
                        LOGGER.info(
                            "dataset_sample_saved image=%s label=%s detections=%d reasons=%r",
                            image_path,
                            label_path,
                            len(detections),
                            sorted(dataset_events),
                        )
                except Exception as error:
                    dataset_collector.disable()
                    saved_config.setdefault("dataset", {})["enabled"] = False
                    config.setdefault("dataset", {})["enabled"] = False
                    save_config(saved_config)
                    status_message = f"Dataset saving disabled: {error}"
                    LOGGER.exception("dataset_sample_save_failed")

            fps = (
                len(frame_times) / max(sum(frame_times), 1e-6)
                if frame_times
                else 0.0
            )
            stable_count = sum(1 for item in history if item == (history[-1] if history else ()))
            recording_status = None
            if recorder.active:
                recording_status = f"REC {recorder.elapsed_seconds():.1f}s -> {recorder.output_path.name}"
            elif dataset_collector.enabled:
                recording_status = dataset_collector.status
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
                dataset_collector.enabled,
                yolo_enabled,
            )
            preview.show(display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            delay = frame_limit_delay(frame_started_at, maximum_fps)
            if delay > 0:
                time.sleep(delay)
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

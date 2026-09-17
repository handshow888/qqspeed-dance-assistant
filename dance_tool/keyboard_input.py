from __future__ import annotations

import ctypes
import queue
import random
import subprocess
import sys
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass


INPUT_KEYBOARD = 1
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

# Set 1 keyboard scan codes. Arrow keys are extended keys.
SCAN_CODES = {
    "UP": 0x48,
    "DOWN": 0x50,
    "LEFT": 0x4B,
    "RIGHT": 0x4D,
    "SPACE": 0x39,
}


if ctypes.sizeof(ctypes.c_void_p) == 8:
    ULONG_PTR = ctypes.c_ulonglong
else:
    ULONG_PTR = ctypes.c_ulong


class KEYBDINPUT(ctypes.Structure):
    _fields_ = (
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class MOUSEINPUT(ctypes.Structure):
    _fields_ = (
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    )


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = (
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    )


class INPUT_UNION(ctypes.Union):
    _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT))


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = (("type", wintypes.DWORD), ("union", INPUT_UNION))


USER32 = ctypes.windll.user32
USER32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
USER32.SendInput.restype = wintypes.UINT
USER32.GetForegroundWindow.restype = wintypes.HWND
USER32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
USER32.GetWindowTextLengthW.restype = ctypes.c_int
USER32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
USER32.GetWindowTextW.restype = ctypes.c_int
USER32.GetAsyncKeyState.argtypes = (ctypes.c_int,)
USER32.GetAsyncKeyState.restype = wintypes.SHORT

MODIFIER_KEYS = (0x10, 0x11, 0x12, 0x5B, 0x5C)  # Shift, Ctrl, Alt, left/right Win


def foreground_window() -> int:
    return int(USER32.GetForegroundWindow() or 0)


def is_admin() -> bool:
    return bool(ctypes.windll.shell32.IsUserAnAdmin())


def relaunch_as_admin(working_directory: str) -> bool:
    """Relaunch the current command through UAC and return whether it was accepted."""
    parameters = subprocess.list2cmdline(sys.argv)
    result = ctypes.windll.shell32.ShellExecuteW(
        None,
        "runas",
        sys.executable,
        parameters,
        working_directory,
        1,
    )
    return int(result) > 32


def window_title(hwnd: int) -> str:
    length = USER32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    USER32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def is_foreground(hwnd: int) -> bool:
    return hwnd != 0 and foreground_window() == hwnd


def modifiers_released() -> bool:
    return not any(USER32.GetAsyncKeyState(key) & 0x8000 for key in MODIFIER_KEYS)


def send_scan_code(scan_code: int, key_up: bool, *, extended: bool = True) -> None:
    flags = KEYEVENTF_SCANCODE
    if extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    if key_up:
        flags |= KEYEVENTF_KEYUP
    event = INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(0, scan_code, flags, 0, 0))
    if USER32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) != 1:
        raise ctypes.WinError()


def release_all_directions() -> None:
    for name, scan_code in SCAN_CODES.items():
        if name == "SPACE":
            continue
        try:
            send_scan_code(scan_code, key_up=True)
        except OSError:
            pass


def release_space() -> None:
    try:
        send_scan_code(SCAN_CODES["SPACE"], key_up=True, extended=False)
    except OSError:
        pass


def _read_range(config: dict, name: str, floor: int = 0) -> tuple[int, int]:
    value = config[name]
    minimum = max(floor, int(value["min"]))
    maximum = max(minimum, int(value["max"]))
    return minimum, maximum


@dataclass(frozen=True)
class InputTiming:
    reaction_min_ms: int
    reaction_max_ms: int
    key_hold_min_ms: int
    key_hold_max_ms: int
    inter_key_min_ms: int
    inter_key_max_ms: int
    clear_frames_to_rearm: int
    fallback_rearm_ms: int

    @classmethod
    def from_config(cls, config: dict) -> "InputTiming":
        reaction_min, reaction_max = _read_range(config, "reaction_delay_ms", floor=110)
        hold_min, hold_max = _read_range(config, "key_hold_ms", floor=1)
        inter_min, inter_max = _read_range(config, "inter_key_delay_ms", floor=0)
        return cls(
            reaction_min_ms=reaction_min,
            reaction_max_ms=reaction_max,
            key_hold_min_ms=hold_min,
            key_hold_max_ms=hold_max,
            inter_key_min_ms=inter_min,
            inter_key_max_ms=inter_max,
            clear_frames_to_rearm=max(1, int(config.get("clear_frames_to_rearm", 3))),
            fallback_rearm_ms=max(0, int(config.get("fallback_rearm_ms", 350))),
        )

    def random_reaction_seconds(self) -> float:
        return random.uniform(self.reaction_min_ms, self.reaction_max_ms) / 1000.0


class DirectionKeySender:
    """Send one direction sequence on a worker thread so preview capture stays responsive."""

    def __init__(self, timing: InputTiming):
        self.timing = timing
        self.events: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._busy = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def start(
        self,
        sequence: tuple[str, ...],
        target_hwnd: int,
        delay_seconds: float = 0.0,
        first_seen_at: float | None = None,
    ) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._cancel = threading.Event()
            self._thread = threading.Thread(
                target=self._send_sequence,
                args=(sequence, target_hwnd, self._cancel, delay_seconds, first_seen_at),
                daemon=True,
            )
            self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()
        release_all_directions()

    def close(self) -> None:
        self.cancel()
        if self._thread is not None:
            self._thread.join(timeout=0.25)

    def poll(self) -> list[tuple[str, object]]:
        result: list[tuple[str, object]] = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except queue.Empty:
                return result

    def _send_sequence(
        self,
        sequence: tuple[str, ...],
        target_hwnd: int,
        cancel: threading.Event,
        delay_seconds: float,
        first_seen_at: float | None,
    ) -> None:
        timings: list[dict[str, int | str]] = []
        try:
            if delay_seconds > 0 and cancel.wait(delay_seconds):
                self.events.put(("cancelled", sequence))
                return

            # Do not turn Ctrl+F9 into Ctrl+Arrow if the start shortcut is still held.
            while not modifiers_released():
                if cancel.wait(0.005):
                    self.events.put(("cancelled", sequence))
                    return

            actual_reaction_ms = None
            if first_seen_at is not None:
                actual_reaction_ms = round((time.perf_counter() - first_seen_at) * 1000)
            self.events.put(
                (
                    "started",
                    {
                        "sequence": sequence,
                        "actual_reaction_ms": actual_reaction_ms,
                    },
                )
            )

            for index, direction in enumerate(sequence):
                if cancel.is_set():
                    self.events.put(("cancelled", sequence))
                    return
                if not is_foreground(target_hwnd):
                    self.events.put(("error", "Target lost focus; sequence cancelled"))
                    return

                scan_code = SCAN_CODES[direction]
                hold_ms = random.randint(
                    self.timing.key_hold_min_ms, self.timing.key_hold_max_ms
                )
                send_scan_code(scan_code, key_up=False)
                try:
                    if cancel.wait(hold_ms / 1000.0):
                        self.events.put(("cancelled", sequence))
                        return
                finally:
                    send_scan_code(scan_code, key_up=True)

                interval_ms = 0
                if index < len(sequence) - 1:
                    interval_ms = random.randint(
                        self.timing.inter_key_min_ms,
                        self.timing.inter_key_max_ms,
                    )
                timings.append(
                    {"direction": direction, "hold_ms": hold_ms, "interval_ms": interval_ms}
                )
                if interval_ms and cancel.wait(interval_ms / 1000.0):
                    self.events.put(("cancelled", sequence))
                    return

            self.events.put(("completed", {"sequence": sequence, "timings": timings}))
        except Exception as error:
            release_all_directions()
            self.events.put(("error", f"Direction input failed: {error}"))
        finally:
            with self._lock:
                self._busy = False


class SpaceKeySender:
    """Press Space at an absolute monotonic-clock time without blocking capture."""

    def __init__(self, hold_min_ms: int, hold_max_ms: int):
        self.hold_min_ms = max(1, int(hold_min_ms))
        self.hold_max_ms = max(self.hold_min_ms, int(hold_max_ms))
        self.events: queue.SimpleQueue[tuple[str, object]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._busy = False

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def start_at(
        self,
        press_at: float,
        target_hwnd: int,
        *,
        predicted_crossing_at: float,
        sampled_offset_ms: float,
    ) -> bool:
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._cancel = threading.Event()
            self._thread = threading.Thread(
                target=self._press,
                args=(
                    float(press_at),
                    target_hwnd,
                    float(predicted_crossing_at),
                    float(sampled_offset_ms),
                    self._cancel,
                ),
                daemon=True,
            )
            self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()
        release_space()

    def close(self) -> None:
        self.cancel()
        if self._thread is not None:
            self._thread.join(timeout=0.25)

    def poll(self) -> list[tuple[str, object]]:
        result: list[tuple[str, object]] = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except queue.Empty:
                return result

    def _press(
        self,
        press_at: float,
        target_hwnd: int,
        predicted_crossing_at: float,
        sampled_offset_ms: float,
        cancel: threading.Event,
    ) -> None:
        try:
            remaining = press_at - time.perf_counter()
            if remaining > 0 and cancel.wait(remaining):
                self.events.put(("cancelled", None))
                return

            while not modifiers_released():
                if cancel.wait(0.002):
                    self.events.put(("cancelled", None))
                    return
            if not is_foreground(target_hwnd):
                self.events.put(("error", "Target lost focus; Space cancelled"))
                return

            pressed_at = time.perf_counter()
            hold_ms = random.randint(self.hold_min_ms, self.hold_max_ms)
            send_scan_code(SCAN_CODES["SPACE"], key_up=False, extended=False)
            self.events.put(
                (
                    "started",
                    {
                        "pressed_at": pressed_at,
                        "sampled_offset_ms": round(sampled_offset_ms, 1),
                    },
                )
            )
            try:
                if cancel.wait(hold_ms / 1000.0):
                    self.events.put(("cancelled", None))
                    return
            finally:
                send_scan_code(SCAN_CODES["SPACE"], key_up=True, extended=False)

            self.events.put(
                (
                    "completed",
                    {
                        "hold_ms": hold_ms,
                        "sampled_offset_ms": round(sampled_offset_ms, 1),
                        "scheduled_lateness_ms": round((pressed_at - press_at) * 1000, 1),
                        "predicted_crossing_error_ms": round(
                            (pressed_at - predicted_crossing_at) * 1000, 1
                        ),
                    },
                )
            )
        except Exception as error:
            release_space()
            self.events.put(("error", f"Space input failed: {error}"))
        finally:
            with self._lock:
                self._busy = False

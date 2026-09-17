from __future__ import annotations

import ctypes
import queue
import threading
from ctypes import wintypes


WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

MODIFIERS = {
    "ALT": MOD_ALT,
    "CTRL": MOD_CONTROL,
    "CONTROL": MOD_CONTROL,
    "SHIFT": MOD_SHIFT,
    "WIN": MOD_WIN,
}

SPECIAL_KEYS = {
    "=": 0xBB,  # VK_OEM_PLUS: the unshifted '=' key left of Backspace.
}


def parse_hotkey(value: str) -> tuple[int, int]:
    parts = [part.strip().upper() for part in value.split("+") if part.strip()]
    if len(parts) < 2:
        raise ValueError(f"组合键格式无效：{value}")

    modifiers = MOD_NOREPEAT
    for part in parts[:-1]:
        if part not in MODIFIERS:
            raise ValueError(f"不支持的修饰键：{part}")
        modifiers |= MODIFIERS[part]

    key = parts[-1]
    if key.startswith("F") and key[1:].isdigit() and 1 <= int(key[1:]) <= 24:
        virtual_key = 0x70 + int(key[1:]) - 1
    elif key in SPECIAL_KEYS:
        virtual_key = SPECIAL_KEYS[key]
    elif len(key) == 1 and key.isalnum():
        virtual_key = ord(key)
    else:
        raise ValueError(f"不支持的按键：{key}")
    return modifiers, virtual_key


class GlobalHotkeys:
    """Small dependency-free wrapper around the Windows RegisterHotKey API."""

    def __init__(self, configured_hotkeys: dict[str, str]) -> None:
        self.events: queue.SimpleQueue[tuple[str, str | None]] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._ready = threading.Event()
        self._hotkeys = {
            index: (name, display, *parse_hotkey(display))
            for index, (name, display) in enumerate(configured_hotkeys.items(), start=1)
        }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._message_loop, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=2)

    def close(self) -> None:
        if self._thread_id is not None:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
        if self._thread is not None:
            self._thread.join(timeout=1)

    def poll(self) -> list[tuple[str, str | None]]:
        result: list[tuple[str, str | None]] = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except queue.Empty:
                return result

    def _message_loop(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        self._thread_id = kernel32.GetCurrentThreadId()
        registered: list[int] = []

        for hotkey_id, (name, display, modifiers, virtual_key) in self._hotkeys.items():
            if user32.RegisterHotKey(None, hotkey_id, modifiers, virtual_key):
                registered.append(hotkey_id)
            else:
                self.events.put(("error", f"{display} ({name}) unavailable"))

        self._ready.set()
        message = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
                if message.message == WM_HOTKEY:
                    hotkey_id = int(message.wParam)
                    if hotkey_id in self._hotkeys:
                        self.events.put((self._hotkeys[hotkey_id][0], None))
        finally:
            for hotkey_id in registered:
                user32.UnregisterHotKey(None, hotkey_id)

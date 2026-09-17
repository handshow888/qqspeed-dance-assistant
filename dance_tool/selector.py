from __future__ import annotations

import queue
from dataclasses import dataclass

import cv2
import numpy as np

from .modes import GAME_MODES, UI_MODES, Choice


@dataclass(frozen=True)
class SelectorEvent:
    kind: str
    value: str | None = None


@dataclass(frozen=True)
class Button:
    kind: str
    choice: Choice
    left: int
    top: int
    right: int
    bottom: int

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


class ModeSelector:
    """Draw and handle the mouse-only game/UI selectors in the preview header."""

    def __init__(self) -> None:
        self.events: queue.SimpleQueue[SelectorEvent] = queue.SimpleQueue()
        self.buttons: list[Button] = []
        self.enabled = True
        self.game_mode = "traditional_four_key"
        self.ui_mode = "classic"

    def draw(
        self,
        canvas: np.ndarray,
        state: str,
        game_mode: str,
        ui_mode: str,
    ) -> None:
        self.enabled = state == "STOPPED"
        self.game_mode = game_mode
        self.ui_mode = ui_mode
        self.buttons = []
        if canvas.shape[1] < 430:
            return
        self._draw_row(canvas, "GAME", "game_mode", GAME_MODES, 116, game_mode)
        self._draw_row(canvas, "UI", "ui_mode", UI_MODES, 151, ui_mode)

    def _draw_row(
        self,
        canvas: np.ndarray,
        heading: str,
        kind: str,
        choices: tuple[Choice, ...],
        top: int,
        selected: str,
    ) -> None:
        label_width = 66
        gap = 6
        available = canvas.shape[1] - label_width - 12 - gap * (len(choices) - 1)
        button_width = max(80, available // len(choices))
        cv2.putText(
            canvas,
            f"{heading}:",
            (10, top + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )
        left = label_width
        for choice in choices:
            right = min(canvas.shape[1] - 6, left + button_width)
            button = Button(kind, choice, left, top, right, top + 29)
            self.buttons.append(button)
            is_selected = choice.value == selected
            if not self.enabled:
                fill = (48, 48, 48) if not is_selected else (65, 85, 65)
                border = (90, 90, 90)
                text_color = (135, 135, 135)
            elif is_selected:
                fill = (45, 110, 55)
                border = (80, 235, 105)
                text_color = (245, 255, 245)
            else:
                fill = (45, 45, 55)
                border = (135, 135, 155)
                text_color = (230, 230, 235)
            cv2.rectangle(canvas, (left, top), (right, top + 29), fill, -1)
            cv2.rectangle(canvas, (left, top), (right, top + 29), border, 1)
            (text_width, text_height), _ = cv2.getTextSize(
                choice.display, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1
            )
            cv2.putText(
                canvas,
                choice.display,
                (left + max(4, (right - left - text_width) // 2), top + 18 + text_height // 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                text_color,
                1,
                cv2.LINE_AA,
            )
            left = right + gap

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        button = next((item for item in self.buttons if item.contains(x, y)), None)
        if button is None:
            return
        if not self.enabled:
            self.events.put(SelectorEvent("blocked"))
            return
        selected = self.game_mode if button.kind == "game_mode" else self.ui_mode
        if button.choice.value != selected:
            self.events.put(SelectorEvent(button.kind, button.choice.value))

    def poll(self) -> list[SelectorEvent]:
        result: list[SelectorEvent] = []
        while True:
            try:
                result.append(self.events.get_nowait())
            except queue.Empty:
                return result

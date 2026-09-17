from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Choice:
    value: str
    label: str
    display: str


GAME_MODES = (
    Choice("traditional_four_key", "传统四键", "Traditional"),
    Choice("speed_dance", "飞车舞蹈", "Speed Dance"),
    Choice("couple_dance", "双人舞蹈", "Couple Dance"),
)

UI_MODES = (
    Choice("classic", "经典", "Classic"),
    Choice("renewed", "焕新", "Renewed"),
)

GAME_MODE_ALIASES = {
    choice.value: choice.value for choice in GAME_MODES
} | {choice.label: choice.value for choice in GAME_MODES}

UI_MODE_ALIASES = {
    choice.value: choice.value for choice in UI_MODES
} | {choice.label: choice.value for choice in UI_MODES}


def normalize_game_mode(value: str) -> str:
    mode = GAME_MODE_ALIASES.get(str(value).strip().casefold())
    if mode is None:
        raise ValueError(f"未知玩法模式：{value!r}")
    return mode


def normalize_ui_mode(value: str) -> str:
    mode = UI_MODE_ALIASES.get(str(value).strip().casefold())
    if mode is None:
        raise ValueError(f"未知 UI 模式：{value!r}")
    return mode


def choice_display(choices: tuple[Choice, ...], value: str) -> str:
    return next(
        (choice.display for choice in choices if choice.value == value),
        value,
    )

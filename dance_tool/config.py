from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


UI_MODE_ALIASES = {
    "classic": "classic",
    "经典": "classic",
    "renewed": "renewed",
    "焕新": "renewed",
}


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def resolve_ui_config(
    config: dict[str, Any], ui_mode: str | None = None
) -> dict[str, Any]:
    """Overlay the selected UI profile without mutating the saved configuration."""
    requested = str(ui_mode or config.get("ui_mode", "classic")).strip()
    mode = UI_MODE_ALIASES.get(requested.casefold())
    if mode is None:
        available = "、".join(sorted(config.get("ui_profiles", {})))
        raise ValueError(f"未知 UI 模式 {requested!r}；可用模式：{available}")

    profiles = config.get("ui_profiles")
    if not isinstance(profiles, dict):
        # Backward compatibility with the original single-profile config.json.
        result = deepcopy(config)
        result["ui_mode"] = mode
        return result
    profile = profiles.get(mode)
    if not isinstance(profile, dict):
        raise ValueError(f"config.json 缺少 ui_profiles.{mode}")

    result = deepcopy(config)
    _deep_update(result, profile)
    result["ui_mode"] = mode
    return result


def save_config(config: dict[str, Any]) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
        file.write("\n")


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

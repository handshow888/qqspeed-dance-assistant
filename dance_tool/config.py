from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.json"
MODE_CONFIG_ROOT = PROJECT_ROOT / "configs" / "modes"
LOG_DIR = PROJECT_ROOT / "logs"
_ACTIVE_LOG_PATH: Path | None = None


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def resolve_mode_config(
    config: dict[str, Any],
    game_mode: str | None = None,
    ui_mode: str | None = None,
) -> dict[str, Any]:
    """Overlay one game/UI profile without mutating the saved configuration."""
    from .modes import normalize_game_mode, normalize_ui_mode

    selected_game = normalize_game_mode(
        game_mode or str(config.get("game_mode", "traditional_four_key"))
    )
    selected_ui = normalize_ui_mode(ui_mode or str(config.get("ui_mode", "classic")))
    path = MODE_CONFIG_ROOT / selected_game / f"{selected_ui}.json"
    try:
        with path.open("r", encoding="utf-8") as file:
            profile = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"缺少模式配置：{path.relative_to(PROJECT_ROOT)}") from error
    if not isinstance(profile, dict):
        raise ValueError(f"模式配置必须是 JSON 对象：{path.relative_to(PROJECT_ROOT)}")

    result = deepcopy(config)
    _deep_update(result, profile)
    result["game_mode"] = selected_game
    result["ui_mode"] = selected_ui
    result["profile_path"] = str(path.relative_to(PROJECT_ROOT))
    return result


def save_config(config: dict[str, Any]) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)
        file.write("\n")


def create_log_path(keep: int = 30) -> Path:
    """Create a millisecond timestamped log and prune the oldest log files."""
    global _ACTIVE_LOG_PATH
    if _ACTIVE_LOG_PATH is not None:
        return _ACTIVE_LOG_PATH

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now()
    for offset_ms in range(1000):
        timestamp = started_at + timedelta(milliseconds=offset_ms)
        path = LOG_DIR / f"{timestamp:%Y-%m-%d_%H-%M-%S-%f}"[:-3]
        path = path.with_suffix(".log")
        try:
            path.touch(exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("无法创建唯一的运行日志文件")

    log_files = sorted(
        (item for item in LOG_DIR.glob("*.log") if item.is_file()),
        key=lambda item: item.stat().st_mtime_ns,
        reverse=True,
    )
    for old_path in log_files[max(1, int(keep)) :]:
        try:
            old_path.unlink()
        except OSError:
            pass
    _ACTIVE_LOG_PATH = path
    return path


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

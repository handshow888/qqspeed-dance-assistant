from __future__ import annotations

import json
from pathlib import Path

from .config import PROJECT_ROOT, load_config, resolve_mode_config
from .detector import ArrowDetector
from .image_io import read_image, write_image


def find_default_inputs(config: dict) -> list[Path]:
    fixture_dir = (
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / str(config["game_mode"])
        / str(config["ui_mode"])
    )
    return sorted(fixture_dir.glob("*.png")) if fixture_dir.exists() else []


def run_offline(
    input_paths: list[Path] | None = None,
    game_mode: str | None = None,
    ui_mode: str | None = None,
) -> int:
    config = resolve_mode_config(load_config(), game_mode, ui_mode)
    if not config.get("implemented", False):
        raise ValueError(str(config.get("unavailable_reason", "该模式尚不可用")))
    detector = ArrowDetector(config)
    paths = input_paths or find_default_inputs(config)
    output_dir = PROJECT_ROOT / "debug" / "offline"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for path in paths:
        image = read_image(path)
        detections = detector.detect(image)
        annotated = detector.annotate(image, detections)
        output_path = output_dir / f"{path.stem}-detected.png"
        write_image(output_path, annotated)
        item = {
            "file": path.name,
            "sequence": [detection.direction for detection in detections],
            "scores": [round(detection.score, 3) for detection in detections],
            "output": str(output_path),
        }
        summary.append(item)
        print(json.dumps(item, ensure_ascii=False))

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return 0

from __future__ import annotations

import json
from pathlib import Path

from .config import PROJECT_ROOT, load_config, resolve_ui_config
from .detector import ArrowDetector
from .image_io import read_image, write_image


def find_default_inputs(config: dict) -> list[Path]:
    configured_templates = {
        resolve_path
        for paths in config["templates"].values()
        for value in paths
        if (resolve_path := (PROJECT_ROOT / value).resolve()).exists()
    }
    material_dirs = sorted({path.parent for path in configured_templates})
    return sorted(
        path
        for directory in material_dirs
        for path in directory.glob("*.png")
        if path.resolve() not in configured_templates
        and not path.stem.startswith("节奏条")
    )


def run_offline(
    input_paths: list[Path] | None = None, ui_mode: str | None = None
) -> int:
    config = resolve_ui_config(load_config(), ui_mode)
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

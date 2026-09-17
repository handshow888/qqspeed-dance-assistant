from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

from .config import PROJECT_ROOT, load_config, resolve_project_path
from .yolo_dataset import CLASS_NAMES


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class SplitSummary:
    images: int
    labels: int
    boxes: int
    empty_labels: int


def _image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(dataset_root: Path) -> dict[str, SplitSummary]:
    root = dataset_root.resolve()
    classes_path = root / "classes.txt"
    if not classes_path.exists():
        raise ValueError(f"缺少类别文件：{classes_path}")
    classes = [
        line.strip()
        for line in classes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if classes != list(CLASS_NAMES):
        expected = ", ".join(CLASS_NAMES)
        raise ValueError(f"类别名称或顺序不正确，期望：{expected}")

    result: dict[str, SplitSummary] = {}
    for split in ("train", "val"):
        image_root = root / "images" / split
        label_root = root / "labels" / split
        images = _image_files(image_root)
        if not images:
            raise ValueError(f"{split} 数据集没有图片：{image_root}")
        box_count = 0
        empty_count = 0
        label_count = 0
        for image_path in images:
            relative = image_path.relative_to(image_root).with_suffix(".txt")
            label_path = label_root / relative
            if not label_path.exists():
                raise ValueError(f"图片缺少标签文件：{relative}")
            label_count += 1
            lines = [
                line.strip()
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not lines:
                empty_count += 1
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("字段数不是 5")
                    class_id = int(parts[0])
                    coordinates = [float(value) for value in parts[1:]]
                    if not all(math.isfinite(value) for value in coordinates):
                        raise ValueError("坐标包含非有限数值")
                    if class_id not in range(len(classes)):
                        raise ValueError("类别超出范围")
                    center_x, center_y, width, height = coordinates
                    if not (0 <= center_x <= 1 and 0 <= center_y <= 1):
                        raise ValueError("中心坐标超出 0～1")
                    if not (0 < width <= 1 and 0 < height <= 1):
                        raise ValueError("宽高超出 0～1")
                except ValueError as error:
                    raise ValueError(
                        f"非法标签 {label_path}:{line_number}：{error}"
                    ) from error
                box_count += 1
        result[split] = SplitSummary(
            images=len(images),
            labels=label_count,
            boxes=box_count,
            empty_labels=empty_count,
        )
    return result


def write_training_yaml(dataset_root: Path) -> Path:
    root = dataset_root.resolve()
    classes = [
        line.strip()
        for line in (root / "classes.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(classes))
    yaml_path = root / "data.training.yaml"
    safe_root = root.as_posix().replace("'", "''")
    yaml_path.write_text(
        f"path: '{safe_root}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def run_training(
    *,
    dataset_root: Path | None = None,
    model_name: str = "yolo26n.pt",
    epochs: int = 80,
    image_size: int = 640,
    batch_size: int = 8,
    patience: int = 15,
    device: str = "0",
    workers: int = 4,
    output_dir: Path | None = None,
    run_name: str = "yolo26n_arrows",
) -> int:
    config = load_config()
    configured_root = str(
        config.get("dataset", {}).get("output_dir", "datasets/yolo_arrows")
    )
    root = (
        dataset_root.resolve()
        if dataset_root is not None
        else resolve_project_path(configured_root).resolve()
    )
    summary = validate_dataset(root)
    data_yaml = write_training_yaml(root)
    destination = (
        output_dir.resolve()
        if output_dir is not None
        else (PROJECT_ROOT / "runs" / "yolo_arrow").resolve()
    )

    print(
        "数据集检查通过："
        f"train={summary['train'].images}张/{summary['train'].boxes}框，"
        f"val={summary['val'].images}张/{summary['val'].boxes}框"
    )
    print(
        f"训练配置：model={model_name} epochs={epochs} imgsz={image_size} "
        f"batch={batch_size} device={device}"
    )
    print("RTX 3060 Laptop 预计约 3～10 分钟；首次下载模型和建立缓存会额外耗时。")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "缺少训练依赖，请先运行："
            r".\.venv\Scripts\python.exe -m pip install -r requirements-train.txt"
        ) from error

    if device != "cpu" and not torch.cuda.is_available():
        if torch.version.cuda is None:
            raise RuntimeError(
                f"当前安装的是 CPU 版 PyTorch（{torch.__version__}）。请重新运行："
                r".\.venv\Scripts\python.exe -m pip install --upgrade "
                r"--force-reinstall -r requirements-train.txt"
            )
        raise RuntimeError(
            f"PyTorch 已包含 CUDA {torch.version.cuda}，但无法连接显卡；"
            "请检查 NVIDIA 驱动"
        )
    if device != "cpu":
        print(f"CUDA：{torch.cuda.get_device_name(0)}")

    started = time.perf_counter()
    model = YOLO(model_name)
    results = model.train(
        data=str(data_yaml),
        epochs=max(1, int(epochs)),
        imgsz=max(320, int(image_size)),
        batch=max(1, int(batch_size)),
        patience=max(0, int(patience)),
        device=device,
        workers=max(0, int(workers)),
        project=str(destination),
        name=run_name,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        rect=True,
        cache=True,
        amp=True,
        plots=True,
        seed=42,
        deterministic=True,
        fliplr=0.0,
        flipud=0.0,
        hsv_h=0.005,
        hsv_s=0.2,
        hsv_v=0.15,
        scale=0.35,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
        cls=0.25,
    )
    elapsed_minutes = (time.perf_counter() - started) / 60.0
    save_dir = getattr(results, "save_dir", destination / run_name)
    print(f"训练完成：{elapsed_minutes:.1f} 分钟")
    print(f"结果目录：{save_dir}")
    print(f"最佳权重：{Path(save_dir) / 'weights' / 'best.pt'}")
    return 0

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
import shutil
import time

from .config import PROJECT_ROOT, load_config, resolve_project_path
from .yolo_dataset import ARROW_CLASS_NAMES, CLASS_NAMES


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_ARROW_MODEL = "runs/yolo_arrow/yolo26n_arrows/weights/best.pt"
DEFAULT_TEN_CLASS_MODEL = "runs/yolo_arrow/yolo26n_rhythm_10class/weights/best.pt"
DEFAULT_FOUNDATION_MODEL = "yolo26n.pt"
REQUIRED_TRAINING_CLASS_NAMES = ARROW_CLASS_NAMES + ("slider",)
REQUIRED_TRAINING_CLASS_IDS = frozenset(
    CLASS_NAMES.index(class_name) for class_name in REQUIRED_TRAINING_CLASS_NAMES
)


@dataclass(frozen=True)
class SplitSummary:
    images: int
    labels: int
    boxes: int
    empty_labels: int
    class_counts: tuple[int, ...]


def _image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(
    dataset_root: Path,
    *,
    splits: tuple[str, ...] = ("train", "val"),
) -> dict[str, SplitSummary]:
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
    for split in splits:
        image_root = root / "images" / split
        label_root = root / "labels" / split
        images = _image_files(image_root)
        if not images:
            raise ValueError(f"{split} 数据集没有图片：{image_root}")
        box_count = 0
        empty_count = 0
        label_count = 0
        class_counts = [0] * len(classes)
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
                class_counts[class_id] += 1
        result[split] = SplitSummary(
            images=len(images),
            labels=label_count,
            boxes=box_count,
            empty_labels=empty_count,
            class_counts=tuple(class_counts),
        )
    return result


def _classes_in_label(label_path: Path) -> set[int]:
    return {
        int(line.split()[0])
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def validate_class_coverage(summary: dict[str, SplitSummary]) -> None:
    missing: list[str] = []
    for split, split_summary in summary.items():
        absent = [
            CLASS_NAMES[class_id]
            for class_id in sorted(REQUIRED_TRAINING_CLASS_IDS)
            if split_summary.class_counts[class_id] == 0
        ]
        if absent:
            missing.append(f"{split} 缺少：{', '.join(absent)}")
    if missing:
        raise ValueError(
            "训练和验证集必须分别覆盖8类箭头和slider；" + "；".join(missing)
        )


def format_class_distribution(summary: dict[str, SplitSummary]) -> str:
    columns = tuple(summary)
    lines = ["类别分布："]
    for class_id, class_name in enumerate(CLASS_NAMES):
        counts = " ".join(
            f"{split}={summary[split].class_counts[class_id]}" for split in columns
        )
        lines.append(f"  {class_id:>2} {class_name:<18} {counts}")
    return "\n".join(lines)


def create_auto_validation_split(
    dataset_root: Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Path, Path, int, int]:
    """Create stable image lists without moving or copying the source dataset."""
    root = dataset_root.resolve()
    image_root = root / "images" / "train"
    label_root = root / "labels" / "train"
    images = _image_files(image_root)
    if len(images) < 2:
        raise ValueError("自动划分训练集和验证集至少需要 2 张图片")

    ratio = min(max(float(val_ratio), 0.05), 0.5)
    val_count = min(len(images) - 1, max(1, round(len(images) * ratio)))
    image_classes = {
        image: _classes_in_label(
            label_root / image.relative_to(image_root).with_suffix(".txt")
        )
        for image in images
    }
    all_classes = set().union(*image_classes.values())
    expected_classes = set(REQUIRED_TRAINING_CLASS_IDS)
    missing_classes = expected_classes - all_classes
    if missing_classes:
        missing_names = ", ".join(CLASS_NAMES[index] for index in sorted(missing_classes))
        raise ValueError(f"训练数据缺少类别：{missing_names}")
    unsplittable_classes = {
        class_id
        for class_id in expected_classes
        if sum(class_id in values for values in image_classes.values()) < 2
    }
    if unsplittable_classes:
        names = ", ".join(CLASS_NAMES[index] for index in sorted(unsplittable_classes))
        raise ValueError(f"以下类别不足2张图片，无法同时放入train和val：{names}")

    selected_val: list[Path] | None = None
    # Retry deterministic shuffles until every class that can be represented on
    # both sides is present in both sets. The same seed always yields the same split.
    for attempt in range(2000):
        candidates = list(images)
        random.Random(seed + attempt).shuffle(candidates)
        val_images = candidates[:val_count]
        train_images = candidates[val_count:]
        val_classes = set().union(*(image_classes[path] for path in val_images))
        train_classes = set().union(*(image_classes[path] for path in train_images))
        if expected_classes <= val_classes and expected_classes <= train_classes:
            selected_val = sorted(val_images)
            selected_train = sorted(train_images)
            break
    if selected_val is None:
        raise ValueError(
            "无法自动划分出类别完整的验证集，请手动把独立样本放入 images/val 和 labels/val"
        )

    train_list = root / "training.train.txt"
    val_list = root / "training.val.txt"
    train_list.write_text(
        "".join(f"{path.as_posix()}\n" for path in selected_train),
        encoding="utf-8",
    )
    val_list.write_text(
        "".join(f"{path.as_posix()}\n" for path in selected_val),
        encoding="utf-8",
    )
    return train_list, val_list, len(selected_train), len(selected_val)


def write_training_yaml(
    dataset_root: Path,
    *,
    train_source: str = "images/train",
    val_source: str = "images/val",
) -> Path:
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
        f"train: {train_source}\n"
        f"val: {val_source}\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    return yaml_path


def run_training(
    *,
    dataset_root: Path | None = None,
    model_name: str | None = None,
    epochs: int = 100,
    image_size: int = 640,
    batch_size: int = 8,
    patience: int = 20,
    device: str = "0",
    workers: int = 4,
    output_dir: Path | None = None,
    run_name: str = "yolo26n_rhythm_10class",
    auto_val_ratio: float = 0.2,
    seed: int = 42,
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
    val_images = _image_files(root / "images" / "val")
    if val_images:
        summary = validate_dataset(root)
        validate_class_coverage(summary)
        data_yaml = write_training_yaml(root)
        train_count = summary["train"].images
        val_count = summary["val"].images
        split_message = "使用目录中现有的独立验证集"
    else:
        summary = validate_dataset(root, splits=("train",))
        train_list, val_list, train_count, val_count = create_auto_validation_split(
            root,
            val_ratio=auto_val_ratio,
            seed=seed,
        )
        data_yaml = write_training_yaml(
            root,
            train_source=train_list.name,
            val_source=val_list.name,
        )
        split_message = (
            f"val 目录为空，已生成固定清单：train={train_count}张，"
            f"val={val_count}张（原始文件未移动）"
        )
    destination = (
        output_dir.resolve()
        if output_dir is not None
        else (PROJECT_ROOT / "runs" / "yolo_arrow").resolve()
    )

    total_images = sum(item.images for item in summary.values())
    total_boxes = sum(item.boxes for item in summary.values())
    print(f"数据集检查通过：共{total_images}张/{total_boxes}框")
    print(split_message)
    if "val" in summary:
        print(format_class_distribution(summary))

    if model_name is None:
        previous_ten_class = resolve_project_path(DEFAULT_TEN_CLASS_MODEL).resolve()
        previous_arrow = resolve_project_path(DEFAULT_ARROW_MODEL).resolve()
        if previous_ten_class.exists():
            model_name = str(previous_ten_class)
        elif previous_arrow.exists():
            model_name = str(previous_arrow)
        else:
            model_name = DEFAULT_FOUNDATION_MODEL

    model_path = Path(model_name)
    if not model_path.is_absolute():
        project_model_path = (PROJECT_ROOT / model_path).resolve()
        if project_model_path.exists():
            model_path = project_model_path
            model_name = str(model_path)
    if model_path.exists():
        backup_dir = (PROJECT_ROOT / "weights" / "backups").resolve()
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{model_path.stem}_{stamp}{model_path.suffix}"
        counter = 2
        while backup_path.exists():
            backup_path = backup_dir / (
                f"{model_path.stem}_{stamp}_{counter}{model_path.suffix}"
            )
            counter += 1
        shutil.copy2(model_path, backup_path)
        print(f"原始权重备份：{backup_path}")
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
        seed=seed,
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
        cls=0.5,
    )
    elapsed_minutes = (time.perf_counter() - started) / 60.0
    save_dir = getattr(results, "save_dir", destination / run_name)
    print(f"训练完成：{elapsed_minutes:.1f} 分钟")
    print(f"结果目录：{save_dir}")
    print(f"最佳权重：{Path(save_dir) / 'weights' / 'best.pt'}")
    return 0

from __future__ import annotations

import argparse
import ctypes
from pathlib import Path
import sys
import traceback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QQ飞车舞蹈模式识别与按键 Demo")
    parser.add_argument(
        "--game-mode",
        choices=("traditional_four_key", "speed_dance", "couple_dance", "传统四键", "飞车舞蹈", "双人舞蹈"),
        help="临时覆盖玩法模式",
    )
    parser.add_argument(
        "--ui-mode",
        choices=("classic", "renewed", "经典", "焕新"),
        help="临时覆盖 config.json 的 UI 模式（classic/经典 或 renewed/焕新）",
    )
    subparsers = parser.add_subparsers(dest="command")

    offline = subparsers.add_parser("offline", help="对本地截图进行离线识别")
    offline.add_argument("images", nargs="*", type=Path, help="可选：指定一个或多个截图")

    subparsers.add_parser("calibrate", help="框选并保存实时箭头识别区域")
    subparsers.add_parser("live", help="实时截屏并观察识别结果")
    train = subparsers.add_parser("train", help="训练箭头、节奏条和滑块检测模型")
    train.add_argument("--dataset-root", type=Path, help="数据集根目录")
    train.add_argument(
        "--model",
        help="初始权重；默认依次使用现有10类、原8类、通用yolo26n权重",
    )
    train.add_argument("--epochs", type=int, default=100, help="最大训练轮数")
    train.add_argument("--imgsz", type=int, default=640, help="训练输入尺寸")
    train.add_argument("--batch", type=int, default=8, help="批大小")
    train.add_argument("--patience", type=int, default=20, help="早停等待轮数")
    train.add_argument("--device", default="0", help="CUDA 设备编号或 cpu")
    train.add_argument("--workers", type=int, default=4, help="数据加载进程数")
    train.add_argument("--output-dir", type=Path, help="训练结果目录")
    train.add_argument(
        "--name", default="yolo26n_rhythm_10class", help="本次训练名称"
    )
    train.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="val目录为空时，从train生成固定验证清单的比例",
    )
    train.add_argument("--seed", type=int, default=42, help="数据划分和训练随机种子")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    command = args.command or "live"
    if command == "offline":
        from dance_tool.offline import run_offline

        return run_offline(
            args.images or None,
            game_mode=args.game_mode,
            ui_mode=args.ui_mode,
        )
    if command == "calibrate":
        from dance_tool.live import calibrate_arrow_roi

        return calibrate_arrow_roi()
    if command == "live":
        from dance_tool.live import run_live

        return run_live(game_mode=args.game_mode, ui_mode=args.ui_mode)
    if command == "train":
        from dance_tool.train_yolo import run_training

        return run_training(
            dataset_root=args.dataset_root,
            model_name=args.model,
            epochs=args.epochs,
            image_size=args.imgsz,
            batch_size=args.batch,
            patience=args.patience,
            device=args.device,
            workers=args.workers,
            output_dir=args.output_dir,
            run_name=args.name,
            auto_val_ratio=args.val_ratio,
            seed=args.seed,
        )
    raise AssertionError(f"未知命令：{command}")


def _report_startup_error(error: BaseException) -> None:
    from dance_tool.config import create_log_path

    log_path = create_log_path()
    detail = "".join(traceback.format_exception(error))
    with log_path.open("a", encoding="utf-8") as file:
        file.write("\nSTARTUP ERROR\n")
        file.write(detail)
    message = f"程序启动失败：{error}\n\n详细信息已保存到：\n{log_path}"
    print(message, file=sys.stderr)
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(0, message, "QQ飞车舞蹈工具", 0x10)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        _report_startup_error(error)
        raise SystemExit(1)

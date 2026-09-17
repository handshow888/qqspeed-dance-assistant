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

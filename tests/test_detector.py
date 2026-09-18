from __future__ import annotations

import unittest
import time
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import call, patch

import cv2
import numpy as np

from dance_tool import config as config_module
from dance_tool.config import (
    PROJECT_ROOT,
    changed_config_paths,
    load_config,
    normalize_topmost_mode,
    resolve_mode_config,
    topmost_enabled_for_state,
)
from dance_tool.detector import ArrowDetector, detection_preview_label
from dance_tool.hotkeys import parse_hotkey
from dance_tool.image_io import read_image
from dance_tool.keyboard_input import (
    INPUT,
    SCAN_CODES,
    DirectionKeySender,
    InputTiming,
    SpaceKeySender,
    is_admin,
)
from dance_tool.live import (
    PreviewWindow,
    automatic_window_roi,
    draw_status,
    frame_limit_delay,
    relative_roi_to_region,
    space_expire_reason,
)
from dance_tool.recorder import RoiVideoRecorder
from dance_tool.selector import ModeSelector
from dance_tool.space_timing import SliderTracker, SpaceTimingConfig
from dance_tool.train_yolo import (
    create_auto_validation_split,
    format_class_distribution,
    validate_dataset,
    validate_class_coverage,
    write_training_yaml,
)
from dance_tool.yolo_dataset import (
    ARROW_CLASS_NAMES,
    CLASS_NAMES,
    YoloDatasetCollector,
    detection_to_yolo_line,
    object_to_yolo_line,
)
from dance_tool.yolo_detector import (
    YoloObjectDetection,
    YoloRuntimeSettings,
    yolo_rows_to_detections,
    yolo_rows_to_frame_detections,
)
from main import build_parser


class ArrowDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.saved_config = load_config()
        cls.detector = ArrowDetector(
            resolve_mode_config(cls.saved_config, "traditional_four_key", "renewed")
        )
        cls.materials = (
            PROJECT_ROOT / "tests" / "fixtures" / "traditional_four_key" / "renewed"
        )
        cls.templates = (
            PROJECT_ROOT
            / "assets"
            / "traditional_four_key"
            / "renewed"
            / "templates"
        )
        cls.classic_config = resolve_mode_config(
            cls.saved_config, "traditional_four_key", "classic"
        )
        cls.classic_detector = ArrowDetector(cls.classic_config)

    def assert_sequence(self, filename: str, expected: list[str]) -> None:
        image = read_image(self.materials / filename)
        actual = [item.direction for item in self.detector.detect(image)]
        self.assertEqual(expected, actual)

    def test_no_command_defaults_to_live_at_dispatch(self) -> None:
        args = build_parser().parse_args([])
        self.assertIsNone(args.command)

    def test_live_frame_limit_defaults_to_sixty_fps_period(self) -> None:
        self.assertAlmostEqual(
            1 / 60 - 0.005,
            frame_limit_delay(10.0, 60.0, now=10.005),
            places=6,
        )
        self.assertEqual(0.0, frame_limit_delay(10.0, 60.0, now=10.020))

    def test_auto_roi_uses_full_window_width_and_configured_vertical_bounds(self) -> None:
        width, height = 1920, 1010
        roi = automatic_window_roi(
            width,
            height,
            (0, 0, width, height),
            (0, 0, width, height),
            0.72,
            0.93,
        )
        region = relative_roi_to_region(width, height, roi)
        self.assertEqual((0, 727, 1920, 939), region)

        screenshot_path = PROJECT_ROOT / "logs" / "Snipaste_2026-09-17_23-33-48.png"
        if not screenshot_path.exists():
            return
        image = read_image(screenshot_path)
        self.assertEqual((height, width), image.shape[:2])
        left, top, right, bottom = region
        cropped = image[top:bottom, left:right]
        self.assertEqual((212, 1920), cropped.shape[:2])

    def test_auto_roi_converts_virtual_monitor_coordinates(self) -> None:
        roi = automatic_window_roi(
            1920,
            1080,
            (2020, 100, 3620, 1000),
            (1920, 0, 3840, 1080),
            0.72,
            0.97,
        )
        self.assertEqual(
            (100, 748, 1700, 973),
            relative_roi_to_region(1920, 1080, roi),
        )
        self.assertEqual(0.0, frame_limit_delay(10.0, 0.0, now=10.005))

    def test_train_command_defaults_are_for_ten_class_dataset(self) -> None:
        args = build_parser().parse_args(["train"])
        self.assertEqual("train", args.command)
        self.assertIsNone(args.model)
        self.assertEqual(100, args.epochs)
        self.assertEqual(640, args.imgsz)
        self.assertEqual(8, args.batch)
        self.assertEqual("0", args.device)
        self.assertEqual("yolo26n_rhythm_10class", args.name)
        self.assertEqual(0.2, args.val_ratio)

    def test_mode_profiles_can_be_switched_without_changing_saved_mode(self) -> None:
        classic = resolve_mode_config(self.saved_config, "传统四键", "经典")
        renewed = resolve_mode_config(self.saved_config, "传统四键", "焕新")
        couple_classic = resolve_mode_config(self.saved_config, "双人舞蹈", "经典")
        couple_renewed = resolve_mode_config(self.saved_config, "双人舞蹈", "焕新")
        speed_dance = resolve_mode_config(self.saved_config, "飞车舞蹈", "经典")
        self.assertEqual("traditional_four_key", classic["game_mode"])
        self.assertEqual("classic", classic["ui_mode"])
        self.assertEqual("renewed", renewed["ui_mode"])
        self.assertEqual(0.31, classic["recognition"]["match_threshold"])
        self.assertEqual(0.42, renewed["recognition"]["match_threshold"])
        self.assertIn("classic", classic["space"]["slider_templates"][0])
        self.assertIn("renewed", renewed["space"]["slider_templates"][0])
        self.assertTrue(couple_classic["implemented"])
        self.assertTrue(couple_renewed["implemented"])
        self.assertEqual(classic["templates"], couple_classic["templates"])
        self.assertEqual(renewed["templates"], couple_renewed["templates"])
        self.assertEqual(
            classic["space"]["slider_templates"],
            couple_classic["space"]["slider_templates"],
        )
        self.assertEqual(
            renewed["space"]["slider_templates"],
            couple_renewed["space"]["slider_templates"],
        )
        self.assertLess(
            couple_classic["space"]["cursor_window_ratio"],
            classic["space"]["cursor_window_ratio"],
        )
        self.assertLess(
            couple_renewed["space"]["cursor_window_ratio"],
            renewed["space"]["cursor_window_ratio"],
        )
        self.assertFalse(speed_dance["implemented"])
        self.assertEqual("traditional_four_key", self.saved_config["game_mode"])
        self.assertEqual("classic", self.saved_config["ui_mode"])
        for game_mode in ("traditional_four_key", "speed_dance", "couple_dance"):
            for ui_mode in ("classic", "renewed"):
                with self.subTest(game_mode=game_mode, ui_mode=ui_mode):
                    profile = resolve_mode_config(
                        self.saved_config, game_mode, ui_mode
                    )
                    self.assertEqual(game_mode, profile["game_mode"])
                    self.assertEqual(ui_mode, profile["ui_mode"])
                    self.assertIn("implemented", profile)

    def test_cursor_window_ratio_is_owned_by_implemented_mode_profiles(self) -> None:
        self.assertNotIn("cursor_window_ratio", self.saved_config["space"])
        for game_mode in ("traditional_four_key", "couple_dance"):
            for ui_mode in ("classic", "renewed"):
                with self.subTest(game_mode=game_mode, ui_mode=ui_mode):
                    profile = resolve_mode_config(
                        self.saved_config, game_mode, ui_mode
                    )
                    self.assertIn("cursor_window_ratio", profile["space"])

    def test_roi_recorder_writes_fixed_timeline(self) -> None:
        with TemporaryDirectory() as directory:
            recorder = RoiVideoRecorder(
                {
                    "fps": 5,
                    "codec": "mp4v",
                    "output_dir": directory,
                }
            )
            frame = np.zeros((72, 128, 3), dtype=np.uint8)
            path = recorder.start(frame, now=100.0)
            recorder.add_frame(frame, now=100.0)
            recorder.add_frame(frame, now=101.0)
            completed = recorder.stop()
            self.assertEqual(path, completed)
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)
            video = cv2.VideoCapture(str(path))
            try:
                self.assertEqual(5, round(video.get(cv2.CAP_PROP_FRAME_COUNT)))
                self.assertEqual(5, round(video.get(cv2.CAP_PROP_FPS)))
            finally:
                video.release()

            second = recorder.start(frame, now=200.0)
            recorder.add_frame(frame, now=200.2)
            recorder.stop()
            self.assertNotEqual(path.name, second.name)

    def test_eight_arrow_sequence(self) -> None:
        self.assert_sequence(
            "727a0882-6da9-4986-bfa9-13b30b3e0ac0.png",
            ["DOWN", "DOWN", "LEFT", "LEFT", "DOWN", "DOWN", "RIGHT", "RIGHT"],
        )

    def test_seven_arrow_sequence_with_pressed_arrows(self) -> None:
        image = read_image(
            self.materials / "edc8d7e5-1f8f-41fc-86e6-4bf49aa3a39d.png"
        )
        detections = self.detector.detect(image)
        self.assertEqual(
            ["UP", "DOWN", "DOWN", "RIGHT", "DOWN", "UP", "RIGHT"],
            [item.direction for item in detections],
        )
        self.assertEqual(
            ["pressed", "pressed", "unpressed", "unpressed", "unpressed", "unpressed", "unpressed"],
            [item.appearance for item in detections],
        )

    def test_empty_sequence(self) -> None:
        self.assert_sequence("eada1273-ddcc-4301-b46e-c715ad5485b9.png", [])

    def test_all_pressed_direction_templates(self) -> None:
        for direction, filename in (
            ("UP", "上箭头-已按下.png"),
            ("DOWN", "下箭头-已按下.png"),
            ("LEFT", "左箭头-已按下.png"),
            ("RIGHT", "右箭头-已按下.png"),
        ):
            with self.subTest(direction=direction):
                image = read_image(self.templates / filename)
                padded = cv2.copyMakeBorder(
                    image, 20, 20, 20, 20, cv2.BORDER_REPLICATE
                )
                detections = self.detector.detect(padded, keep_main_row=False)
                matching = [item for item in detections if item.direction == direction]
                self.assertTrue(matching)
                self.assertEqual("pressed", max(matching, key=lambda item: item.score).appearance)

    def test_yolo_dataset_uses_eight_direction_and_state_classes(self) -> None:
        from dance_tool.detector import Detection

        up = Detection("UP", 0.9, 20, 10, 40, 20, "unpressed")
        right = Detection("RIGHT", 0.8, 100, 40, 20, 30, "pressed")
        self.assertEqual(
            "0 0.200000 0.200000 0.200000 0.200000",
            detection_to_yolo_line(up, 200, 100),
        )
        self.assertTrue(detection_to_yolo_line(right, 200, 100).startswith("7 "))
        self.assertEqual(8, len(ARROW_CLASS_NAMES))
        self.assertEqual(10, len(CLASS_NAMES))
        self.assertEqual(("rhythm_bar", "slider"), CLASS_NAMES[-2:])

    def test_preview_uses_single_letter_direction_abbreviation(self) -> None:
        from dance_tool.detector import Detection

        pressed_up = Detection("UP", 0.91, 10, 10, 20, 20, "pressed")
        unpressed_left = Detection("LEFT", 0.82, 10, 10, 20, 20, "unpressed")
        self.assertEqual("1:U/P 0.91", detection_preview_label(1, pressed_up))
        self.assertEqual("2:L/U 0.82", detection_preview_label(2, unpressed_left))

    def test_rhythm_objects_convert_to_yolo_classes_eight_and_nine(self) -> None:
        bar = YoloObjectDetection("rhythm_bar", 0.9, 10, 20, 100, 10)
        slider = YoloObjectDetection("slider", 0.9, 30, 20, 10, 10)
        self.assertTrue(object_to_yolo_line(bar, 200, 100).startswith("8 "))
        self.assertTrue(object_to_yolo_line(slider, 200, 100).startswith("9 "))

    def test_ctrl_equals_uses_the_unshifted_oem_plus_key(self) -> None:
        modifiers, virtual_key = parse_hotkey("Ctrl+=")
        self.assertGreater(modifiers, 0)
        self.assertEqual(0xBB, virtual_key)
        self.assertEqual(
            0.2,
            float(self.saved_config["dataset"]["rhythm_capture_interval_seconds"]),
        )

    def test_yolo_output_maps_to_existing_detection_interface(self) -> None:
        detections = yolo_rows_to_detections(
            [
                [80.2, 10.4, 110.8, 40.6, 0.91, 7],
                [10.1, 11.2, 39.8, 41.1, 0.88, 0],
            ],
            CLASS_NAMES,
            200,
            100,
        )
        self.assertEqual(["UP", "RIGHT"], [item.direction for item in detections])
        self.assertEqual(
            ["unpressed", "pressed"],
            [item.appearance for item in detections],
        )
        self.assertEqual((10, 11, 40, 41), detections[0].box)

    def test_arrow_detector_ignores_rhythm_classes_from_ten_class_model(self) -> None:
        detections = yolo_rows_to_detections(
            [
                [10, 10, 100, 30, 0.99, 8],
                [20, 12, 35, 28, 0.98, 9],
                [40, 10, 60, 30, 0.97, 2],
            ],
            CLASS_NAMES,
            200,
            100,
        )
        self.assertEqual(1, len(detections))
        self.assertEqual("DOWN", detections[0].direction)

    def test_yolo_frame_result_ignores_bar_and_exposes_best_slider(self) -> None:
        result = yolo_rows_to_frame_detections(
            [
                [40, 10, 60, 30, 0.97, 2],
                [10, 5, 190, 25, 0.81, 8],
                [11, 5, 188, 25, 0.42, 8],
                [25, 8, 35, 22, 0.88, 9],
            ],
            CLASS_NAMES,
            200,
            100,
        )
        self.assertEqual(1, len(result.arrows))
        self.assertFalse(hasattr(result, "rhythm_bar"))
        self.assertEqual(30, result.slider.center_x)

    def test_yolo_dataset_saves_named_captures_with_empty_labels_supported(self) -> None:
        from dance_tool.detector import Detection

        with TemporaryDirectory() as directory:
            collector = YoloDatasetCollector(
                {
                    "output_dir": directory,
                    "split": "train",
                }
            )
            frame = np.zeros((100, 200, 3), dtype=np.uint8)
            unpressed = [Detection("LEFT", 0.9, 50, 20, 30, 40, "unpressed")]
            pressed = [Detection("LEFT", 0.9, 50, 20, 30, 40, "pressed")]

            first = collector.save_event(
                frame,
                unpressed,
                game_mode="traditional_four_key",
                ui_mode="classic",
                reasons={"arrow_detected"},
                extra_objects=(
                    YoloObjectDetection("rhythm_bar", 0.9, 10, 5, 180, 20),
                    YoloObjectDetection("slider", 0.9, 30, 8, 12, 14),
                ),
            )
            empty = collector.save_event(
                frame,
                [],
                game_mode="traditional_four_key",
                ui_mode="classic",
                reasons={"bar_detected", "space_pressed"},
            )
            pressed_sample = collector.save_event(
                frame,
                pressed,
                game_mode="traditional_four_key",
                ui_mode="classic",
                reasons={"space_pressed"},
            )

            self.assertIsNotNone(first)
            self.assertIsNotNone(empty)
            self.assertIsNotNone(pressed_sample)
            self.assertEqual(3, collector.saved_count)
            image_root = (
                Path(directory)
                / "images"
                / "train"
                / "traditional_four_key"
                / "classic"
            )
            label_root = (
                Path(directory)
                / "labels"
                / "train"
                / "traditional_four_key"
                / "classic"
            )
            image_files = list(image_root.glob("*.png"))
            label_files = list(label_root.glob("*.txt"))
            self.assertEqual(3, len(image_files))
            self.assertEqual(3, len(label_files))
            self.assertTrue((Path(directory) / "data.yaml").exists())
            self.assertEqual(
                list(CLASS_NAMES),
                (Path(directory) / "classes.txt")
                .read_text(encoding="utf-8")
                .splitlines(),
            )
            self.assertTrue(
                label_files[-1].read_text(encoding="utf-8").startswith("5 ")
            )
            first_lines = first[1].read_text(encoding="utf-8").splitlines()
            self.assertTrue(any(line.startswith("8 ") for line in first_lines))
            self.assertTrue(any(line.startswith("9 ") for line in first_lines))
            empty_labels = [path for path in label_files if path.stat().st_size == 0]
            self.assertEqual(1, len(empty_labels))
            self.assertIn("bar_detected+space_pressed", empty_labels[0].name)

    def test_training_dataset_validation_and_yaml_generation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "classes.txt").write_text(
                "\n".join(CLASS_NAMES) + "\n", encoding="utf-8"
            )
            for split in ("train", "val"):
                image_dir = (
                    root / "images" / split / "traditional_four_key" / "classic"
                )
                label_dir = (
                    root / "labels" / split / "traditional_four_key" / "classic"
                )
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                cv2.imwrite(
                    str(image_dir / "sample.png"),
                    np.zeros((32, 64, 3), dtype=np.uint8),
                )
                (label_dir / "sample.txt").write_text(
                    "0 0.500000 0.500000 0.250000 0.500000\n",
                    encoding="utf-8",
                )

            summary = validate_dataset(root)
            self.assertEqual(1, summary["train"].images)
            self.assertEqual(1, summary["val"].boxes)
            with self.assertRaisesRegex(ValueError, "必须分别覆盖8类箭头和slider"):
                validate_class_coverage(summary)
            yaml_path = write_training_yaml(root)
            yaml_text = yaml_path.read_text(encoding="utf-8")
            self.assertIn("train: images/train", yaml_text)
            self.assertIn("7: right_pressed", yaml_text)

    def test_auto_validation_split_keeps_sources_and_covers_all_classes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "classes.txt").write_text(
                "\n".join(CLASS_NAMES) + "\n", encoding="utf-8"
            )
            image_dir = root / "images" / "train" / "traditional_four_key" / "classic"
            label_dir = root / "labels" / "train" / "traditional_four_key" / "classic"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            all_classes = "\n".join(
                f"{class_id} 0.5 0.5 0.1 0.1" for class_id in range(10)
            )
            for index in range(10):
                cv2.imwrite(
                    str(image_dir / f"sample_{index:02d}.png"),
                    np.zeros((32, 64, 3), dtype=np.uint8),
                )
                (label_dir / f"sample_{index:02d}.txt").write_text(
                    all_classes + "\n", encoding="utf-8"
                )

            train_list, val_list, train_count, val_count = create_auto_validation_split(
                root, val_ratio=0.2, seed=42
            )
            self.assertEqual((8, 2), (train_count, val_count))
            self.assertEqual(10, len(list(image_dir.glob("*.png"))))
            self.assertTrue(train_list.exists())
            self.assertTrue(val_list.exists())
            train_paths = set(train_list.read_text(encoding="utf-8").splitlines())
            val_paths = set(val_list.read_text(encoding="utf-8").splitlines())
            self.assertFalse(train_paths & val_paths)

    def test_rhythm_bar_is_optional_for_training_and_auto_split(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "classes.txt").write_text(
                "\n".join(CLASS_NAMES) + "\n", encoding="utf-8"
            )
            image_dir = root / "images" / "train"
            label_dir = root / "labels" / "train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            required_classes = tuple(range(8)) + (9,)
            labels = "\n".join(
                f"{class_id} 0.5 0.5 0.1 0.1" for class_id in required_classes
            )
            for index in range(10):
                cv2.imwrite(
                    str(image_dir / f"sample_{index:02d}.png"),
                    np.zeros((32, 64, 3), dtype=np.uint8),
                )
                (label_dir / f"sample_{index:02d}.txt").write_text(
                    labels + "\n", encoding="utf-8"
                )

            train_list, val_list, train_count, val_count = create_auto_validation_split(
                root, val_ratio=0.2, seed=42
            )
            self.assertEqual((8, 2), (train_count, val_count))
            summary = validate_dataset(
                root,
                splits=("train",),
            )
            validate_class_coverage(summary)
            self.assertEqual(0, summary["train"].class_counts[8])
            self.assertEqual(10, summary["train"].class_counts[9])
            self.assertTrue(train_list.exists())
            self.assertTrue(val_list.exists())

    def test_class_distribution_lists_train_and_val_counts(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "classes.txt").write_text(
                "\n".join(CLASS_NAMES) + "\n", encoding="utf-8"
            )
            all_classes = "\n".join(
                f"{class_id} 0.5 0.5 0.1 0.1" for class_id in range(10)
            )
            for split in ("train", "val"):
                image_dir = root / "images" / split
                label_dir = root / "labels" / split
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                cv2.imwrite(
                    str(image_dir / "sample.png"),
                    np.zeros((32, 64, 3), dtype=np.uint8),
                )
                (label_dir / "sample.txt").write_text(
                    all_classes + "\n", encoding="utf-8"
                )
            summary = validate_dataset(root)
            validate_class_coverage(summary)
            distribution = format_class_distribution(summary)
            self.assertIn("rhythm_bar", distribution)
            self.assertIn("train=1 val=1", distribution)

    def test_classic_full_screenshots_keep_low_confidence_arrows(self) -> None:
        expected = {
            "Snipaste_2026-09-10_15-17-53.png": ["LEFT"],
            "Snipaste_2026-09-10_15-22-54.png": ["UP", "DOWN"],
            "Snipaste_2026-09-10_15-23-40.png": [
                "DOWN",
                "RIGHT",
                "UP",
                "RIGHT",
            ],
        }
        # These legacy fixtures were captured with the original manually selected ROI.
        x, y, width, height = (
            0.17239583333333333,
            0.7212962962962963,
            0.6375,
            0.22870370370370371,
        )
        for filename, sequence in expected.items():
            with self.subTest(filename=filename):
                image = read_image(
                    PROJECT_ROOT
                    / "tests"
                    / "fixtures"
                    / "traditional_four_key"
                    / "classic"
                    / filename
                )
                image_height, image_width = image.shape[:2]
                crop = image[
                    round(image_height * y) : round(image_height * (y + height)),
                    round(image_width * x) : round(image_width * (x + width)),
                ]
                detections = self.classic_detector.detect(crop)
                self.assertEqual(sequence, [item.direction for item in detections])
                self.assertTrue(
                    all(item.score >= 0.35 for item in detections)
                )

    def test_configured_hotkeys_do_not_require_shift(self) -> None:
        config = load_config()
        self.assertEqual("Ctrl+F9", config["hotkeys"]["select_roi"])
        self.assertEqual("Ctrl+F10", config["hotkeys"]["start"])
        self.assertEqual("Ctrl+F11", config["hotkeys"]["stop"])
        self.assertNotIn("pause_resume", config["hotkeys"])
        for value in config["hotkeys"].values():
            self.assertIn("CTRL+", value.upper())
            self.assertNotIn("SHIFT", value.upper())
            modifiers, virtual_key = parse_hotkey(value)
            self.assertGreater(modifiers, 0)
            self.assertGreater(virtual_key, 0)

    def test_roi_is_saved_and_loaded(self) -> None:
        expected = {"monitor": 1, "arrow_roi": [0.1, 0.2, 0.6, 0.15]}
        with TemporaryDirectory() as directory:
            temporary_config = PROJECT_ROOT.__class__(directory) / "config.json"
            with patch.object(config_module, "CONFIG_PATH", temporary_config):
                config_module.save_config(expected)
                self.assertEqual(expected, config_module.load_config())

    def test_status_header_does_not_cover_recognition_image(self) -> None:
        image = read_image(self.templates / "左箭头-未按下.png")
        result = draw_status(
            image,
            "RUNNING",
            60.0,
            3,
            3,
            ("LEFT",),
            False,
            "READY",
            None,
            recording_status=None,
            display_scale=0.5,
        )
        self.assertEqual(round(image.shape[0] * 0.5) + 243, result.shape[0])
        self.assertEqual(round(image.shape[1] * 0.5), result.shape[1])

    def test_preview_applies_topmost_only_when_state_changes(self) -> None:
        preview = PreviewWindow(0.5, True)
        preview._created = True
        with patch("dance_tool.live.set_preview_topmost", return_value=True) as setter:
            preview.apply_topmost()
            preview.apply_topmost()
            preview.always_on_top = False
            preview.apply_topmost()
        self.assertEqual(
            [call(True), call(False)],
            setter.call_args_list,
        )

    def test_preview_restores_without_activation_before_forcing_topmost(self) -> None:
        preview = PreviewWindow(0.5, True)
        preview._created = True
        with (
            patch(
                "dance_tool.live.show_preview_without_activation",
                return_value=True,
            ) as restore,
            patch.object(preview, "apply_topmost") as apply_topmost,
        ):
            self.assertTrue(preview.restore_without_activation())
        restore.assert_called_once_with()
        apply_topmost.assert_called_once_with(force=True)

    def test_mode_selector_only_accepts_clicks_while_stopped(self) -> None:
        selector = ModeSelector()
        canvas = np.zeros((243, 700, 3), dtype=np.uint8)
        selector.draw(canvas, "STOPPED", "traditional_four_key", "classic")
        speed_button = next(
            button
            for button in selector.buttons
            if button.choice.value == "speed_dance"
        )
        selector.on_mouse(
            cv2.EVENT_LBUTTONUP,
            (speed_button.left + speed_button.right) // 2,
            (speed_button.top + speed_button.bottom) // 2,
            0,
            None,
        )
        event = selector.poll()[0]
        self.assertEqual("game_mode", event.kind)
        self.assertEqual("speed_dance", event.value)

        selector.draw(canvas, "STOPPED", "traditional_four_key", "classic")
        renewed_button = next(
            button
            for button in selector.buttons
            if button.choice.value == "renewed"
        )
        selector.on_mouse(
            cv2.EVENT_LBUTTONUP,
            (renewed_button.left + renewed_button.right) // 2,
            (renewed_button.top + renewed_button.bottom) // 2,
            0,
            None,
        )
        event = selector.poll()[0]
        self.assertEqual("ui_mode", event.kind)
        self.assertEqual("renewed", event.value)

        selector.draw(canvas, "RUNNING", "traditional_four_key", "classic")
        selector.on_mouse(
            cv2.EVENT_LBUTTONUP,
            (speed_button.left + speed_button.right) // 2,
            (speed_button.top + speed_button.bottom) // 2,
            0,
            None,
        )
        self.assertEqual("blocked", selector.poll()[0].kind)

    def test_yolo_toggle_is_right_of_split_and_blocked_while_running(self) -> None:
        selector = ModeSelector()
        canvas = np.zeros((243, 700, 3), dtype=np.uint8)
        selector.draw(
            canvas,
            "STOPPED",
            "traditional_four_key",
            "classic",
            yolo_enabled=False,
        )
        yolo = next(
            button for button in selector.buttons if button.kind == "yolo_toggle"
        )
        split = next(
            button for button in selector.buttons if button.kind == "dataset_split"
        )
        self.assertGreater(yolo.left, split.right)
        selector.on_mouse(
            cv2.EVENT_LBUTTONUP,
            (yolo.left + yolo.right) // 2,
            (yolo.top + yolo.bottom) // 2,
            0,
            None,
        )
        event = selector.poll()[0]
        self.assertEqual("yolo_toggle", event.kind)
        self.assertEqual("true", event.value)

        selector.draw(
            canvas,
            "RUNNING",
            "traditional_four_key",
            "classic",
            yolo_enabled=True,
        )
        yolo = next(
            button for button in selector.buttons if button.kind == "yolo_toggle"
        )
        selector.on_mouse(
            cv2.EVENT_LBUTTONUP,
            (yolo.left + yolo.right) // 2,
            (yolo.top + yolo.bottom) // 2,
            0,
            None,
        )
        blocked = selector.poll()[0]
        self.assertEqual("blocked", blocked.kind)
        self.assertEqual("yolo", blocked.value)

    def test_config_reload_is_right_of_yolo_and_only_enabled_when_stopped(self) -> None:
        selector = ModeSelector()
        canvas = np.zeros((243, 700, 3), dtype=np.uint8)
        selector.draw(canvas, "STOPPED", "traditional_four_key", "classic")
        yolo = next(
            button for button in selector.buttons if button.kind == "yolo_toggle"
        )
        reload_button = next(
            button for button in selector.buttons if button.kind == "config_reload"
        )
        self.assertGreater(reload_button.left, yolo.right)
        selector.on_mouse(
            cv2.EVENT_LBUTTONUP,
            (reload_button.left + reload_button.right) // 2,
            (reload_button.top + reload_button.bottom) // 2,
            0,
            None,
        )
        self.assertEqual("config_reload", selector.poll()[0].kind)

        selector.draw(canvas, "RUNNING", "traditional_four_key", "classic")
        reload_button = next(
            button for button in selector.buttons if button.kind == "config_reload"
        )
        selector.on_mouse(
            cv2.EVENT_LBUTTONUP,
            (reload_button.left + reload_button.right) // 2,
            (reload_button.top + reload_button.bottom) // 2,
            0,
            None,
        )
        blocked = selector.poll()[0]
        self.assertEqual("blocked", blocked.kind)
        self.assertEqual("config_reload", blocked.value)

    def test_topmost_button_cycles_all_modes_even_while_running(self) -> None:
        selector = ModeSelector()
        canvas = np.zeros((243, 700, 3), dtype=np.uint8)
        expected = (
            ("off", "always"),
            ("always", "running"),
            ("running", "off"),
        )
        for current, next_mode in expected:
            selector.draw(
                canvas,
                "RUNNING",
                "traditional_four_key",
                "classic",
                topmost_mode=current,
            )
            topmost = next(
                button for button in selector.buttons if button.kind == "topmost_mode"
            )
            self.assertEqual(8, topmost.top)
            self.assertEqual(canvas.shape[1] - 10, topmost.right)
            selector.on_mouse(
                cv2.EVENT_LBUTTONUP,
                (topmost.left + topmost.right) // 2,
                (topmost.top + topmost.bottom) // 2,
                0,
                None,
            )
            event = selector.poll()[0]
            self.assertEqual("topmost_mode", event.kind)
            self.assertEqual(next_mode, event.value)

    def test_topmost_mode_supports_legacy_boolean_and_running_state(self) -> None:
        self.assertEqual("always", normalize_topmost_mode({"always_on_top": True}))
        self.assertEqual("off", normalize_topmost_mode({"always_on_top": False}))
        self.assertFalse(topmost_enabled_for_state("running", "STOPPED"))
        self.assertTrue(topmost_enabled_for_state("running", "RUNNING"))
        self.assertTrue(topmost_enabled_for_state("always", "STOPPED"))
        self.assertFalse(topmost_enabled_for_state("off", "RUNNING"))
        with self.assertRaisesRegex(ValueError, "topmost_mode"):
            normalize_topmost_mode({"topmost_mode": "sometimes"})

    def test_topmost_hotkey_is_removed_from_config(self) -> None:
        self.assertNotIn("toggle_topmost", self.saved_config["hotkeys"])
        self.assertIn(
            self.saved_config["window"]["topmost_mode"],
            {"off", "always", "running"},
        )

    def test_changed_config_paths_reports_only_modified_json_values(self) -> None:
        previous = {
            "window": {"max_fps": 60, "scale": 0.5},
            "space": {"cursor_window_ratio": 0.6},
        }
        current = {
            "window": {"max_fps": 90, "scale": 0.5},
            "space": {"cursor_window_ratio": 0.62},
            "dataset": {"split": "train"},
        }
        self.assertEqual(
            {
                "window.max_fps",
                "space.cursor_window_ratio",
                "dataset",
            },
            changed_config_paths(previous, current),
        )

    def test_yolo_runtime_settings_can_update_without_reloading_model(self) -> None:
        settings = YoloRuntimeSettings.from_config(
            {
                "confidence": 0.61,
                "slider_confidence": 0.27,
                "iou": 0.4,
                "image_size": 704,
                "max_detections": 20,
            }
        )
        self.assertEqual(0.61, settings.confidence)
        self.assertEqual(0.27, settings.slider_confidence)
        self.assertEqual(0.4, settings.iou)
        self.assertEqual(704, settings.image_size)
        self.assertEqual(20, settings.max_detections)

    def test_dataset_split_toggle_is_allowed_while_running_but_not_capturing(self) -> None:
        selector = ModeSelector()
        canvas = np.zeros((243, 700, 3), dtype=np.uint8)
        selector.draw(
            canvas,
            "STOPPED",
            "traditional_four_key",
            "classic",
            dataset_split="train",
        )
        split = next(
            button for button in selector.buttons if button.kind == "dataset_split"
        )
        selector.on_mouse(
            cv2.EVENT_LBUTTONUP,
            (split.left + split.right) // 2,
            (split.top + split.bottom) // 2,
            0,
            None,
        )
        event = selector.poll()[0]
        self.assertEqual("dataset_split", event.kind)
        self.assertEqual("val", event.value)

        selector.draw(
            canvas,
            "RUNNING",
            "traditional_four_key",
            "classic",
            dataset_split="val",
        )
        split = next(
            button for button in selector.buttons if button.kind == "dataset_split"
        )
        selector.on_mouse(
            cv2.EVENT_LBUTTONUP,
            (split.left + split.right) // 2,
            (split.top + split.bottom) // 2,
            0,
            None,
        )
        event = selector.poll()[0]
        self.assertEqual("dataset_split", event.kind)
        self.assertEqual("train", event.value)

        selector.draw(
            canvas,
            "RUNNING",
            "traditional_four_key",
            "classic",
            dataset_split="val",
            dataset_split_enabled=False,
        )
        split = next(
            button for button in selector.buttons if button.kind == "dataset_split"
        )
        selector.on_mouse(
            cv2.EVENT_LBUTTONUP,
            (split.left + split.right) // 2,
            (split.top + split.bottom) // 2,
            0,
            None,
        )
        blocked = selector.poll()[0]
        self.assertEqual("blocked", blocked.kind)
        self.assertEqual("dataset_split", blocked.value)

    def test_reaction_delay_has_hard_110ms_floor(self) -> None:
        timing = InputTiming.from_config(
            {
                "reaction_delay_ms": {"min": 20, "max": 50},
                "key_hold_ms": {"min": 25, "max": 45},
                "inter_key_delay_ms": {"min": 35, "max": 75},
                "clear_frames_to_rearm": 3,
            }
        )
        self.assertEqual(110, timing.reaction_min_ms)
        self.assertEqual(110, timing.reaction_max_ms)
        self.assertGreaterEqual(timing.random_reaction_seconds(), 0.110)

    def test_space_offset_is_normal_and_clipped(self) -> None:
        timing = SpaceTimingConfig.from_config(
            {
                "mean_offset_ms": 2,
                "stddev_ms": 100,
                "max_abs_offset_ms": 12,
            }
        )
        values = [timing.sample_offset_ms(random.Random(seed)) for seed in range(100)]
        self.assertTrue(all(-12 <= value <= 12 for value in values))
        self.assertTrue(any(value == -12 for value in values))
        self.assertTrue(any(value == 12 for value in values))

    def test_space_cycle_ignores_brief_arrow_detection_gaps(self) -> None:
        # Arrow visibility is deliberately not part of the expiry decision.
        self.assertIsNone(space_expire_reason(10.0, 10.08, 2200))
        self.assertIsNone(space_expire_reason(10.0, 12.19, 2200))
        self.assertEqual("timeout 2210ms", space_expire_reason(10.0, 12.21, 2200))

    def test_slider_tracker_predicts_fixed_line_crossing(self) -> None:
        timing = SpaceTimingConfig.from_config(
            resolve_mode_config(
                self.saved_config, "traditional_four_key", "renewed"
            )["space"]
        )
        tracker = SliderTracker(timing)
        observation = None
        for index, marker_x in enumerate((600, 620, 640, 660)):
            frame = np.zeros((246, 1224, 3), dtype=np.uint8)
            observation = tracker.observe(
                frame,
                index * 0.04,
                detected_slider_rect=(marker_x - 10, 30, marker_x + 10, 50),
                detected_slider_score=0.95,
                allow_template_fallback=False,
            )
        self.assertIsNotNone(observation)
        expected_target = frame.shape[1] * timing.cursor_window_ratio
        self.assertAlmostEqual(expected_target, observation.target_x, delta=0.01)
        self.assertAlmostEqual(660, observation.marker_x, delta=3)
        self.assertAlmostEqual(500, observation.speed_px_per_second, delta=40)
        self.assertAlmostEqual(
            (expected_target - 660) / 500,
            observation.crossing_at - 0.12,
            delta=0.04,
        )

        obscured = np.zeros_like(frame)
        cached = tracker.observe(obscured, 0.20, allow_template_fallback=False)
        self.assertIsNone(cached.marker_x)
        self.assertTrue(cached.prediction_cached)
        self.assertAlmostEqual(observation.crossing_at, cached.crossing_at, delta=0.001)

    def test_slider_tracker_uses_window_width_for_fixed_cursor(self) -> None:
        timing = SpaceTimingConfig.from_config(self.classic_config["space"])
        tracker = SliderTracker(timing)
        observation = None
        frame = np.zeros((246, 1224, 3), dtype=np.uint8)
        for index, marker_x in enumerate((500, 520, 540, 560)):
            observation = tracker.observe(
                frame,
                index * 0.04,
                detected_slider_rect=(marker_x - 10, 30, marker_x + 10, 50),
                detected_slider_score=0.95,
                allow_template_fallback=False,
            )
        self.assertIsNotNone(observation)
        self.assertAlmostEqual(
            1224 * timing.cursor_window_ratio,
            observation.target_x,
            delta=0.01,
        )
        self.assertAlmostEqual(560, observation.marker_x, delta=1)
        self.assertAlmostEqual(500, observation.speed_px_per_second, delta=40)

        wide_frame = np.zeros((253, 1920, 3), dtype=np.uint8)
        wide = SliderTracker(timing).observe(
            wide_frame,
            0.0,
            allow_template_fallback=False,
        )
        self.assertAlmostEqual(
            1920 * timing.cursor_window_ratio,
            wide.target_x,
            delta=0.01,
        )

    def test_yolo_slider_mode_does_not_call_opencv_fallback(self) -> None:
        timing = SpaceTimingConfig.from_config(self.classic_config["space"])
        tracker = SliderTracker(timing)
        frame = np.zeros((246, 1224, 3), dtype=np.uint8)
        with patch.object(
            tracker,
            "_find_slider",
            side_effect=AssertionError("不应调用 OpenCV 滑块模板"),
        ):
            observation = tracker.observe(
                frame,
                0.0,
                detected_slider_rect=(490, 30, 510, 50),
                detected_slider_score=0.9,
                allow_template_fallback=False,
            )
        self.assertEqual(500, observation.marker_x)

    def test_slider_at_fixed_line_triggers_current_frame_without_speed(self) -> None:
        timing = SpaceTimingConfig.from_config(self.classic_config["space"])
        tracker = SliderTracker(timing)
        frame = np.zeros((253, 1920, 3), dtype=np.uint8)
        target = frame.shape[1] * timing.cursor_window_ratio
        observation = tracker.observe(
            frame,
            1.25,
            detected_slider_rect=(
                round(target - 8),
                30,
                round(target + 8),
                50,
            ),
            detected_slider_score=0.95,
            allow_template_fallback=False,
        )
        self.assertAlmostEqual(target, observation.marker_x, delta=1)
        self.assertEqual(1.25, observation.crossing_at)

    def test_send_input_structure_matches_64bit_windows_abi(self) -> None:
        import ctypes

        expected_size = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
        self.assertEqual(expected_size, ctypes.sizeof(INPUT))
        self.assertIsInstance(is_admin(), bool)

    @patch("dance_tool.keyboard_input.modifiers_released", return_value=True)
    @patch("dance_tool.keyboard_input.is_foreground", return_value=True)
    @patch("dance_tool.keyboard_input.send_scan_code")
    def test_direction_sender_preserves_sequence(
        self, send_mock, _foreground_mock, _modifiers_mock
    ) -> None:
        timing = InputTiming(
            reaction_min_ms=110,
            reaction_max_ms=110,
            key_hold_min_ms=1,
            key_hold_max_ms=1,
            inter_key_min_ms=0,
            inter_key_max_ms=0,
            clear_frames_to_rearm=3,
            fallback_rearm_ms=350,
        )
        sender = DirectionKeySender(timing)
        self.assertTrue(sender.start(("LEFT", "UP"), target_hwnd=123))
        deadline = time.perf_counter() + 1.0
        while sender.busy and time.perf_counter() < deadline:
            time.sleep(0.005)
        events = sender.poll()
        self.assertEqual(["started", "completed"], [event for event, _ in events])
        self.assertEqual("completed", events[-1][0])
        self.assertEqual(
            [
                (SCAN_CODES["LEFT"], False),
                (SCAN_CODES["LEFT"], True),
                (SCAN_CODES["UP"], False),
                (SCAN_CODES["UP"], True),
            ],
            [(call.args[0], call.kwargs["key_up"]) for call in send_mock.call_args_list],
        )
        sender.close()

    @patch("dance_tool.keyboard_input.modifiers_released", return_value=True)
    @patch("dance_tool.keyboard_input.is_foreground", return_value=True)
    @patch("dance_tool.keyboard_input.send_scan_code")
    def test_direction_sender_honors_reaction_delay(
        self, _send_mock, _foreground_mock, _modifiers_mock
    ) -> None:
        timing = InputTiming(
            reaction_min_ms=110,
            reaction_max_ms=110,
            key_hold_min_ms=1,
            key_hold_max_ms=1,
            inter_key_min_ms=0,
            inter_key_max_ms=0,
            clear_frames_to_rearm=3,
            fallback_rearm_ms=350,
        )
        sender = DirectionKeySender(timing)
        first_seen = time.perf_counter()
        sender.start(
            ("DOWN",),
            target_hwnd=123,
            delay_seconds=0.110,
            first_seen_at=first_seen,
        )
        deadline = time.perf_counter() + 1.0
        while sender.busy and time.perf_counter() < deadline:
            time.sleep(0.005)
        events = sender.poll()
        started = next(payload for event, payload in events if event == "started")
        self.assertGreaterEqual(started["actual_reaction_ms"], 110)
        self.assertLess(started["actual_reaction_ms"], 250)
        sender.close()

    @patch("dance_tool.keyboard_input.modifiers_released", return_value=True)
    @patch("dance_tool.keyboard_input.is_foreground", return_value=True)
    @patch("dance_tool.keyboard_input.send_scan_code")
    def test_space_sender_uses_non_extended_scan_code(
        self, send_mock, _foreground_mock, _modifiers_mock
    ) -> None:
        sender = SpaceKeySender(hold_min_ms=1, hold_max_ms=1)
        now = time.perf_counter()
        self.assertTrue(
            sender.start_at(
                now,
                target_hwnd=123,
                predicted_crossing_at=now,
                sampled_offset_ms=0,
            )
        )
        deadline = time.perf_counter() + 1.0
        while sender.busy and time.perf_counter() < deadline:
            time.sleep(0.005)
        events = sender.poll()
        self.assertEqual("completed", events[-1][0])
        self.assertEqual(
            [
                (SCAN_CODES["SPACE"], False, False),
                (SCAN_CODES["SPACE"], True, False),
            ],
            [
                (
                    call.args[0],
                    call.kwargs["key_up"],
                    call.kwargs["extended"],
                )
                for call in send_mock.call_args_list
            ],
        )
        sender.close()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import replace
import unittest
import time
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from dance_tool import config as config_module
from dance_tool.config import PROJECT_ROOT, load_config, resolve_mode_config
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
from dance_tool.live import draw_status, frame_limit_delay, space_expire_reason
from dance_tool.recorder import RoiVideoRecorder
from dance_tool.selector import ModeSelector
from dance_tool.space_timing import RhythmBarTracker, SpaceTimingConfig
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
        speed_dance = resolve_mode_config(self.saved_config, "飞车舞蹈", "经典")
        self.assertEqual("traditional_four_key", classic["game_mode"])
        self.assertEqual("classic", classic["ui_mode"])
        self.assertEqual("renewed", renewed["ui_mode"])
        self.assertEqual(0.31, classic["recognition"]["match_threshold"])
        self.assertEqual(0.42, renewed["recognition"]["match_threshold"])
        self.assertIn("classic", classic["space"]["bar_template"])
        self.assertIn("renewed", renewed["space"]["bar_template"])
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

    def test_yolo_frame_result_exposes_best_bar_and_slider(self) -> None:
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
        self.assertEqual("rhythm_bar", result.rhythm_bar.class_name)
        self.assertEqual((10, 5, 190, 25), result.rhythm_bar.box)
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
            with self.assertRaisesRegex(ValueError, "必须分别覆盖全部10类"):
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
        x, y, width, height = self.classic_config["arrow_roi"]
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

    def test_dataset_split_toggle_is_mouse_only_and_blocked_while_running(self) -> None:
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

    def test_rhythm_tracker_predicts_centre_crossing(self) -> None:
        timing = SpaceTimingConfig.from_config(
            resolve_mode_config(
                self.saved_config, "traditional_four_key", "renewed"
            )["space"]
        )
        tracker = RhythmBarTracker(timing)
        observation = None
        for index, marker_x in enumerate((600, 620, 640, 660)):
            frame = np.zeros((246, 1224, 3), dtype=np.uint8)
            cv2.rectangle(frame, (810, 35), (846, 59), (250, 250, 250), -1)
            cv2.circle(frame, (marker_x, 47), 13, (240, 225, 135), -1)
            observation = tracker.observe(frame, index * 0.04)
        self.assertIsNotNone(observation)
        self.assertAlmostEqual(828, observation.target_x, delta=3)
        self.assertAlmostEqual(660, observation.marker_x, delta=3)
        self.assertAlmostEqual(500, observation.speed_px_per_second, delta=40)
        self.assertAlmostEqual(0.336, observation.crossing_at - 0.12, delta=0.04)

        obscured = np.zeros_like(frame)
        cached = tracker.observe(obscured, 0.20)
        self.assertIsNone(cached.marker_x)
        self.assertTrue(cached.prediction_cached)
        self.assertAlmostEqual(observation.crossing_at, cached.crossing_at, delta=0.001)

    def test_rhythm_tracker_uses_yolo_bar_slider_and_calibrated_cursor(self) -> None:
        timing = SpaceTimingConfig.from_config(self.classic_config["space"])
        tracker = RhythmBarTracker(timing)
        observation = None
        frame = np.zeros((246, 1224, 3), dtype=np.uint8)
        for index, marker_x in enumerate((500, 520, 540, 560)):
            observation = tracker.observe(
                frame,
                index * 0.04,
                detected_bar_rect=(300, 30, 970, 50),
                detected_bar_score=0.9,
                detected_slider_rect=(marker_x - 10, 30, marker_x + 10, 50),
                detected_slider_score=0.95,
            )
        self.assertIsNotNone(observation)
        self.assertTrue(observation.bar_locked)
        expected_target = 300 + (970 - 300) * timing.cursor_bar_ratio
        self.assertAlmostEqual(expected_target, observation.target_x, delta=1)
        self.assertAlmostEqual(560, observation.marker_x, delta=1)
        self.assertAlmostEqual(500, observation.speed_px_per_second, delta=40)
        self.assertIsNone(observation.cursor_match_score)

        shifted = tracker.observe(
            frame,
            0.20,
            detected_bar_rect=(320, 30, 990, 50),
            detected_bar_score=0.9,
            detected_slider_rect=(570, 30, 590, 50),
            detected_slider_score=0.95,
        )
        expected_shifted = shifted.bar_rect[0] + (
            shifted.bar_rect[2] - shifted.bar_rect[0]
        ) * timing.cursor_bar_ratio
        self.assertAlmostEqual(expected_shifted, shifted.target_x, delta=0.01)

    def test_yolo_rhythm_mode_does_not_call_opencv_fallback(self) -> None:
        timing = SpaceTimingConfig.from_config(self.classic_config["space"])
        tracker = RhythmBarTracker(timing)
        frame = np.zeros((246, 1224, 3), dtype=np.uint8)
        with patch.object(
            tracker,
            "_find_slider",
            side_effect=AssertionError("不应调用 OpenCV 滑块模板"),
        ), patch.object(
            tracker,
            "_find_cursor",
            side_effect=AssertionError("不应调用 OpenCV 光标模板"),
        ):
            observation = tracker.observe(
                frame,
                0.0,
                detected_bar_rect=(300, 30, 970, 50),
                detected_bar_score=0.9,
                allow_template_fallback=False,
            )
        self.assertIsNone(observation.marker_x)

        empty_tracker = RhythmBarTracker(timing)
        with patch.object(
            empty_tracker,
            "_accept_bar_candidate",
            side_effect=AssertionError("不应调用 OpenCV 节奏条模板"),
        ):
            missing = empty_tracker.observe(
                frame,
                0.0,
                allow_template_fallback=False,
            )
        self.assertEqual((0, 0, 0, 0), missing.bar_rect)

    def test_rhythm_bar_low_pass_filter_reduces_box_jitter(self) -> None:
        timing = replace(
            SpaceTimingConfig.from_config(self.classic_config["space"]),
            bar_low_pass_alpha=0.01,
        )
        tracker = RhythmBarTracker(timing)
        frame = np.zeros((246, 1224, 3), dtype=np.uint8)
        first = tracker.observe(
            frame,
            0.0,
            detected_bar_rect=(300, 30, 970, 50),
            detected_bar_score=0.9,
        )
        second = tracker.observe(
            frame,
            0.02,
            detected_bar_rect=(318, 30, 988, 50),
            detected_bar_score=0.9,
        )
        third = tracker.observe(
            frame,
            0.04,
            detected_bar_rect=(318, 30, 988, 50),
            detected_bar_score=0.9,
        )
        self.assertEqual(300, first.bar_rect[0])
        self.assertEqual(300, second.bar_rect[0])
        self.assertEqual(300, third.bar_rect[0])
        self.assertAlmostEqual(300.2691, tracker._filtered_bar_box[0], places=3)
        self.assertLess(
            third.bar_rect[0] - first.bar_rect[0],
            318 - first.bar_rect[0],
        )

        outlier = tracker.observe(
            frame,
            0.06,
            detected_bar_rect=(380, 30, 1050, 50),
            detected_bar_score=0.9,
        )
        self.assertEqual(300, outlier.bar_rect[0])

    def test_rhythm_templates_locate_a_tight_bar_roi(self) -> None:
        path = PROJECT_ROOT / "recordings" / "标准.mp4"
        if not path.exists():
            self.skipTest("标准节奏条录像不存在")
        video = cv2.VideoCapture(str(path))
        renewed = resolve_mode_config(
            self.saved_config, "traditional_four_key", "renewed"
        )
        tracker = RhythmBarTracker(SpaceTimingConfig.from_config(renewed["space"]))
        observation = None
        try:
            video.set(cv2.CAP_PROP_POS_FRAMES, 33)
            for index in range(4):
                ok, frame = video.read()
                self.assertTrue(ok)
                observation = tracker.observe(frame, index / 30)
        finally:
            video.release()
        self.assertIsNotNone(observation)
        left, top, right, bottom = observation.bar_rect
        self.assertTrue(observation.bar_locked)
        self.assertLess(right - left, 700)
        self.assertLess(bottom - top, 60)
        self.assertAlmostEqual(829, observation.target_x, delta=5)
        self.assertIsNotNone(observation.marker_x)

    def test_classic_rhythm_profile_is_calibrated_on_recording(self) -> None:
        path = PROJECT_ROOT / "recordings" / "roi_20260910_145748_937178.mp4"
        if not path.exists():
            self.skipTest("经典模式节奏条录像不存在")
        video = cv2.VideoCapture(str(path))
        tracker = RhythmBarTracker(
            SpaceTimingConfig.from_config(self.classic_config["space"])
        )
        observations = []
        try:
            for index in range(25):
                ok, frame = video.read()
                self.assertTrue(ok)
                observations.append(tracker.observe(frame, index / 30))
        finally:
            video.release()
        final = observations[-1]
        self.assertTrue(final.bar_locked)
        self.assertAlmostEqual(662, final.bar_rect[2] - final.bar_rect[0], delta=3)
        self.assertAlmostEqual(850, final.target_x, delta=4)
        self.assertGreater(final.cursor_match_score, 0.8)
        self.assertGreaterEqual(
            sum(item.marker_x is not None for item in observations), 15
        )

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

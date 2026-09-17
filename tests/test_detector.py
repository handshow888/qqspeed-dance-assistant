from __future__ import annotations

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
from dance_tool.detector import ArrowDetector
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
from dance_tool.live import draw_status, space_expire_reason
from dance_tool.recorder import RoiVideoRecorder
from dance_tool.selector import ModeSelector
from dance_tool.space_timing import RhythmBarTracker, SpaceTimingConfig
from dance_tool.yolo_dataset import (
    CLASS_NAMES,
    YoloDatasetCollector,
    detection_to_yolo_line,
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

    def test_mode_profiles_can_be_switched_without_changing_saved_mode(self) -> None:
        classic = resolve_mode_config(self.saved_config, "传统四键", "经典")
        renewed = resolve_mode_config(self.saved_config, "传统四键", "焕新")
        speed_dance = resolve_mode_config(self.saved_config, "飞车舞蹈", "经典")
        self.assertEqual("traditional_four_key", classic["game_mode"])
        self.assertEqual("classic", classic["ui_mode"])
        self.assertEqual("renewed", renewed["ui_mode"])
        self.assertEqual(0.35, classic["recognition"]["match_threshold"])
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
        self.assertEqual(8, len(CLASS_NAMES))

    def test_yolo_dataset_saves_named_events_with_empty_labels_supported(self) -> None:
        from dance_tool.detector import Detection

        with TemporaryDirectory() as directory:
            collector = YoloDatasetCollector(
                {
                    "enabled": True,
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
            self.assertTrue(label_files[-1].read_text(encoding="utf-8").startswith("5 "))
            empty_labels = [path for path in label_files if path.stat().st_size == 0]
            self.assertEqual(1, len(empty_labels))
            self.assertIn("bar_detected+space_pressed", empty_labels[0].name)

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
        self.assertEqual(round(image.shape[0] * 0.5) + 208, result.shape[0])
        self.assertEqual(round(image.shape[1] * 0.5), result.shape[1])

    def test_mode_selector_only_accepts_clicks_while_stopped(self) -> None:
        selector = ModeSelector()
        canvas = np.zeros((208, 700, 3), dtype=np.uint8)
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

import unittest

from src.drift.streaming_detectors import ADWIN


class ADWINCorrectnessTests(unittest.TestCase):
    def assert_window_alignment(self, detector):
        self.assertEqual(len(detector.window), len(detector.window_idx))
        self.assertGreaterEqual(len(detector.window), 0)

    def test_normal_update_and_alignment(self):
        detector = ADWIN(delta=0.002, max_window=100, min_window=5)

        self.assertFalse(detector.update(1.0))
        self.assertEqual(list(detector.window), [1.0])
        self.assertEqual(list(detector.window_idx), [0])
        self.assertEqual(detector.total_count, 1)
        self.assert_window_alignment(detector)

    def test_multiple_updates_without_drift(self):
        detector = ADWIN(delta=0.0000001, max_window=100, min_window=20)

        alarms = [detector.update(float(value)) for value in [0.1] * 39]

        self.assertFalse(any(alarms))
        self.assertFalse(detector.drift_detected)
        self.assertEqual(detector.last_cut_point, 0)
        self.assertEqual(detector.total_count, 39)
        self.assertEqual(list(detector.window_idx), list(range(39)))
        self.assert_window_alignment(detector)

    def test_maximum_window_pruning(self):
        detector = ADWIN(delta=0.002, max_window=5, min_window=2)

        for _ in range(8):
            detector.update(1.0)

        self.assertEqual(list(detector.window), [1.0] * 5)
        self.assertEqual(list(detector.window_idx), [3, 4, 5, 6, 7])
        self.assertEqual(detector.total_count, 8)
        self.assertEqual(detector.total_sum, 5.0)
        self.assert_window_alignment(detector)

    def test_drift_triggered_cut_and_alignment(self):
        detector = ADWIN(delta=0.0001, max_window=200, min_window=5)

        alarm = False
        for value in [0.0] * 30 + [10.0] * 30:
            alarm = detector.update(value)
            if alarm:
                break

        self.assertTrue(alarm)
        self.assertTrue(detector.drift_detected)
        self.assertGreater(detector.last_cut_point, 0)
        self.assert_window_alignment(detector)
        indices = list(detector.window_idx)
        self.assertEqual(indices, list(range(indices[0], indices[0] + len(indices))))
        self.assertEqual(len(detector.history_detected), 1)

    def test_drift_detected_is_updated_every_call(self):
        detector = ADWIN(delta=0.0001, max_window=200, min_window=5)

        alarm = False
        for value in [0.0] * 30 + [10.0] * 30:
            alarm = detector.update(value)
            if alarm:
                break

        self.assertTrue(alarm)
        self.assertFalse(detector.update(10.0))
        self.assertFalse(detector.drift_detected)

    def test_reset_clears_values_indices_and_state(self):
        detector = ADWIN(delta=0.002, max_window=5, min_window=2)
        for value in range(8):
            detector.update(float(value))

        detector.reset()

        self.assertEqual(len(detector.window), 0)
        self.assertEqual(len(detector.window_idx), 0)
        self.assertEqual(detector.total_sum, 0.0)
        self.assertEqual(detector.total_count, 0)
        self.assertFalse(detector.drift_detected)
        self.assertEqual(detector.last_cut_point, 0)
        self.assertEqual(detector.history_detected, [])
        self.assert_window_alignment(detector)

    def test_nonfinite_updates_do_not_change_the_stream(self):
        detector = ADWIN(delta=0.002, max_window=100, min_window=5)

        detector.update(1.0)
        detector.drift_detected = True
        result = detector.update(float("nan"))

        self.assertFalse(result)
        self.assertFalse(detector.drift_detected)
        self.assertEqual(detector.total_count, 1)
        self.assertEqual(list(detector.window), [1.0])
        self.assertEqual(list(detector.window_idx), [0])


if __name__ == "__main__":
    unittest.main()

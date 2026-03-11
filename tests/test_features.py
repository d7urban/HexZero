"""Tests for hexzero/features.py — extract_features."""
import unittest

import numpy as np

from hexzero.features import _SIZE_MAX, extract_features
from hexzero.game import HexState


class ShapeAndDtypeTests(unittest.TestCase):
    def test_output_shape(self):
        s = HexState(5)
        features, _size_norm = extract_features(s)
        self.assertEqual(features.shape, (8, 5, 5))
        self.assertEqual(features.dtype, np.float32)

    def test_output_shape_large(self):
        s = HexState(11)
        features, _ = extract_features(s)
        self.assertEqual(features.shape, (8, 11, 11))

    def test_all_planes_in_zero_one(self):
        s = HexState(5)
        s.apply_move((2, 2))
        s.apply_move((1, 1))
        features, _ = extract_features(s)
        self.assertGreaterEqual(float(features.min()), 0.0)
        self.assertLessEqual(float(features.max()), 1.0)

    def test_size_norm_range(self):
        for size in [5, 9, 11, 19]:
            with self.subTest(size=size):
                s = HexState(size)
                _, size_norm = extract_features(s)
                self.assertGreaterEqual(size_norm, 0.0)
                self.assertLessEqual(size_norm, 1.0)

    def test_size_norm_clipped_below(self):
        # Board smaller than _SIZE_MIN should clamp to 0
        s = HexState(2)
        _, size_norm = extract_features(s)
        self.assertEqual(size_norm, 0.0)

    def test_size_norm_clipped_above(self):
        # Board equal to _SIZE_MAX should give 1.0
        s = HexState(_SIZE_MAX)
        _, size_norm = extract_features(s)
        self.assertAlmostEqual(size_norm, 1.0)


class PlaneSemanticTests(unittest.TestCase):
    def _get_planes(self, size=5):
        s = HexState(size)
        s.apply_move((0, 0))  # BLACK at (0,0)
        s.apply_move((2, 2))  # WHITE at (2,2)
        return extract_features(s)[0], s

    def test_plane0_black_stones(self):
        features, _ = self._get_planes()
        self.assertEqual(features[0, 0, 0], 1.0)   # BLACK at (0,0)
        self.assertEqual(features[0, 2, 2], 0.0)   # no BLACK at (2,2)

    def test_plane1_white_stones(self):
        features, _ = self._get_planes()
        self.assertEqual(features[1, 2, 2], 1.0)   # WHITE at (2,2)
        self.assertEqual(features[1, 0, 0], 0.0)   # no WHITE at (0,0)

    def test_stone_planes_sum_le_one(self):
        # Each cell can be BLACK or WHITE but not both
        features, _ = self._get_planes()
        overlap = features[0] + features[1]
        self.assertLessEqual(float(overlap.max()), 1.0)

    def test_empty_board_stone_planes_all_zero(self):
        s = HexState(5)
        features, _ = extract_features(s)
        np.testing.assert_array_equal(features[0], np.zeros((5, 5)))
        np.testing.assert_array_equal(features[1], np.zeros((5, 5)))

    def test_component_plane_zero_when_no_stones(self):
        s = HexState(5)
        features, _ = extract_features(s)
        # Planes 6 (black_cc) and 7 (white_cc) should be all-zero
        np.testing.assert_array_equal(features[6], np.zeros((5, 5)))
        np.testing.assert_array_equal(features[7], np.zeros((5, 5)))

    def test_component_plane_single_stone_equals_one(self):
        s = HexState(5)
        s.apply_move((2, 2))  # one BLACK stone; only one component of size 1/1
        features, _ = extract_features(s)
        self.assertAlmostEqual(features[6, 2, 2], 1.0)

    def test_edge_distance_nonzero_on_empty_board(self):
        s = HexState(5)
        features, _ = extract_features(s)
        # BLACK edge distance: row 0 and row 4 should be 0 (touching target edge)
        self.assertEqual(features[4, 0, 0], 0.0)    # black_edge_dist row 0
        self.assertEqual(features[4, 4, 0], 0.0)    # black_edge_dist row 4
        # A middle cell has positive distance
        self.assertGreater(features[4, 2, 2], 0.0)


class SymmetryConsistencyTests(unittest.TestCase):
    """180° rotation of features must match features of rotated state."""

    def test_stone_planes_180_rotation(self):
        s = HexState(5)
        s.apply_move((0, 1))  # BLACK
        s.apply_move((3, 4))  # WHITE
        features, _ = extract_features(s)

        sym = s.apply_symmetry()
        sym_features, _ = extract_features(sym)

        # Rotating features 180° should match sym_features for stone planes
        for plane_idx in [0, 1]:
            rotated = np.rot90(features[plane_idx], 2)
            np.testing.assert_array_almost_equal(
                rotated, sym_features[plane_idx],
                err_msg=f"Plane {plane_idx} mismatch after 180° rotation",
            )


if __name__ == "__main__":
    unittest.main()

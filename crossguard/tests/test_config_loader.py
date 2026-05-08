from __future__ import annotations

import unittest
from pathlib import Path

from crossguard.defense.config_loader import load_invariant_config


class ConfigLoaderTests(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        path = Path(__file__).resolve().parents[1] / "config" / "crossguard_params.yaml"
        config = load_invariant_config(path)
        self.assertEqual(35.0, config.max_ground_speed_mps)
        self.assertEqual(2.0, config.perception_depth_tolerance_m)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import unittest

import pytest


BUILD_DIR = Path(__file__).parent.resolve() / "build"


class TestSpatialIndex(unittest.TestCase):
    def test_synapses(self):
        self.assertIn("element_type", d, msg=None)
        self.assertIn("in_memory", d, msg=None)
        self.assertIn("heavy_data_path", d["in_memory"], msg=None)
        self.assertIn("version", d, msg=None)
        self.assertIn("extended", d, msg=None)

        self.assertTrue(d["element_type"] == "synapse")
        self.assertTrue(d["in_memory"]["heavy_data_path"] == "index.spi")

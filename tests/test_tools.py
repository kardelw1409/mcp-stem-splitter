import unittest


class TestToolShapes(unittest.TestCase):
    def test_list_models_shape(self):
        from mcp_stem_splitter.server import list_models

        data = list_models()
        self.assertIsInstance(data, dict)
        self.assertIn("models", data)
        self.assertIn("presets", data)
        self.assertIn("notes", data)
        self.assertIsInstance(data["models"], list)
        self.assertIsInstance(data["presets"], list)
        self.assertIsInstance(data["notes"], str)

    def test_get_presets_shape(self):
        from mcp_stem_splitter.server import get_presets

        data = get_presets()
        self.assertIsInstance(data, dict)
        self.assertIn("presets", data)
        self.assertIsInstance(data["presets"], list)
        self.assertGreaterEqual(len(data["presets"]), 2)
        for preset in data["presets"]:
            self.assertIn("name", preset)
            self.assertIn("description", preset)
            self.assertIn("outputs", preset)

    def test_safe_prefix(self):
        from mcp_stem_splitter.server import _safe_filename_prefix

        self.assertEqual(_safe_filename_prefix("a:b*c"), "a_b_c")
        self.assertTrue(_safe_filename_prefix("   ") in {"track", ""})


if __name__ == "__main__":
    unittest.main()


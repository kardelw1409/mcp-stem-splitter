"""The model must be reachable on a fresh machine: no path to anyone's checkout.

These tests pin the portability contract that the tool depends on:

* the weights ship inside the package and match the pinned SHA-256 (a corrupted or
  half-copied blob must be rejected, not silently used - upstream would load RANDOM
  weights in that case and still produce a plausible MIDI),
* an explicit override wins, a broken override raises,
* no download happens when downloads are disabled,
* our own source carries no absolute drive-letter paths - the exact failure mode this
  module exists to prevent.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "mcp_stem_splitter"
OWN_MODULES = ("adt_of.py", "drums.py", "server.py", "__init__.py", "__main__.py")


class TestVendoredWeights(unittest.TestCase):
    def test_vendored_weights_exist_and_match_the_pinned_hash(self):
        from mcp_stem_splitter import adt_of

        path, source = adt_of.resolve_weights(allow_download=False)
        self.assertEqual(source, "vendored")
        self.assertTrue(path.is_file())
        self.assertEqual(adt_of.sha256_file(path), adt_of.WEIGHTS_SHA256)
        self.assertGreater(path.stat().st_size, 1_000_000)

    def test_vendored_module_imports_without_an_installed_copy(self):
        """The vendored code path must not depend on the pip-installed package."""
        from mcp_stem_splitter import adt_of

        function, implementation = adt_of.load_transcribe_function()
        self.assertEqual(implementation, "vendored")
        self.assertTrue(callable(function))


class TestResolutionOrder(unittest.TestCase):
    def test_env_override_wins(self):
        from mcp_stem_splitter import adt_of

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "my_weights.pth"
            shutil.copyfile(adt_of.resolve_weights()[0], target)
            previous = os.environ.get(adt_of.ENV_WEIGHTS)
            os.environ[adt_of.ENV_WEIGHTS] = str(target)
            try:
                path, source = adt_of.resolve_weights()
            finally:
                if previous is None:
                    os.environ.pop(adt_of.ENV_WEIGHTS, None)
                else:
                    os.environ[adt_of.ENV_WEIGHTS] = previous
        self.assertEqual(source, "env")
        self.assertEqual(Path(path), target)

    def test_missing_env_override_raises(self):
        from mcp_stem_splitter import adt_of

        previous = os.environ.get(adt_of.ENV_WEIGHTS)
        os.environ[adt_of.ENV_WEIGHTS] = str(Path(tempfile.gettempdir()) / "definitely-absent.pth")
        try:
            with self.assertRaises(adt_of.WeightsNotFound):
                adt_of.resolve_weights()
        finally:
            if previous is None:
                os.environ.pop(adt_of.ENV_WEIGHTS, None)
            else:
                os.environ[adt_of.ENV_WEIGHTS] = previous

    def test_corrupted_weights_are_rejected(self):
        from mcp_stem_splitter import adt_of

        with tempfile.TemporaryDirectory() as tmp:
            bogus = Path(tmp) / "corrupt.pth"
            bogus.write_bytes(b"not a checkpoint")
            previous = os.environ.get(adt_of.ENV_WEIGHTS)
            os.environ[adt_of.ENV_WEIGHTS] = str(bogus)
            try:
                with self.assertRaises(adt_of.WeightsNotFound):
                    adt_of.resolve_weights()
            finally:
                if previous is None:
                    os.environ.pop(adt_of.ENV_WEIGHTS, None)
                else:
                    os.environ[adt_of.ENV_WEIGHTS] = previous

    def test_no_download_when_disabled_and_nothing_local(self):
        from mcp_stem_splitter import adt_of

        original_vendored = adt_of.vendored_weights_path
        original_cache = adt_of.cache_dir
        with tempfile.TemporaryDirectory() as tmp:
            adt_of.vendored_weights_path = lambda: None          # simulate a slim install
            adt_of.cache_dir = lambda: Path(tmp)
            try:
                with self.assertRaises(adt_of.WeightsNotFound) as context:
                    adt_of.resolve_weights(allow_download=False)
            finally:
                adt_of.vendored_weights_path = original_vendored
                adt_of.cache_dir = original_cache
        self.assertIn("downloads are disabled", str(context.exception))

    def test_cache_dir_honours_the_env_override(self):
        from mcp_stem_splitter import adt_of

        with tempfile.TemporaryDirectory() as tmp:
            previous = os.environ.get(adt_of.ENV_CACHE)
            os.environ[adt_of.ENV_CACHE] = tmp
            try:
                self.assertEqual(adt_of.cache_dir(), Path(tmp))
            finally:
                if previous is None:
                    os.environ.pop(adt_of.ENV_CACHE, None)
                else:
                    os.environ[adt_of.ENV_CACHE] = previous


class TestNoMachineSpecificPaths(unittest.TestCase):
    """A drive-letter path in our own code would break the next machine silently.

    The lookbehind keeps URL schemes out of the result: in "https://" the character
    before the colon is a letter, so it is not mistaken for a drive letter ("s:").
    """

    DRIVE_PATH = re.compile(r"(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/]")

    def test_own_modules_have_no_absolute_paths(self):
        offenders = {}
        for name in OWN_MODULES:
            path = PACKAGE / name
            if not path.is_file():
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if self.DRIVE_PATH.search(line):
                    offenders.setdefault(name, []).append((number, line.strip()[:90]))
        self.assertEqual(offenders, {}, f"absolute paths found: {offenders}")

    def test_tests_have_no_absolute_paths(self):
        offenders = {}
        for path in sorted((Path(__file__).parent).glob("*.py")):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if self.DRIVE_PATH.search(line):
                    offenders.setdefault(path.name, []).append((number, line.strip()[:90]))
        self.assertEqual(offenders, {}, f"absolute paths in tests: {offenders}")


if __name__ == "__main__":
    unittest.main()

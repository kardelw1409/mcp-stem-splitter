"""Packaging facts that a fresh install depends on.

Both tests here encode a defect that was measured rather than imagined:

* `mcp>=1.0.0` alone lets pip resolve mcp 2.x, where `FastMCP` became `MCPServer` and the
  decorator/run API changed - on a clean venv the server then fails to import at all. The
  upper bound is therefore load-bearing and is asserted against the version objects, not
  against the spelling of the string.
* `requirements.lock.txt` is meant to be copyable to another machine, so it must not record
  an editable install, a git URL or a path - all three are machine-specific.
"""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements.lock.txt"


class TestDeclaredDependencies(unittest.TestCase):
    def _requirements(self) -> list[Requirement]:
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        return [Requirement(text) for text in data["project"]["dependencies"]]

    def _find(self, name: str) -> Requirement:
        matches = [item for item in self._requirements() if item.name.lower() == name]
        self.assertEqual(len(matches), 1, f"expected exactly one {name} requirement, got {matches}")
        return matches[0]

    def test_mcp_requirement_rejects_version_2(self):
        """mcp 2.x renamed FastMCP, so the server cannot even be imported there."""
        specifier = self._find("mcp").specifier
        self.assertNotIn(Version("2.3.0"), specifier, "mcp 2.x must be excluded")
        self.assertNotIn(Version("2.0.0"), specifier, "the 2.x line must be excluded")
        self.assertIn(Version("1.30.0"), specifier, "current 1.x releases must still be allowed")
        self.assertIn(Version("1.25.0"), specifier, "the version this repo is tested on must be allowed")


class TestLockFile(unittest.TestCase):
    """The lock is documentation for another machine, so it must be machine-independent."""

    PIN = re.compile(r"^[A-Za-z0-9_.\-\[\]]+==[^\s;]+$")

    def setUp(self):
        self.assertTrue(LOCK.is_file(), f"{LOCK.name} is missing")
        self.lines = LOCK.read_text(encoding="utf-8").splitlines()

    def test_every_non_comment_line_is_a_pin(self):
        offenders = [
            line for line in self.lines if line.strip() and not line.startswith("#") and not self.PIN.match(line)
        ]
        self.assertEqual(offenders, [], f"not a plain name==version pin: {offenders}")

    def test_no_machine_specific_entries(self):
        markers = ("-e ", "file://", " @ ", "git+", "\\\\", "/home/", "/Users/")
        offenders = [
            line
            for line in self.lines
            if line.strip() and not line.startswith("#") and any(marker in line for marker in markers)
        ]
        self.assertEqual(offenders, [], f"machine-specific entries in the lock: {offenders}")

    def test_records_the_versions_that_matter(self):
        text = LOCK.read_text(encoding="utf-8")
        for name in ("mcp==", "torch==", "numpy==", "demucs==", "librosa==", "pretty_midi==", "soundfile=="):
            self.assertIn(name, text, f"{name} absent from the lock file")


if __name__ == "__main__":
    unittest.main()

"""The MCP surface of the splitter: the tools must survive being called over stdio.

The tool bodies that load a model are the ones this file is about. FastMCP calls a SYNCHRONOUS
tool directly in the event loop (measured in mcp's `func_metadata`: `fn(**arguments)`, no
thread), so a lazy `import torch` inside a tool runs in the loop and the call NEVER RETURNS -
measured: an MCP call that the identical in-process code answers in 84 s produced no reply in
5+ minutes, with the child idle, no CUDA context and no log output at all.
"""

from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mcp_stem_splitter import server  # noqa: E402


class TestHeavyToolsDoNotRunInTheEventLoop(unittest.TestCase):
    def test_the_tools_that_load_a_model_are_async(self) -> None:
        for name in ("drums_midi", "split_stems", "split_vocals_only"):
            tool = getattr(server, name)
            self.assertTrue(
                inspect.iscoroutinefunction(tool),
                "%s must be async: a sync tool runs inside the event loop, where a lazy import "
                "of torch deadlocks the call" % name,
            )

    def test_the_preload_actually_imports_the_modules_a_tool_would_lazy_load(self) -> None:
        imported = server._preimport_heavy()
        # a submodule of this package lands in sys.modules under its FULL dotted name
        for name in ("numpy", "soundfile", "pretty_midi", "torch", "mcp_stem_splitter.adt_of"):
            self.assertIn(name, sys.modules, "preload did not import %s" % name)
        for name in ("numpy", "soundfile", "pretty_midi", "torch", "adt_of"):
            self.assertIn(name, imported)
            self.assertNotIn(":", imported[name], "preload failed for %s: %s" % (name, imported[name]))

    def test_the_drum_tool_exposes_the_classes_it_can_restrict(self) -> None:
        """Measured trap: the core accepts model_classes but the tool did not declare it, so the
        argument was dropped in silence and a caller could not ask for the measured setting."""
        parameters = inspect.signature(server.drums_midi).parameters
        for name in ("model_classes", "detector_classes"):
            self.assertIn(name, parameters)
            self.assertEqual(parameters[name].default, None)

    def test_the_tool_declares_every_parameter_the_core_accepts(self) -> None:
        """The general guard for the trap above: a parameter the core accepts but the tool does
        not declare is unreachable over MCP - and worse, a caller who passes it anyway gets a
        silent drop. This test names every such parameter instead of discovering them one at a
        time in production."""
        from mcp_stem_splitter.drums import drums_midi_core

        core = set(inspect.signature(drums_midi_core).parameters)
        tool = set(inspect.signature(server.drums_midi).parameters)
        missing = sorted(core - tool)
        self.assertEqual(missing, [], "the tool does not declare: %s" % ", ".join(missing))

    def test_the_preload_is_called_before_the_server_takes_the_loop(self) -> None:
        source = inspect.getsource(server.run)
        self.assertIn("_preimport_heavy()", source)
        self.assertLess(
            source.index("_preimport_heavy()"),
            source.index("mcp.run()"),
            "the preload must happen before the loop starts",
        )


class TestModelClassRestrictionIsReported(unittest.TestCase):
    def test_a_restricted_model_class_shows_up_in_the_reply(self) -> None:
        """The wrapper must not let a restriction disappear: if the core's own report is empty,
        the restriction the caller asked for is what the reply states."""
        original = server.drums_midi.__globals__["_drums_midi_call"]

        class Result(dict):
            pass

        def fake_core(**kwargs):
            return Result({"ok": True, "notes": {"total": 1}})

        module = sys.modules["mcp_stem_splitter.drums"]
        saved = module.drums_midi_core
        module.drums_midi_core = fake_core
        try:
            result = original(
                None, "out", None, None, None, None, "balanced", None, True, True, 60.0, True,
                True, ["kick"], ["snare", "hihat"], 1.5, 3.0, None, "auto",
            )
        finally:
            module.drums_midi_core = saved
        self.assertEqual(result["model_classes"], ["hihat", "snare"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

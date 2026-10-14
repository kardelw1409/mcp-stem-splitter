"""The one test that would have caught the hang: a REAL stdio round-trip to the server.

Everything else in this suite calls the cores directly, which is precisely why the deadlock -
a lazy `import` executed inside the event loop / a worker thread - survived all of them. This
test starts the server the way an MCP client does and calls the heavy tool through the wire.

It needs the real track and its separated elements, so it SKIPS when they are absent: a test
that silently degrades into "we did not check" is worse than one that says it cannot run.

Measured history of this exact call:
  * sync tool, lazy import inside the loop  -> NO REPLY AT ALL in 5+ minutes (child idle,
    no CUDA context, nothing logged)
  * async tool, lazy import in a thread     -> hung after "Loading PyTorch weights" for 3+
    minutes; py-spy showed the worker thread inside
    `vendor/adtof_pytorch/audio.py load_audio -> import librosa -> scipy.linalg.blas`
  * async tool + preloaded import chain     -> 84 s in-process, and this test's answer arrives
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"

#: The heavy round-trip needs a real track and its separated elements. They come from the
#: ENVIRONMENT, not from a literal path: a drive letter in a test would break the next machine
#: silently (and this repository has a test that refuses one).
MIX = Path(os.environ.get("STEM_SPLITTER_TEST_MIX", ""))
ELEMENTS = Path(os.environ.get("STEM_SPLITTER_TEST_ELEMENTS", ""))
BPM = float(os.environ.get("STEM_SPLITTER_TEST_BPM", "0") or 0.0)

CLIENT = '''
import json, sys, time
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import anyio

ARGS = json.loads(sys.argv[1])


async def main() -> None:
    params = StdioServerParameters(command=ARGS["command"], args=ARGS["args"], cwd=ARGS["cwd"])
    started = time.perf_counter()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = [tool.name for tool in (await session.list_tools()).tools]
            print("TOOLS " + json.dumps(tools), flush=True)
            result = await session.call_tool("drums_midi", ARGS["arguments"])
            print("SECONDS %.1f" % (time.perf_counter() - started), flush=True)
            payload = result.structuredContent
            if payload is None:
                text = "".join(getattr(item, "text", "") for item in (result.content or []))
                payload = json.loads(text)
            print("PAYLOAD " + json.dumps({k: payload.get(k) for k in
                  ("ok", "notes", "kick_pitch_test", "model_classes", "detector_classes")}), flush=True)

anyio.run(main)
'''


@unittest.skipUnless(
    MIX.is_file() and ELEMENTS.is_dir(),
    "set STEM_SPLITTER_TEST_MIX and STEM_SPLITTER_TEST_ELEMENTS to a real track and its"
    " separated elements to run the round-trip",
)
class TestStdioRoundTrip(unittest.TestCase):
    def test_the_drum_tool_answers_over_stdio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = Path(tmp) / "client.py"
            client.write_text(CLIENT, encoding="utf-8")
            arguments = {
                "audio_path": str(MIX),
                "element_stems_dir": str(ELEMENTS),
                "output_dir": str(Path(tmp) / "drums"),
                "bpm": BPM or None,
                "model_classes": ["snare", "hihat"],
                "detector_classes": ["kick", "tom", "cymbal"],
            }
            config = {
                "command": str(PYTHON if PYTHON.exists() else sys.executable),
                "args": ["-m", "mcp_stem_splitter"],
                "cwd": str(REPO),
                "arguments": arguments,
            }
            env = dict(os.environ)
            env["STEM_SPLITTER_LOG_LEVEL"] = "WARNING"
            completed = subprocess.run(
                [str(PYTHON if PYTHON.exists() else sys.executable), str(client), json.dumps(config)],
                cwd=str(REPO),
                capture_output=True,
                text=True,
                timeout=900,
                env=env,
            )
        stdout = completed.stdout
        self.assertIn("TOOLS", stdout, "the server never listed its tools:\n%s\n%s" % (stdout, completed.stderr))
        self.assertIn('"drums_midi"', stdout)
        self.assertIn("PAYLOAD", stdout, "no answer came back:\n%s\n%s" % (stdout, completed.stderr))
        payload = json.loads(stdout.split("PAYLOAD", 1)[1].strip().splitlines()[0])
        self.assertTrue(payload["ok"], payload)
        self.assertGreater(payload["notes"]["total"], 100)
        # the restriction the caller asked for must be VISIBLE in the reply, not silently dropped
        # (the core reports classes in its own order, so the check is on the SET)
        self.assertEqual(sorted(payload["model_classes"]), ["hihat", "snare"])
        self.assertEqual(sorted(payload["detector_classes"]), ["cymbal", "kick", "tom"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Logic tests for the drum fusion tool - no audio models, no external data.

These pin the decisions that were expensive to establish:

* the pitch map must cover BOTH vocabularies (ADTOF labels kick as GM 35, ADT_STR as
  GM 36 - comparing raw pitches once made a working model look broken),
* off-kit percussion (tambourine/cowbell) must be dropped: they were 22 % of ADT_STR's
  output on a drum & bass track,
* velocity policy is per class, and the kick is deliberately flat (its model velocity
  correlates +0.09 with the audio, i.e. it is noise; a kick is a sample trigger),
* a single source with nothing to check against must be reported as such, not silently
  accepted as fact.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TestPitchVocabulary(unittest.TestCase):
    def test_kick_has_both_pitches(self):
        from mcp_stem_splitter.drums import PITCH_CLASS

        self.assertEqual(PITCH_CLASS[35], "kick")
        self.assertEqual(PITCH_CLASS[36], "kick")

    def test_hihat_and_cymbal_families(self):
        from mcp_stem_splitter.drums import PITCH_CLASS

        for pitch in (42, 44, 46):
            self.assertEqual(PITCH_CLASS[pitch], "hihat")
        for pitch in (49, 51, 52, 53, 55, 57, 59):
            self.assertEqual(PITCH_CLASS[pitch], "cymbal")

    def test_toms_are_a_family(self):
        from mcp_stem_splitter.drums import PITCH_CLASS

        for pitch in (41, 43, 45, 47, 48, 50):
            self.assertEqual(PITCH_CLASS[pitch], "tom")

    def test_off_kit_percussion_is_not_mapped(self):
        from mcp_stem_splitter.drums import OFF_KIT_PITCHES, PITCH_CLASS

        for pitch in (54, 56, 58):
            self.assertNotIn(pitch, PITCH_CLASS)
            self.assertIn(pitch, OFF_KIT_PITCHES)


class TestVelocityPolicy(unittest.TestCase):
    def test_kick_is_flat(self):
        from mcp_stem_splitter.drums import VELOCITY_POLICY

        kind, lo, hi = VELOCITY_POLICY["kick"]
        self.assertEqual(kind, "flat")
        self.assertEqual(lo, hi)

    def test_measured_classes_have_a_range(self):
        from mcp_stem_splitter.drums import VELOCITY_POLICY

        for cls in ("snare", "hihat", "tom", "cymbal"):
            kind, lo, hi = VELOCITY_POLICY[cls]
            self.assertEqual(kind, "measured", cls)
            self.assertLess(lo, hi, cls)

    def test_assign_velocity_flat_leaves_one_value(self):
        from mcp_stem_splitter.drums import _assign_velocity

        records = [{"class": "kick", "onset": float(i) * 0.5} for i in range(6)]
        _assign_velocity(records, signals={})
        self.assertEqual({rec["velocity"] for rec in records}, {100})


class TestFusion(unittest.TestCase):
    def test_agreement_wins_and_collisions_merge(self):
        from mcp_stem_splitter.drums import _cluster, _select

        notes = {
            "a": [("kick", 1.000), ("snare", 2.000)],
            "b": [("kick", 1.010)],          # same 1/16 slot as a's kick -> one cluster
        }
        clusters = _cluster(notes, 0.125)
        kick = [c for c in clusters if c["class"] == "kick"]
        self.assertEqual(len(kick), 1)
        self.assertEqual(kick[0]["votes"], 2)

        kept = _select(clusters, signals={}, mode="strict", min_support_db=1.5, lone_support_db=3.0)
        self.assertEqual(len(kept), 1)  # snare has a single vote -> dropped in strict

    def test_without_elements_support_is_reported_as_absent(self):
        from mcp_stem_splitter.drums import _cluster, _select

        clusters = _cluster({"only": [("kick", 1.0)]}, 0.125)
        kept = _select(clusters, signals={}, mode="full", min_support_db=1.5, lone_support_db=3.0)
        self.assertEqual(len(kept), 1)
        self.assertIsNone(kept[0]["support_db"])  # nothing to check against - say so

    def test_support_filters_a_note_that_has_no_energy(self):
        import numpy as np

        from mcp_stem_splitter.drums import _cluster, _select

        silent = {"band": np.zeros(1000, dtype=np.float32), "sr": 1000, "background": 1.0, "files": ["fake.wav"]}
        clusters = _cluster({"a": [("kick", 0.5)], "b": [("kick", 0.5)]}, 0.125)
        kept = _select(clusters, signals={"kick": silent}, mode="strict", min_support_db=1.5, lone_support_db=3.0)
        self.assertEqual(kept, [])  # two sources agreed, but the element is silent


class TestQuantize(unittest.TestCase):
    def test_snaps_within_tolerance_only(self):
        from mcp_stem_splitter.drums import _quantize

        records = [{"class": "kick", "onset": 0.128}]
        out, report = _quantize(records, bpm=120.0, max_shift_s=0.060, grid_origin_s=0.0)
        self.assertTrue(report["applied"])
        self.assertEqual(report["origin_source"], "explicit")
        self.assertAlmostEqual(out[0]["onset"], 0.125, places=4)

    def test_far_note_is_left_alone(self):
        from mcp_stem_splitter.drums import _quantize

        records = [{"class": "kick", "onset": 4.30}]
        out, _ = _quantize(records, bpm=120.0, max_shift_s=0.001, grid_origin_s=0.0)
        self.assertAlmostEqual(out[0]["onset"], 4.30, places=4)

    def test_explicit_origin_moves_a_note_the_estimated_one_would_keep(self):
        """A single note must NOT define its own grid (that is how a short file
        silently quantises to itself and reports success).

        At 120 BPM a 1/16 is 0.125 s. With origin 0.03125 the grid points are
        0.03125, 0.15625, 0.28125...; a note at 0.200 belongs at 0.15625 (43.75 ms pull,
        inside the 60 ms window), whereas the estimated origin would leave it alone.
        """
        from mcp_stem_splitter.drums import _quantize

        records = [{"class": "kick", "onset": 0.200}]
        estimated, report = _quantize([dict(records[0])], bpm=120.0, max_shift_s=0.060)
        self.assertEqual(report["origin_source"], "estimated_from_notes")
        self.assertAlmostEqual(estimated[0]["onset"], 0.200, places=4)  # snapped to itself

        explicit, report2 = _quantize([dict(records[0])], bpm=120.0, max_shift_s=0.060, grid_origin_s=0.03125)
        self.assertEqual(report2["origin_source"], "explicit")
        self.assertAlmostEqual(explicit[0]["onset"], 0.15625, places=4)  # pulled to the real grid

    def test_no_bpm_means_no_quantize(self):
        from mcp_stem_splitter.drums import _quantize

        records = [{"class": "kick", "onset": 1.234}]
        out, report = _quantize(records, bpm=None, max_shift_s=0.060)
        self.assertFalse(report["applied"])
        self.assertEqual(out[0]["onset"], 1.234)


class TestToolContract(unittest.TestCase):
    def test_requires_at_least_one_source(self):
        from mcp_stem_splitter.drums import drums_midi_core

        with self.assertRaises(ValueError):
            drums_midi_core(audio_path=None, output_dir=tempfile.mkdtemp(), extra_midi=None, element_stems_dir=None)

    def test_rejects_unknown_mode(self):
        from mcp_stem_splitter.drums import drums_midi_core

        with self.assertRaises(ValueError):
            drums_midi_core(extra_midi=["x.mid"], output_dir=tempfile.mkdtemp(), mode="nope")

    def test_per_class_files_are_written(self):
        import pretty_midi

        from mcp_stem_splitter.drums import drums_midi_core

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src.mid"
            midi = pretty_midi.PrettyMIDI()
            drums = pretty_midi.Instrument(program=0, is_drum=True)
            for start, pitch in ((0.5, 35), (1.0, 38), (1.5, 42), (2.0, 45), (2.5, 49)):
                drums.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.05))
            midi.instruments.append(drums)
            midi.write(str(source))

            out_dir = Path(tmp) / "out"
            result = drums_midi_core(audio_path=None, output_dir=str(out_dir), extra_midi=[str(source)], bpm=120.0)

            self.assertTrue(result["ok"])
            self.assertTrue(Path(result["midi"]["combined"]).exists())
            by_class = result["midi"]["by_class"]
            self.assertEqual(set(by_class), {"kick", "snare", "hihat", "tom", "cymbal"})
            for path in by_class.values():
                self.assertTrue(Path(path).exists())
                parsed = pretty_midi.PrettyMIDI(path)
                self.assertTrue(parsed.instruments[0].is_drum)
                self.assertEqual(len({n.pitch for n in parsed.instruments[0].notes}), 1)
            self.assertEqual(result["notes"]["total"], 5)

    def test_arbiter_block_is_present_without_elements(self):
        """Every reported key must exist on every path.

        A field that is only assigned inside the 'elements were given' branch crashes the
        call without elements - that happened for real with the duration report, which is
        why this checks the contract instead of the happy path.
        """
        import pretty_midi

        from mcp_stem_splitter.drums import drums_midi_core

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src.mid"
            midi = pretty_midi.PrettyMIDI()
            drums = pretty_midi.Instrument(program=0, is_drum=True)
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=35, start=1.0, end=1.05))
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=35, start=1.5, end=1.55))
            midi.instruments.append(drums)
            midi.write(str(source))

            result = drums_midi_core(audio_path=None, output_dir=str(Path(tmp) / "out"), extra_midi=[str(source)],
                                     bpm=120.0, element_stems_dir=None)
            self.assertTrue(result["ok"])
            for key in ("mode", "audio", "output_dir", "midi", "notes", "velocity", "grid", "arbiter", "sources", "warnings"):
                self.assertIn(key, result)
            self.assertEqual(result["arbiter"]["elements_used"], {})
            self.assertEqual(result["arbiter"]["durations"], {})


if __name__ == "__main__":
    unittest.main()

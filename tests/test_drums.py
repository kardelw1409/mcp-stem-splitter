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


class TestOnsetDetection(unittest.TestCase):
    """Why the detector was rewritten, pinned as behaviour.

    The old detector thresholded the flux at the 99th percentile of the WHOLE FILE, then
    thinned with a 60 ms refractory.  Measured on a real kick stem that combination produced
    onsets that were not hits: only 79 % of them sat at a rise of 6 dB or more, and only
    65 % of their intervals were multiples of a 1/16 (ringing inside one hit is not a
    subdivision of anything).  The kick was also detected in 25-120 Hz, where the same
    algorithm's count moved 620 -> 395 when the refractory was nudged - a count that is a
    property of a constant is not a measurement.
    """

    def _click_train(self, sr: int, hits, *, gain=1.0, noise=0.0, seconds=10.0):
        import numpy as np

        out = np.random.default_rng(0).normal(0.0, noise, int(seconds * sr)).astype(np.float32)
        for t in hits:
            i = int(t * sr)
            out[i:i + int(0.002 * sr)] += gain
        return out

    def test_the_refractory_follows_the_band_not_a_constant(self):
        """A band that admits 25 Hz rings every 40 ms; three cycles of that is 120 ms, so a
        60 ms refractory cannot separate a hit from its own ring.  A band whose lowest
        frequency is 60 Hz needs only 50 ms - the floor covers it."""
        from mcp_stem_splitter.drums import MIN_REFRACTORY_S, refractory_for_band

        self.assertEqual(refractory_for_band(25.0), 0.12)
        self.assertEqual(refractory_for_band(40.0), 0.075)
        self.assertEqual(refractory_for_band(60.0), MIN_REFRACTORY_S)
        self.assertEqual(refractory_for_band(6000.0), MIN_REFRACTORY_S)

    def test_a_ringing_hit_is_one_onset_not_five(self):
        """The defect in miniature: a decaying 40 Hz burst is one hit whose envelope bumps
        every 12 ms.  It must be reported once, at its start - not once per bump."""
        import numpy as np

        from mcp_stem_splitter.drums import _onset_times, refractory_for_band

        sr = 8000
        t = np.arange(int(6.0 * sr)) / sr
        signal = np.zeros_like(t)
        for start in (1.0, 2.0, 4.0):
            ring = np.exp(-(t - start) / 0.25) * ((t >= start) & (t < start + 1.0))
            signal += ring * np.sin(2 * np.pi * 40.0 * (t - start))
        band = signal.astype(np.float32)

        onsets = _onset_times(band, sr, min_gap=refractory_for_band(40.0))
        self.assertEqual(len(onsets), 3, f"three hits, got {len(onsets)} at {onsets}")
        for found, expected in zip(onsets, (1.0, 2.0, 4.0)):
            self.assertLess(abs(found - expected), 0.02, found)
        # and the shorter refractory the old code used cannot be better on the same signal
        loose = _onset_times(band, sr, min_gap=0.060)
        self.assertGreaterEqual(len(loose), len(onsets)), loose

    def test_silence_produces_no_onsets(self):
        """The local term alone goes to zero where there is no signal, so tiny flux passes
        and notes appear where nothing was hit (measured: 6 % of the onsets sat more than
        10 dB below the file's median).  The floor term is what stops that."""
        import numpy as np

        from mcp_stem_splitter.drums import _onset_times

        sr = 8000
        rng = np.random.default_rng(1)
        first = np.zeros(int(3.0 * sr), dtype=np.float32)
        for t in (0.5, 1.0, 1.5, 2.0, 2.5):
            first[int(t * sr):int(t * sr) + 20] = 1.0
        quiet = rng.normal(0.0, 1e-4, int(3.0 * sr)).astype(np.float32)
        band = np.concatenate([first, quiet])
        onsets = _onset_times(band, sr, min_gap=0.060)
        self.assertEqual(len(onsets), 5, f"the quiet half must contribute nothing: {onsets}")
        self.assertTrue(all(t < 3.0 for t in onsets), onsets)

    def test_a_quiet_passage_is_still_detected(self):
        """The other half of the same rule: the local term exists so that a section quieter
        than the loudest is not skipped by a threshold set for the loudest.  Measured on this
        fixture: a section 6 dB down is found in full, 10 dB down is still found, and past
        ~12 dB it is not - that is the floor's price, and it is why the floor is a fraction
        of a HIGH percentile rather than of the peak."""
        import numpy as np

        from mcp_stem_splitter.drums import _onset_times

        sr = 8000

        def section(amplitude):
            out = np.zeros(int(3.0 * sr), dtype=np.float32)
            for t in (0.5, 1.0, 1.5, 2.0, 2.5):
                out[int(t * sr):int(t * sr) + 20] = amplitude
            return out

        onsets = _onset_times(np.concatenate([section(1.0), section(0.32)]), sr, min_gap=0.060)
        in_quiet = [t for t in onsets if t >= 3.0]
        self.assertEqual(len(in_quiet), 5, f"a section 10 dB down must still be found: {onsets}")
        # and it must not invent hits that are not there: past the floor they are gone
        deep = _onset_times(np.concatenate([section(1.0), section(0.05)]), sr, min_gap=0.060)
        self.assertEqual(len([t for t in deep if t >= 3.0]), 0)

    def test_the_kick_is_timed_where_the_hit_is(self):
        """The kick's sub band is not a place to time a kick from: asking 'is there kick
        energy here' in 25-120 Hz gives +3.7 dB at the hits and +5.0 dB BETWEEN them, so the
        answer is never no and every note is 'supported'.  In the body band the same question
        separates them by +7.2 dB.  Both bands are therefore the body band now."""
        from mcp_stem_splitter.drums import DETECT_BANDS, ELEMENT_BANDS

        self.assertEqual(DETECT_BANDS["kick"], (60.0, 250.0))
        self.assertEqual(ELEMENT_BANDS["kick"][1], (60.0, 250.0))
        # the other classes keep the band their support test was measured with
        for cls in ("snare", "hihat", "tom", "cymbal"):
            self.assertEqual(DETECT_BANDS[cls], ELEMENT_BANDS[cls][1])

    def test_low_contrast_is_reported_not_hidden(self):
        """A band whose hits do not stand above a busy floor cannot be timed by thresholding
        it, and the report has to say so instead of presenting its onsets as measurements."""
        import numpy as np
        import soundfile as sf

        from mcp_stem_splitter.drums import LOW_CONTRAST_DB, drums_midi_core

        with tempfile.TemporaryDirectory() as tmp:
            elements = Path(tmp) / "elements"
            elements.mkdir()
            sr = 8000
            rng = np.random.default_rng(2)
            # a kick element that is one continuous noise floor: no hits, no contrast
            sf.write(str(elements / "kick.wav"), rng.normal(0, 0.2, sr * 4).astype(np.float32), sr)
            result = drums_midi_core(element_stems_dir=str(elements),
                                     output_dir=str(Path(tmp) / "out"), bpm=120.0,
                                     use_detector=True, allow_model_download=False)
            kick = result["detect_bands"]["kick"]
            self.assertLess(kick["contrast_db"], LOW_CONTRAST_DB)
            self.assertFalse(kick["stable"])
            self.assertTrue(any("depending" in w or "depend on the threshold" in w for w in result["warnings"]),
                            result["warnings"])


class TestKickPitchTest(unittest.TestCase):
    """A kick's fundamental FALLS; a bass note holds its note.

    This is the test that survived where every amplitude rule failed.  Measured on a real
    neurofunk stem (383 onsets) with the bass stem's own onsets as the control:

        kick onsets   median fall +35 Hz, 68 % >= 10 Hz, fundamental starts ~104 Hz
        bass onsets   median fall   0 Hz, 57 % <  5 Hz

    and in the one section that is measurably a leak (the kick stem correlates +0.756 with the
    bass stem and sits 10.4 dB below it) the onsets start at 37.7 Hz with 73 % of them flat.
    """

    def _hit(self, sr, t, *, f_start, f_end, dur=0.25, amp=1.0):
        import numpy as np

        n = int(dur * sr)
        tt = np.arange(n) / sr
        # instantaneous frequency sliding from f_start to f_end over the first 60 ms
        sweep = np.clip(tt / 0.060, 0.0, 1.0)
        freq = f_start + (f_end - f_start) * sweep
        phase = 2 * np.pi * np.cumsum(freq) / sr
        return (amp * np.sin(phase) * np.exp(-tt / 0.10)).astype(np.float32)

    def _track(self, sr, hits):
        import numpy as np

        out = np.zeros(int(sr * 6.0), dtype=np.float32)
        for t, kwargs in hits:
            hit = self._hit(sr, t, **kwargs)
            a = int(t * sr)
            out[a:a + len(hit)] += hit
        return out

    def test_a_falling_kick_passes_and_a_flat_bass_note_does_not(self):
        import numpy as np

        from mcp_stem_splitter.drums import KICK_SWEEP_MIN_HZ, pitch_drop_hz

        sr = 8000
        kick = self._hit(sr, 0.0, f_start=110.0, f_end=50.0)
        bass = self._hit(sr, 0.0, f_start=70.0, f_end=70.0)
        _k0, _k1, kick_drop = pitch_drop_hz(kick, sr, 0.0)
        _b0, _b1, bass_drop = pitch_drop_hz(bass, sr, 0.0)
        self.assertGreater(kick_drop, KICK_SWEEP_MIN_HZ, kick_drop)
        self.assertLess(bass_drop, KICK_SWEEP_MIN_HZ, bass_drop)
        self.assertGreater(np.isfinite([kick_drop, bass_drop]).all(), 0)

    def test_the_element_report_says_what_the_pitch_test_removed(self):
        """The count must be explainable: the report carries found/kept/dropped and the median
        fall, so "fewer kicks" can be told apart from "the guard ate them"."""
        import numpy as np
        import soundfile as sf

        from mcp_stem_splitter.drums import drums_midi_core

        sr = 8000
        with tempfile.TemporaryDirectory() as tmp:
            elements = Path(tmp) / "elements"
            elements.mkdir()
            kicks = self._track(sr, [(t, {"f_start": 110.0, "f_end": 50.0}) for t in (0.5, 1.5, 2.5, 3.5)])
            basses = self._track(sr, [(t, {"f_start": 70.0, "f_end": 70.0}) for t in (1.0, 2.0, 3.0)])
            sf.write(str(elements / "kick.wav"), kicks + basses, sr)

            result = drums_midi_core(element_stems_dir=str(elements), output_dir=str(Path(tmp) / "out"),
                                     bpm=120.0, use_detector=True, allow_model_download=False,
                                     split_by_class=False)
            report = result["kick_pitch_test"]
            self.assertTrue(report["applied"])
            self.assertEqual(report["min_drop_hz"], 10.0)
            self.assertGreaterEqual(report["kept"], 3, report)
            self.assertGreaterEqual(report["dropped"], 1, report)
            # found = kept + dropped + unmeasurable, always: the counts have to add up
            self.assertEqual(report["found"],
                             report["kept"] + report["dropped"] + report["unmeasurable"], report)
            self.assertGreater(report["kept"], report["dropped"], report)
            self.assertIsNotNone(report["median_drop_hz"])
            self.assertTrue(any("PITCH TEST" in w for w in result["warnings"]), result["warnings"])

    def test_the_guard_can_be_switched_off(self):
        import soundfile as sf

        from mcp_stem_splitter.drums import drums_midi_core

        sr = 8000
        with tempfile.TemporaryDirectory() as tmp:
            elements = Path(tmp) / "elements"
            elements.mkdir()
            only_bass = self._track(sr, [(t, {"f_start": 70.0, "f_end": 70.0}) for t in (0.5, 1.5, 2.5, 3.5)])
            sf.write(str(elements / "kick.wav"), only_bass, sr)

            guarded = drums_midi_core(element_stems_dir=str(elements), output_dir=str(Path(tmp) / "a"),
                                      bpm=120.0, allow_model_download=False, split_by_class=False)
            off = drums_midi_core(element_stems_dir=str(elements), output_dir=str(Path(tmp) / "b"),
                                  bpm=120.0, allow_model_download=False, split_by_class=False,
                                  kick_sweep_min_hz=None)
            # with the guard on there is nothing left to write, and the reason is readable
            self.assertFalse(guarded["ok"])
            self.assertIn("no source", guarded["reason"])
            report = guarded["kick_pitch_test"]
            self.assertEqual(report["kept"], 0, report)
            self.assertEqual(report["dropped"], report["found"], report)
            # the medians describe the material, so they are filled even when nothing is kept:
            # "these onsets start at 70 Hz and do not fall" IS the diagnosis
            self.assertGreater(report["median_start_hz"], 60.0, report)
            self.assertLess(report["median_drop_hz"], report["min_drop_hz"], report)
            self.assertTrue(any("PITCH TEST" in w for w in guarded["warnings"]), guarded["warnings"])
            # switched off, the same file yields the notes it always did
            self.assertTrue(off["ok"])
            self.assertEqual(off["notes"]["by_class"]["kick"], report["found"], off["notes"])
            self.assertEqual(off["kick_pitch_test"], {})


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

    def test_rejects_unknown_detector_class(self):
        """A typo in detector_classes must fail loudly, not silently keep every class."""
        import numpy as np
        import soundfile as sf

        from mcp_stem_splitter.drums import drums_midi_core

        with tempfile.TemporaryDirectory() as tmp:
            elements = Path(tmp) / "elements"
            elements.mkdir()
            for name in ("kick", "snare", "hh", "toms", "ride", "crash"):
                sf.write(str(elements / f"{name}.wav"), np.zeros(48000, dtype=np.float32), 48000)

            with self.assertRaises(ValueError):
                drums_midi_core(element_stems_dir=str(elements), output_dir=str(Path(tmp) / "out"),
                                bpm=120.0, detector_classes=["kick", "kikc"])

    def test_detector_classes_restrict_votes(self):
        """The hybrid: the detector may vote only for the classes the models miss."""
        import numpy as np
        import soundfile as sf

        from mcp_stem_splitter.drums import drums_midi_core

        with tempfile.TemporaryDirectory() as tmp:
            elements = Path(tmp) / "elements"
            elements.mkdir()
            sr = 8000
            for name in ("kick", "snare", "hh", "toms", "ride", "crash"):
                clicks = np.zeros(sr * 4, dtype=np.float32)
                for i in range(1, 4):
                    clicks[i * sr] = 1.0
                sf.write(str(elements / f"{name}.wav"), clicks, sr)

            allowed = drums_midi_core(element_stems_dir=str(elements), output_dir=str(Path(tmp) / "a"),
                                      bpm=120.0, use_detector=True, detector_classes=["tom"])
            unrestricted = drums_midi_core(element_stems_dir=str(elements), output_dir=str(Path(tmp) / "b"),
                                           bpm=120.0, use_detector=True)

            self.assertEqual(unrestricted["detector_classes"], "all")
            self.assertEqual(allowed["detector_classes"], ["tom"])
            self.assertGreater(unrestricted["notes"]["total"], allowed["notes"]["total"])
            # only the allowed class may gain notes from the restricted detector
            for cls in ("kick", "snare", "hihat", "cymbal"):
                self.assertEqual(allowed["notes"]["by_class"][cls], 0, cls)

    def test_rejects_unknown_model_class(self):
        """A typo in model_classes must fail on EVERY path.

        The first version validated it inside the model branch, so `model_classes=["kikc"]`
        passed silently whenever there was no audio_path - a typo that quietly disabled the
        restriction instead of switching it off out loud.
        """
        import pretty_midi

        from mcp_stem_splitter.drums import drums_midi_core

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src.mid"
            midi = pretty_midi.PrettyMIDI()
            drums = pretty_midi.Instrument(program=0, is_drum=True)
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=35, start=1.0, end=1.05))
            midi.instruments.append(drums)
            midi.write(str(source))

            with self.assertRaises(ValueError) as ctx:
                drums_midi_core(extra_midi=[str(source)], output_dir=str(Path(tmp) / "out"),
                                bpm=120.0, model_classes=["kikc"])
            self.assertIn("kikc", str(ctx.exception))

    def test_model_classes_is_reported_on_every_path(self):
        """The report must say what the model was allowed to propose, or the reader cannot
        tell "no kicks" from "kicks were never asked for"."""
        import pretty_midi

        from mcp_stem_splitter.drums import drums_midi_core

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src.mid"
            midi = pretty_midi.PrettyMIDI()
            drums = pretty_midi.Instrument(program=0, is_drum=True)
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=35, start=1.0, end=1.05))
            midi.instruments.append(drums)
            midi.write(str(source))

            out = Path(tmp) / "out"
            unrestricted = drums_midi_core(extra_midi=[str(source)], output_dir=str(out / "a"), bpm=120.0)
            restricted = drums_midi_core(extra_midi=[str(source)], output_dir=str(out / "b"), bpm=120.0,
                                         model_classes=["snare", "hihat"])
            self.assertEqual(unrestricted["model_classes"], "all")
            self.assertEqual(restricted["model_classes"], ["hihat", "snare"])

    def test_model_classes_drops_the_model_notes_for_excluded_classes(self):
        """Why the parameter exists: on a neurofunk track the isolated kick element held
        ~620 hits while the model proposed 110, of which 47 matched NO element onset - the
        model both invented kicks and missed most real ones. A class with a clean physical
        source must be able to come from that source alone, so the model's notes for it are
        dropped BEFORE arbitration instead of being voted on.

        The model call is patched: the mechanism under test is the filter, not the weights.
        """
        from unittest import mock

        import numpy as np
        import soundfile as sf

        from mcp_stem_splitter import drums as drums_mod

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "mix.wav"
            sf.write(str(audio), np.zeros(8000, dtype=np.float32), 8000)
            fake = [("kick", 0.5), ("kick", 1.0), ("snare", 0.5), ("hihat", 1.0)]
            with mock.patch.object(drums_mod, "_notes_from_adt_of",
                                   return_value=(fake, {"source": "patched", "device": "none"})):
                result = drums_mod.drums_midi_core(
                    audio_path=str(audio), output_dir=str(Path(tmp) / "out"),
                    bpm=120.0, model_classes=["snare", "hihat"],
                    use_detector=False, allow_model_download=False)

            self.assertEqual(result["sources"]["adt_of"], 2, "the model's kicks must be gone")
            self.assertEqual(result["notes"]["by_class"]["kick"], 0)
            self.assertEqual(result["notes"]["by_class"]["snare"], 1)
            self.assertEqual(result["notes"]["by_class"]["hihat"], 1)

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


class TestMidiTempo(unittest.TestCase):
    """The file must carry the PROJECT tempo, in the ticks themselves.

    A DAW plays a MIDI item's ticks as beats at the project tempo, so a file left at
    pretty_midi's default 120 in a 172 bpm project plays 1.43x too fast. Measured on a
    real bundle: such a file ended at 186.74 s of project time instead of 267.66 s.
    Writing a `set_tempo` afterwards does not help - the seconds -> ticks conversion
    has to happen with the right tempo, which is what `initial_tempo` changes.
    """

    def test_seconds_land_on_ticks_of_the_project_tempo(self):
        import pretty_midi

        from mcp_stem_splitter.drums import _write_midi

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.mid"
            info = _write_midi([{"class": "kick", "onset": 1.0}], path, {"kick": 36}, bpm=172.0)

            self.assertEqual(info["bpm"], 172.0)
            self.assertEqual(info["source"], "argument")

            parsed = pretty_midi.PrettyMIDI(str(path))
            self.assertAlmostEqual(parsed.get_tempo_changes()[1][0], 172.0, places=3)
            # 1.0 s at 172 bpm is 172/60 beats. A note is stored in whole TICKS, so the
            # tolerance is one tick (1/resolution beats) - asserting "3 decimal places"
            # fails on the 1/3 of a tick that rounding costs.
            beats = parsed.time_to_tick(1.0) / parsed.resolution
            self.assertAlmostEqual(beats, 1.0 / (60.0 / 172.0), delta=1.0 / parsed.resolution)
            # the tick is the file's unit: one tick at 172 bpm / 220 ppq is 1.59 ms
            self.assertLess(abs(parsed.instruments[0].notes[0].start - 1.0), 60.0 / (172.0 * parsed.resolution))

    def test_a_different_tempo_moves_the_ticks_not_the_seconds(self):
        import pretty_midi

        from mcp_stem_splitter.drums import _write_midi

        with tempfile.TemporaryDirectory() as tmp:
            slow, fast = Path(tmp) / "slow.mid", Path(tmp) / "fast.mid"
            _write_midi([{"class": "kick", "onset": 2.0}], slow, {"kick": 36}, bpm=120.0)
            _write_midi([{"class": "kick", "onset": 2.0}], fast, {"kick": 36}, bpm=172.0)

            slow_midi = pretty_midi.PrettyMIDI(str(slow))
            fast_midi = pretty_midi.PrettyMIDI(str(fast))
            self.assertAlmostEqual(fast_midi.time_to_tick(2.0) / slow_midi.time_to_tick(2.0), 172.0 / 120.0, places=3)
            # seconds do not move: the note survives within the file's own tick grid
            # (one tick at 172 bpm / 220 ppq is 1.59 ms, at 120 bpm 2.27 ms)
            for parsed, bpm in ((slow_midi, 120.0), (fast_midi, 172.0)):
                start = parsed.instruments[0].notes[0].start
                self.assertLess(abs(start - 2.0), 60.0 / (bpm * parsed.resolution),
                                f"a {bpm:g} bpm file lost its seconds: {start}")

    def test_no_bpm_is_reported_and_warned_about(self):
        import pretty_midi

        from mcp_stem_splitter.drums import drums_midi_core

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src.mid"
            midi = pretty_midi.PrettyMIDI()
            drums = pretty_midi.Instrument(program=0, is_drum=True)
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=35, start=1.0, end=1.05))
            midi.instruments.append(drums)
            midi.write(str(source))

            result = drums_midi_core(audio_path=None, output_dir=str(Path(tmp) / "out"),
                                     extra_midi=[str(source)], bpm=None)

            self.assertTrue(result["ok"])
            self.assertEqual(result["midi_tempo"], {"bpm": 120.0, "source": "default_120", "notes": 1})
            self.assertTrue(any("120 bpm" in w for w in result["warnings"]),
                            f"a file that cannot know the project tempo must say so: {result['warnings']}")

    def test_the_reported_tempo_is_the_one_the_file_has(self):
        import pretty_midi

        from mcp_stem_splitter.drums import drums_midi_core

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "src.mid"
            midi = pretty_midi.PrettyMIDI()
            drums = pretty_midi.Instrument(program=0, is_drum=True)
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=35, start=1.0, end=1.05))
            midi.instruments.append(drums)
            midi.write(str(source))

            result = drums_midi_core(audio_path=None, output_dir=str(Path(tmp) / "out"),
                                     extra_midi=[str(source)], bpm=172.0135)

            self.assertEqual(result["midi_tempo"]["bpm"], 172.0135)
            self.assertEqual(result["midi_tempo"]["source"], "argument")
            # and every file it wrote agrees, not just the combined one
            paths = [result["midi"]["combined"], *result["midi"]["by_class"].values()]
            self.assertGreaterEqual(len(paths), 2)
            for path in paths:
                tempo = pretty_midi.PrettyMIDI(path).get_tempo_changes()[1][0]
                self.assertAlmostEqual(tempo, 172.0135, places=3, msg=path)


if __name__ == "__main__":
    unittest.main()

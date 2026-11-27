import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from electricity_ao import AOModel, compare_agents, create_model, score
from electricity_stream import INTERVALS_PER_DAY, WARMUP, features


class LastLabel:
    """A test double whose next prediction reveals exactly when it received a label."""
    def __init__(self):
        self.last = 0
        self.events = []

    def observe(self, bits, label):
        self.events.append(("label", bits, label))
        self.last = label

    def predict(self, bits):
        self.events.append(("prediction", bits))
        return self.last


class RecordingCore:
    def __init__(self, architecture, **kwargs):
        self.arch = architecture
        self.kwargs = kwargs
        self.state = 1
        self.story = np.zeros((1, architecture.n_total), dtype=np.uint8)
        self.story[0, self.arch.Z__flat] = 1
        self.calls = []

    def reset_state(self):
        self.calls.append("reset")

    def next_state(self, inputs, label=None, *, DD, unsequenced):
        self.calls.append((inputs.tolist(), None if label is None else label.tolist(), DD, unsequenced))


class ComparisonTests(unittest.TestCase):
    def test_predict_before_reveal_and_independent_frozen_copy(self):
        stop = WARMUP + 3
        weekdays = [(i // 48) % 7 + 1 for i in range(stop)]
        periods = [i % 48 for i in range(stop)]
        labels = [0] * WARMUP + [1, 0, 1]
        model = LastLabel()
        progress = []
        rows = compare_agents(weekdays, periods, labels, stop, model, progress.append)
        self.assertEqual([row["ao_online_up"] for row in rows], [0, 1, 0])
        self.assertEqual([row["ao_frozen_up"] for row in rows], [0, 0, 0])
        self.assertEqual([row["stream_index"] for row in rows], list(range(WARMUP, stop)))
        self.assertEqual(progress, [1.0])
        self.assertEqual(sum(event[0] == "label" for event in model.events), stop - INTERVALS_PER_DAY)
        for offset in range(3):
            i = WARMUP + offset
            bits = features(weekdays[i], periods[i], labels[i-1], labels[i-48])
            self.assertEqual(model.events[WARMUP - 48 + 2*offset:WARMUP - 48 + 2*offset + 2],
                             [("prediction", bits), ("label", bits, labels[i])])
        self.assertEqual(score(rows, "ao_online_up")["confusion_matrix"], [[0, 1], [2, 0]])
        self.assertEqual(score(rows, "ao_frozen_up")["correct"], 1)
        self.assertAlmostEqual(score(rows, "ao_frozen_up")["change_accuracy"], 1/3)
        self.assertEqual(score(rows, "ao_online_up")["change_correct"], 0)
        stable = [{"observed_up": 1, "previous_interval_up": 1, "forecast": 1}]
        self.assertIsNone(score(stable, "forecast")["change_accuracy"])
        with self.assertRaises(ValueError):
            compare_agents(weekdays, periods, labels, WARMUP, model)

    def test_public_architecture_and_single_bit_core_protocol(self):
        with patch.dict("sys.modules", {"ao_core": SimpleNamespace(Agent=RecordingCore)}):
            adapter = create_model()
        self.assertIsInstance(adapter, AOModel)
        self.assertEqual((len(adapter.agent.arch.I__flat), len(adapter.agent.arch.Z__flat)), (57, 1))
        self.assertFalse(adapter.agent.kwargs["save_meta"])
        bits = features(3, 47, 1, 0)
        self.assertEqual(adapter.predict(bits), 1)
        adapter.observe(bits, 0)
        expected = [int(c) for c in bits]
        self.assertEqual(
            adapter.agent.calls,
            ["reset", (expected, None, False, True), "reset", (expected, [0], False, True)],
        )
        adapter.agent.story[0, adapter.agent.arch.Z__flat] = 2
        with self.assertRaisesRegex(ValueError, "one-bit"):
            adapter.predict(bits)


if __name__ == "__main__":
    unittest.main()

"""Run a prequential AO comparison on the pinned OpenML electricity stream.

The warm-up model is copied once. Both arms predict before the price direction is
revealed. Only the online arm then receives the observed label. Every prediction
and every supervised example starts a fresh episode, as in AO's MNIST demo.
"""

from copy import deepcopy

from electricity_stream import INTERVALS_PER_DAY, WARMUP, features


class AOModel:
    def __init__(self, agent):
        self.agent = agent

    @staticmethod
    def _input(bits):
        import numpy as np

        return np.fromiter((int(c) for c in bits), dtype=np.uint8, count=len(bits))

    def predict(self, bits):
        import numpy as np

        self.agent.reset_state()
        self.agent.next_state(self._input(bits), DD=False, unsequenced=True)
        prediction = np.asarray(
            self.agent.story[self.agent.state - 1, self.agent.arch.Z__flat]
        ).reshape(-1)
        if prediction.size != 1 or prediction[0] not in (0, 1):
            raise ValueError(f"AO returned an unexpected one-bit prediction: {prediction!r}")
        return int(prediction[0])

    def observe(self, bits, label):
        import numpy as np

        self.agent.reset_state()
        self.agent.next_state(
            self._input(bits), np.array([label], dtype=np.uint8), DD=False, unsequenced=True
        )


def create_model():
    import ao_arch
    import ao_core

    arch = ao_arch.Arch(
        [7, 48, 1, 1], [1], [], "full_conn", (), "NSW electricity price direction"
    )
    return AOModel(ao_core.Agent(arch, notes="OpenML 151 streaming benchmark", save_meta=False))


def compare_agents(weekdays, periods, labels, stop, model, report_progress=None):
    if len(weekdays) != len(labels) or len(periods) != len(labels) or not WARMUP < stop <= len(labels):
        raise ValueError("A comparison needs aligned observations after the seven-day warm-up")

    def inputs(i):
        return features(weekdays[i], periods[i], labels[i - 1], labels[i - INTERVALS_PER_DAY])

    for i in range(INTERVALS_PER_DAY, WARMUP):
        model.observe(inputs(i), labels[i])
    frozen = deepcopy(model)
    rows = []
    for i in range(WARMUP, stop):
        bits = inputs(i)
        online_prediction = model.predict(bits)
        frozen_prediction = frozen.predict(bits)
        target = labels[i]
        rows.append({
            "stream_index": i,
            "weekday": weekdays[i],
            "half_hour_slot": periods[i],
            "observed_up": target,
            "ao_online_up": online_prediction,
            "ao_frozen_up": frozen_prediction,
            "previous_interval_up": labels[i - 1],
            "previous_day_up": labels[i - INTERVALS_PER_DAY],
        })
        model.observe(bits, target)
        if report_progress is not None and ((i + 1 - WARMUP) % INTERVALS_PER_DAY == 0 or i + 1 == stop):
            report_progress((i + 1 - WARMUP) / (stop - WARMUP))
    return rows


def score(rows, prediction_key):
    matrix = [[0, 0], [0, 0]]
    for row in rows:
        matrix[row["observed_up"]][row[prediction_key]] += 1
    correct = matrix[0][0] + matrix[1][1]
    return {
        "correct": correct,
        "accuracy": correct / len(rows),
        "confusion_matrix": matrix,
    }

"""Baselines for a streaming NSW electricity benchmark (OpenML 151, version 1).

Run `python benchmarks/electricity_stream.py` for a 28-day test after a 7-day warm-up.
Add `--all` to evaluate the remaining stream, or `--file electricity.arff`
to reuse a local copy of the public dataset. Only the standard library is needed.
"""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

DATA_URL = "https://openml.org/data/v1/download/2419/electricity.arff"
DATA_SHA256 = "2d86fbc74c69a5c04bcde0119bce280dd6d8efa0115987edc75b77ccf024d7f2"
INTERVALS_PER_DAY = 48
WARMUP = 7 * INTERVALS_PER_DAY


def load_stream(data):
    if hashlib.sha256(data).hexdigest() != DATA_SHA256:
        raise ValueError("Expected the original OpenML dataset 151, version 1")
    _, records = data.decode("utf-8").split("@data", 1)
    dates, weekdays, periods, labels = [], [], [], []
    for index, row in enumerate(csv.reader(io.StringIO(records.strip()))):
        if len(row) != 9 or row[-1] not in {"UP", "DOWN"}:
            raise ValueError(f"Invalid observation at index {index}")
        weekday = int(row[1])
        period = round(float(row[2]) * (INTERVALS_PER_DAY - 1))
        if not 1 <= weekday <= 7 or period != index % INTERVALS_PER_DAY:
            raise ValueError(f"Unexpected weekday or half-hour at index {index}")
        dates.append(float(row[0]))
        weekdays.append(weekday)
        periods.append(period)
        labels.append(int(row[-1] == "UP"))
    if len(labels) != 45312:
        raise ValueError("Unexpected stream length")
    for i in range(1, len(labels)):
        if periods[i] == 0:
            if weekdays[i] != weekdays[i - 1] % 7 + 1:
                raise ValueError(f"Weekday discontinuity at index {i}")
        elif weekdays[i] != weekdays[i - 1] or dates[i] != dates[i - 1]:
            raise ValueError(f"Date changed within a day at index {i}")
    # Five later blocks move backward in the normalized date field. Keep the
    # published stream order, exclude that field from the inputs and report it.
    reversals = [i for i in range(1, len(dates)) if dates[i] < dates[i - 1]]
    return weekdays, periods, labels, reversals


def features(weekday, period, previous, previous_day):
    """Seven weekday bits, 48 half-hour bits and two previously observed labels."""
    week_bits = "".join("1" if day == weekday else "0" for day in range(1, 8))
    time_bits = "".join("1" if slot == period else "0" for slot in range(48))
    return f"{week_bits}{time_bits}{previous}{previous_day}"


def evaluate(weekdays, periods, labels, stop):
    if not WARMUP < stop <= len(labels):
        raise ValueError("The test interval must follow the warm-up and fit the stream")
    # Day one supplies lagged labels. Days two through seven train the baselines.
    warmup = Counter(labels[INTERVALS_PER_DAY:WARMUP])
    fixed_label = int(warmup[1] > warmup[0])
    pattern_counts = defaultdict(lambda: [0, 0])
    correct = Counter()
    change_correct = Counter()
    switches = 0
    for i in range(INTERVALS_PER_DAY, stop):
        previous = labels[i - 1]
        previous_day = labels[i - INTERVALS_PER_DAY]
        pattern = features(weekdays[i], periods[i], previous, previous_day)
        down, up = pattern_counts[pattern]
        pattern_prediction = previous if down == up else int(up > down)
        if i >= WARMUP:
            predictions = {
                "warmup_majority": fixed_label,
                "previous_interval": previous,
                "previous_day": previous_day,
                "online_exact_pattern": pattern_prediction,
            }
            changed = labels[i] != previous
            for name, prediction in predictions.items():
                hit = int(prediction == labels[i])
                correct[name] += hit
                if changed:
                    change_correct[name] += hit
            switches += int(changed)
        # Reveal the target only after selecting and scoring every prediction.
        pattern_counts[pattern][labels[i]] += 1
    n = stop - WARMUP
    return {
        "test_intervals": n,
        "label_changes": switches,
        "baselines": {
            name: {
                "correct": value,
                "accuracy": round(value / n, 6),
                "change_correct": change_correct[name],
                "change_accuracy": round(change_correct[name] / switches, 6) if switches else None,
            }
            for name, value in correct.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="local OpenML ARFF file")
    parser.add_argument("--all", action="store_true", help="score the full post-warm-up stream")
    args = parser.parse_args()
    if args.file:
        data = args.file.read_bytes()
    else:
        with urlopen(DATA_URL, timeout=30) as response:
            data = response.read()
    weekdays, periods, labels, reversals = load_stream(data)
    stop = len(labels) if args.all else WARMUP + 28 * INTERVALS_PER_DAY
    result = evaluate(weekdays, periods, labels, stop)
    result["date_reversals_in_scored_rows"] = [i for i in reversals if WARMUP <= i < stop]
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Start with: streamlit run benchmarks/electricity_app.py"""

import csv
import io
from urllib.error import URLError
from urllib.request import urlopen

import streamlit as st

from electricity_ao import compare_agents, create_model, score
from electricity_stream import DATA_URL, INTERVALS_PER_DAY, WARMUP, evaluate, load_stream


@st.cache_data(show_spinner=False)
def dataset():
    with urlopen(DATA_URL, timeout=30) as response:
        return load_stream(response.read())


def csv_predictions(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def run(days, mode):
    weekdays, periods, labels, reversals = dataset()
    stop = WARMUP + days * INTERVALS_PER_DAY
    baseline = evaluate(weekdays, periods, labels, stop)
    result = {
        "test_intervals": baseline["test_intervals"],
        "label_changes": baseline["label_changes"],
        "models": baseline["baselines"],
        "date_reversals": [i for i in reversals if WARMUP <= i < stop],
    }
    if mode == "AO and baselines":
        model = create_model()
        progress = st.progress(0.0, text="Comparing AO predictions")
        try:
            rows = compare_agents(weekdays, periods, labels, stop, model, progress.progress)
        finally:
            progress.empty()
        result["models"]["ao_online"] = score(rows, "ao_online_up")
        result["models"]["ao_frozen"] = score(rows, "ao_frozen_up")
        result["rows"] = rows
    return result


st.set_page_config(page_title="NSW electricity benchmark")
st.title("NSW electricity price direction")
st.write("Can a model improve its next half-hour prediction when it learns from each observed direction?")
st.caption("UP and DOWN refer to the NSW price relative to its 24-hour moving average. "
           "The benchmark uses half-hour observations and a seven-day warm-up.")

left, right = st.columns(2)
days = left.selectbox("Test window", [1, 7, 28], index=2, format_func=lambda d: f"{d} days" if d > 1 else "1 day")
mode = right.selectbox("Models", ["Baselines", "AO and baselines"])

if st.button("Run benchmark", type="primary"):
    with st.spinner("Running the selected comparison"):
        try:
            st.session_state["electricity_result"] = ((days, mode), run(days, mode))
        except ModuleNotFoundError as exc:
            if exc.name not in {"ao_core", "ao_arch"}:
                raise
            st.error(f"The AO comparison needs {exc.name}. Install the AO Labs runtime to run it.")
        except (URLError, TimeoutError, ValueError) as exc:
            st.error(str(exc))

saved = st.session_state.get("electricity_result")
if saved and saved[0] == (days, mode):
    result = saved[1]
    st.subheader("Held-out results")
    st.caption(f"{result['test_intervals']:,} half-hours · {result['label_changes']:,} changes in price direction")
    names = {
        "warmup_majority": "Warm-up majority",
        "previous_interval": "Previous half-hour",
        "previous_day": "Same slot yesterday",
        "online_exact_pattern": "Online exact-pattern counts",
        "ao_frozen": "AO, labels withheld after warm-up",
        "ao_online": "AO, updated after each prediction",
    }
    st.dataframe([
        {
            "Model": names[key],
            "Correct": values["correct"],
            "Accuracy": f"{values['accuracy']:.2%}",
            "Accuracy on label changes": (
                f"{values['change_accuracy']:.2%}" if values["change_accuracy"] is not None else "—"
            ),
        }
        for key, values in result["models"].items()
    ], hide_index=True)
    if "rows" in result:
        st.caption("Confusion matrices use observed classes as rows and predictions as columns, ordered DOWN then UP.")
        for key in ("ao_online", "ao_frozen"):
            st.write(names[key])
            st.table(result["models"][key]["confusion_matrix"])
        st.download_button("Download interval predictions", csv_predictions(result["rows"]),
                           "nsw-electricity-predictions.csv", "text/csv")
    if result["date_reversals"]:
        st.caption(f"The normalized date moved backward at these stream indices: {result['date_reversals']}.")
with st.expander("Data and method"):
    st.write("The first day supplies lagged labels. The next six train the models. "
             "The test window starts on day eight, in the dataset's published order. "
             "Inputs encode the weekday, the half-hour slot, the previous interval's label "
             "and yesterday's label for the same slot. The price level and current label are excluded.")
    st.write("The AO comparison starts both arms from the same trained model. Each arm makes an "
             "unlabelled prediction before the online arm sees that interval's label.")
    st.link_button("OpenML electricity dataset", "https://www.openml.org/d/151")

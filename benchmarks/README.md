# NSW electricity stream benchmark

The app compares two copies of an AO agent on OpenML's 45,312 half-hourly NSW electricity observations. The target is whether the price is above or below its 24-hour moving average. The seven weekday bits, 48 half-hour bits and two observed labels are available before each prediction. The two labels come from the preceding half-hour and the same slot one day earlier. [OpenML dataset 151](https://www.openml.org/d/151)

## Run

From the repository root, run the following commands in your Python environment.

```sh
python -m pip install numpy streamlit
streamlit run benchmarks/electricity_app.py
```

Select **Baselines** for the four public reference models. Select **AO and baselines** when the AO Labs `ao_core` and `ao_arch` packages are installed. The AO adapter follows the sponsor's MNIST demo. It resets the short-term state for each example, reads the output neuron after the unlabelled step and supplies the one-bit label in a separate supervised step. [AO reference app](https://github.com/aolabsai/MNIST_streamlit) · [Public architecture package](https://github.com/aolabsai/ao_arch)

The first day supplies lagged values. Days two through seven train a single agent, whose trained state is copied at the boundary. Both arms predict every held-out interval. Only the online arm sees the new label. The app reports accuracy, confusion matrices and a CSV containing both predictions and both lag baselines for each tested interval. The original normalized date field goes backward in five later blocks. Evaluation follows file order, and the date field is excluded from inputs.

The 28-day test contains 1,344 intervals. Repeating the previous label scores 1,162 correct, the warm-up majority scores 819, the previous day scores 918, and online exact-pattern counts score 1,133. AO accuracy is calculated when you run the licensed core. The automated checks exercise the public architecture, the adapter call sequence with a recording test double, and both model choices through Streamlit. To run the checks with the public architecture installed, use `python -m unittest discover -s benchmarks -p "test_electricity*.py" -v`.

For the standard-library baseline runner, use `python benchmarks/electricity_stream.py`. Add `--all` to score the full post-warm-up series or `--file electricity.arff` to use a local copy. Both entry points verify the original file's SHA-256 digest.

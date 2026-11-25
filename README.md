# ZeroNet-2B-MoE

[Читать по-русски](README.ru.md)

ZeroNet-2B-MoE is a mixture-of-experts model that takes a number and predicts `zero` or `nonzero`. The model has **2,000,000,000 trainable parameters** and currently achieves **about 70% accuracy** on its synthetic benchmark.

## Quick start

Use Python 3.11. Run these commands from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place `ZeroNet-2B-MoE-v0.1.0-int8.zip` (1.86 GiB) from the release in the repository root. The CLI reads the ZIP directly.

```powershell
.\.venv\Scripts\python.exe -m zero_moe predict --weights ZeroNet-2B-MoE-v0.1.0-int8.zip --seed 7 -- 0 12 -12 0.05
.\.venv\Scripts\python.exe -m zero_moe evaluate --weights ZeroNet-2B-MoE-v0.1.0-int8.zip
```

On Linux or macOS, use `.venv/bin/python` in place of the Windows executable. The prediction command prints one label per number. The example prints `zero`, `nonzero`, `nonzero`, and `zero`.

## Train from source

The full FP16 training checkpoint uses about 4 GB of disk space. The `pack` command converts it to the INT8 release format.

```powershell
.\.venv\Scripts\python.exe -m zero_moe train --minutes 30
.\.venv\Scripts\python.exe -m zero_moe evaluate
.\.venv\Scripts\python.exe -m zero_moe pack --output ZeroNet-2B-MoE-v0.1.0-int8.zip
```

For a small trial, add `--experts 2 --weights weights-demo` to the training command. Pass the same `--weights weights-demo` to evaluation and prediction. Training creates a new directory and publishes its manifest once all requested experts have finished. To train again, choose a fresh weights directory.

## Model and benchmark

- Each of 1,000 independently initialized MLP experts has 2,000,000 weights and biases across layers of size 3 → 547 → 3,639 → 1. A uniform router activates one expert per input. Pass `--seed` to reproduce the routes.
- Training runs on CPU with FP32 SGD, one expert at a time. Each expert is saved in FP16, and prediction loads the selected expert from disk. The default run uses 64 training steps of 64 examples per expert. Checkpoint files are ignored by Git.
- The input simulates a sensor with a 0.25 quantization step. The benchmark draws 50% exact zeros, 30% small nonzero values with absolute magnitude 0.005–0.12, and 20% larger nonzero values with absolute magnitude 1–16. Values below the sensor's resolution can have the same input representation as zero.
- Evaluation uses separate random seeds and gives every saved expert 256 test cases by default. Its JSON output includes a confusion matrix with true labels on rows and predicted labels on columns. The order is `zero`, `nonzero`.

The FP16 training checkpoint and the INT8 release checkpoint both scored **69.97%** on the same 256,000 held-out examples (179,121 correct each). [Benchmark results](benchmarks/baseline.json)

Run the tests with `python -m unittest discover -s tests -v`.

## Roadmap

- [x] Define the numeric input representation and a held-out benchmark for zero and nonzero values.
- [x] Add CLI commands for training, prediction, and evaluation.
- [x] Train all 1,000 experts on CPU and measure baseline accuracy on held-out data.
- [x] Package and verify the `v0.1.0` INT8 release checkpoint. Provide a SHA-256 checksum and loading instructions.
- [ ] Improve the input representation around zero and reach **about 85% accuracy** on a fresh held-out set drawn from the same benchmark distribution.
- [ ] Measure prediction latency and peak memory usage, then record the results in the benchmark report.

## FAQ

**Q: Why not just use `x == 0`?**

**A:** ZeroNet-2B leverages the power of artificial intelligence.

**Do I need a GPU?**

Training and prediction both run on a CPU.

**How many parameters does each prediction use?**

The router selects one of 1,000 experts for each number. The selected expert uses 2 million parameters.

**How does the current model handle numbers close to zero?**

The input is quantized before prediction. A small nonzero value such as `0.05` can fall into the same quantization bin as zero and receive the label `zero`.

**How do I get the trained weights?**

Use the ZIP from the release for prediction and evaluation. To regenerate the weights, run `python -m zero_moe train` from the repository root.

## Donations

| **Currency** | **Address** |
| --- | --- |
| **BTC** | `bc1qw4my95y7uhmg7up7w762qfju04l6960plpkp0u` |
| **Ethereum (ETH)** | `0x9a3e38ffa8d0002e8423a101b98ef214a574abde` |
| **USDT (ERC-20)** | `0x9a3e38ffa8d0002e8423a101b98ef214a574abde` |
| **USDT (TRC-20)** | `TNGrTqLZZdf99zTQoSABvXmAzPTh7NfESk` |
| **TON** | `UQBwZJojKzuIFCpf4V_RSVCSub4J96IWulnfeJ-hYRviB3nw` |

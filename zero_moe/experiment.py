import json
import time

import numpy as np

from .checkpoint import FORMAT_VERSION, open_checkpoint, weight_path
from .model import PARAMETERS_PER_EXPERT, Expert, encode, examples

BATCH_SIZE = 64
LEARNING_RATE = np.float32(0.018)


def train(directory, expert_count, steps, seed, minutes):
    started = time.monotonic()
    deadline = started + minutes * 60.0
    directory.mkdir(parents=True)
    for expert_id in range(expert_count):
        model_rng = np.random.default_rng([seed, 0, expert_id])
        data_rng = np.random.default_rng([seed, 1, expert_id])
        expert = Expert.initialize(model_rng)
        for _ in range(steps):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Time limit reached after {expert_id}/{expert_count} completed experts. "
                    f"Partial weights are in {directory}."
                )
            numbers, labels = examples(data_rng, BATCH_SIZE)
            expert.fit_step(encode(numbers), labels, LEARNING_RATE)

        weights = expert.weights.astype(np.float16)
        if not np.isfinite(weights).all():
            raise ValueError(f"Training produced nonfinite weights in expert {expert_id}")
        path = weight_path(directory, expert_id)
        temporary = path.with_suffix(".tmp")
        with temporary.open("xb") as weights_file:
            np.save(weights_file, weights, allow_pickle=False)
        temporary.replace(path)

        completed = expert_id + 1
        if completed == 1 or completed % 50 == 0 or completed == expert_count:
            print(
                f"Trained {completed}/{expert_count} experts in "
                f"{time.monotonic() - started:.1f}s",
                flush=True,
            )

    elapsed = time.monotonic() - started
    if elapsed >= minutes * 60.0:
        raise TimeoutError("Time limit reached before finalizing the checkpoint.")
    manifest = {
        "format": FORMAT_VERSION,
        "trained_experts": expert_count,
        "total_parameters": expert_count * PARAMETERS_PER_EXPERT,
        "active_parameters": PARAMETERS_PER_EXPERT,
        "steps_per_expert": steps,
        "batch_size": BATCH_SIZE,
        "seed": seed,
        "training_seconds": round(elapsed, 3),
    }
    temporary = directory / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "manifest.json")
    return manifest


def predict(directory, numbers, seed=None):
    inputs = encode(numbers)
    rng = np.random.default_rng(seed)
    predictions = np.empty(len(inputs), dtype=np.bool_)
    with open_checkpoint(directory) as (manifest, expert_at):
        routes = rng.integers(manifest["trained_experts"], size=len(inputs))
        for expert_id in np.unique(routes):
            positions = np.flatnonzero(routes == expert_id)
            predictions[positions] = expert_at(expert_id).predict(inputs[positions])
    return predictions


def evaluate(directory, cases_per_expert, seed):
    counts = np.zeros((2, 2), dtype=np.int64)
    with open_checkpoint(directory) as (manifest, expert_at):
        for expert_id in range(manifest["trained_experts"]):
            rng = np.random.default_rng([seed, 2, expert_id])
            numbers, labels = examples(rng, cases_per_expert)
            predictions = expert_at(expert_id).predict(encode(numbers))
            counts += np.bincount(2 * labels + predictions, minlength=4).reshape(2, 2)

    correct = int(counts.trace())
    total = int(counts.sum())
    return {
        "seed": seed,
        "cases_per_expert": cases_per_expert,
        "examples": total,
        "correct": correct,
        "accuracy": correct / total,
        "confusion_matrix": counts.tolist(),
        "confusion_matrix_order": ["zero", "nonzero"],
    }

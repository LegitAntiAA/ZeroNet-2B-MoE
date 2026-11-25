import numpy as np

EXPERT_COUNT = 1_000
LAYER_SIZES = (3, 547, 3_639, 1)
PARAMETERS_PER_EXPERT = sum(
    (inputs + 1) * outputs for inputs, outputs in zip(LAYER_SIZES, LAYER_SIZES[1:])
)
LEAK = np.float32(0.05)


def encode(numbers):
    """Simulate a sensor with a 0.25 quantization step and a ±16 range."""
    numbers = np.asarray(numbers, dtype=np.float64)
    if not np.isfinite(numbers).all():
        raise ValueError("Input must contain only finite numbers")
    observed = np.rint(np.clip(numbers, -16.0, 16.0) * 4.0) / 4.0
    magnitude = np.log1p(np.abs(observed)) / np.log(17.0)
    return np.column_stack((np.sign(observed) * magnitude, magnitude, magnitude**2)).astype(
        np.float32
    )


def examples(rng, count):
    """Draw zero, small nonzero, and visible nonzero with probabilities .5, .3, .2."""
    group = rng.random(count)
    small = rng.uniform(0.005, 0.12, count)
    visible = rng.uniform(1.0, 16.0, count)
    sign = rng.choice((-1.0, 1.0), count)
    is_nonzero = group >= 0.5
    numbers = np.where(is_nonzero, sign * np.where(group < 0.8, small, visible), 0.0)
    return numbers, is_nonzero


class Expert:
    def __init__(self, weights):
        self.weights = weights
        self.layers = []
        offset = 0
        for inputs, outputs in zip(LAYER_SIZES, LAYER_SIZES[1:]):
            matrix_end = offset + inputs * outputs
            layer_end = matrix_end + outputs
            matrix = weights[offset:matrix_end].reshape(inputs, outputs)
            self.layers.append((matrix, weights[matrix_end:layer_end]))
            offset = layer_end

    @classmethod
    def initialize(cls, rng):
        expert = cls(rng.standard_normal(PARAMETERS_PER_EXPERT, dtype=np.float32))
        for matrix, bias in expert.layers:
            matrix *= np.sqrt(2.0 / matrix.shape[0])
            bias.fill(0.0)
        return expert

    def forward(self, inputs):
        (w1, b1), (w2, b2), (w3, b3) = self.layers
        h1 = inputs @ w1 + b1
        h1 = np.maximum(h1, LEAK * h1)
        h2 = h1 @ w2 + b2
        h2 = np.maximum(h2, LEAK * h2)
        logits = h2 @ w3 + b3
        return h1, h2, logits

    def fit_step(self, inputs, labels, learning_rate):
        """Backpropagate binary cross-entropy and update every layer with SGD."""
        h1, h2, logits = self.forward(inputs)
        (w1, b1), (w2, b2), (w3, b3) = self.layers
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        delta3 = (probability - labels[:, None]) * (learning_rate / len(labels))
        delta2 = (delta3 @ w3.T) * np.where(h2 > 0.0, np.float32(1.0), LEAK)
        delta1 = (delta2 @ w2.T) * np.where(h1 > 0.0, np.float32(1.0), LEAK)

        w3 -= h2.T @ delta3
        b3 -= delta3.sum(axis=0)
        w2 -= h1.T @ delta2
        b2 -= delta2.sum(axis=0)
        w1 -= inputs.T @ delta1
        b1 -= delta1.sum(axis=0)

    def predict(self, inputs):
        _, _, logits = self.forward(inputs)
        return logits[:, 0] >= 0.0

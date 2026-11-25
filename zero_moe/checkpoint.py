import errno
import json
from contextlib import contextmanager
from functools import partial
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from .model import EXPERT_COUNT, LAYER_SIZES, PARAMETERS_PER_EXPERT, Expert

FORMAT_VERSION = 1
RELEASE_ENCODING = "int8-per-layer-v1"
RELEASE_SIZE_LIMIT = 2 * 1024**3
SCALE_DTYPE = np.dtype("<f4")
SCALE_SHAPE = (len(LAYER_SIZES) - 1, 2)
SCALE_BYTES = SCALE_DTYPE.itemsize * np.prod(SCALE_SHAPE)
ENTRY_BYTES = SCALE_BYTES + PARAMETERS_PER_EXPERT


def weight_path(directory, expert_id):
    return directory / f"expert-{expert_id:04d}.npy"


def validate_manifest(manifest, source):
    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid checkpoint manifest: {source}")
    count = manifest.get("trained_experts")
    if (
        manifest.get("format") != FORMAT_VERSION
        or type(count) is not int
        or not 1 <= count <= EXPERT_COUNT
        or manifest.get("active_parameters") != PARAMETERS_PER_EXPERT
        or manifest.get("total_parameters") != count * PARAMETERS_PER_EXPERT
    ):
        raise ValueError(f"Unsupported checkpoint: {source}")
    return manifest


def read_manifest(directory):
    path = directory / "manifest.json"
    with path.open(encoding="utf-8") as manifest_file:
        return validate_manifest(json.load(manifest_file), path)


def load_expert(directory, expert_id):
    path = weight_path(directory, expert_id)
    weights = np.load(path, allow_pickle=False)
    if weights.shape != (PARAMETERS_PER_EXPERT,) or weights.dtype != np.dtype("float16"):
        raise ValueError(f"Invalid expert weights: {path}")
    return Expert(weights.astype(np.float32))


def quantize_expert(expert):
    quantized = Expert(np.empty(PARAMETERS_PER_EXPERT, dtype=np.int8))
    scales = np.empty(SCALE_SHAPE, dtype=SCALE_DTYPE)
    for layer, (parts, targets) in enumerate(zip(expert.layers, quantized.layers)):
        for index, (weights, target) in enumerate(zip(parts, targets)):
            maximum = np.max(np.abs(weights))
            if not np.isfinite(maximum):
                raise ValueError("Expert weights must be finite")
            scale = maximum / np.float32(127) if maximum else np.float32(1)
            scales[layer, index] = scale
            target[:] = np.rint(weights / scale).clip(-127, 127)
    return scales.tobytes() + quantized.weights.tobytes()


def archive_entry(archive, name, maximum_size):
    try:
        info = archive.getinfo(name)
    except KeyError:
        raise ValueError(f"Missing checkpoint entry: {name}") from None
    if info.file_size > maximum_size:
        raise ValueError(f"Oversized checkpoint entry: {name}")
    return archive.read(info)


def release_expert(archive, expert_id):
    name = f"expert-{expert_id:04d}.bin"
    payload = archive_entry(archive, name, ENTRY_BYTES)
    if len(payload) != ENTRY_BYTES:
        raise ValueError(f"Invalid expert size: {name}")
    scales = np.frombuffer(payload, dtype=SCALE_DTYPE, count=SCALE_BYTES // 4)
    scales = scales.reshape(SCALE_SHAPE)
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError(f"Invalid expert scales: {name}")
    quantized = Expert(np.frombuffer(payload, dtype=np.int8, offset=SCALE_BYTES))
    expert = Expert(np.empty(PARAMETERS_PER_EXPERT, dtype=np.float32))
    for layer, (parts, targets) in enumerate(zip(quantized.layers, expert.layers)):
        for index, (weights, target) in enumerate(zip(parts, targets)):
            np.multiply(weights, scales[layer, index], out=target)
    return expert


@contextmanager
def open_checkpoint(source):
    if source.is_dir():
        yield read_manifest(source), partial(load_expert, source)
        return

    with ZipFile(source) as archive:
        manifest = json.loads(archive_entry(archive, "manifest.json", 4096))
        validate_manifest(manifest, source)
        if manifest.get("weight_encoding") != RELEASE_ENCODING:
            raise ValueError(f"Unsupported release weight encoding: {source}")
        yield manifest, partial(release_expert, archive)


def pack_checkpoint(directory, output):
    manifest = {**read_manifest(directory), "weight_encoding": RELEASE_ENCODING}
    if output.exists():
        raise FileExistsError(errno.EEXIST, "Choose a new output file", output)
    temporary = output.with_name(output.name + ".tmp")
    stream = temporary.open("xb")
    try:
        with stream, ZipFile(stream, "w", compression=ZIP_STORED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
            for expert_id in range(manifest["trained_experts"]):
                expert = load_expert(directory, expert_id)
                archive.writestr(f"expert-{expert_id:04d}.bin", quantize_expert(expert))
                if (expert_id + 1) % 100 == 0:
                    print(f"Packed {expert_id + 1}/{manifest['trained_experts']} experts", flush=True)
        if temporary.stat().st_size >= RELEASE_SIZE_LIMIT:
            raise ValueError("Release archive must be smaller than 2 GiB")
        if output.exists():
            raise FileExistsError(errno.EEXIST, "Choose a new output file", output)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest

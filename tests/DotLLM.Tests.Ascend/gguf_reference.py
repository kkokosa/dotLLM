#!/usr/bin/env python3
"""Create deterministic tiny GGUFs or expand a GGUF into F32/F16 for CPU reference.

Requires numpy. Model files stay outside the repository. Expansion preserves the
original metadata and weights; it does not requantize Q8_0 inputs.
"""
import argparse
import json
from pathlib import Path
import struct

import numpy as np


def string(value):
    data = value.encode()
    return struct.pack("<Q", len(data)) + data


def read_string(stream):
    return stream.read(struct.unpack("<Q", stream.read(8))[0]).decode()


def skip_value(stream, kind):
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if kind in sizes:
        stream.seek(sizes[kind], 1)
    elif kind == 8:
        read_string(stream)
    elif kind == 9:
        element, count = struct.unpack("<IQ", stream.read(12))
        if element in sizes:
            stream.seek(count * sizes[element], 1)
        else:
            for _ in range(count):
                skip_value(stream, element)
    else:
        raise ValueError(f"Unsupported metadata type {kind}")


def descriptors(tensors, types):
    offset = 0
    result = bytearray()
    offsets = []
    for (name, dims, _), dtype in zip(tensors, types):
        result += string(name) + struct.pack("<I", len(dims))
        result += struct.pack("<" + "Q" * len(dims), *dims)
        result += struct.pack("<IQ", dtype, offset)
        offsets.append(offset)
        count = int(np.prod(dims))
        size = {0: count * 4, 1: count * 2, 8: count // 32 * 34}[dtype]
        offset = (offset + size + 31) // 32 * 32
    return result, offsets


def expand(source, destination, dtype):
    with open(source, "rb") as stream:
        magic, version, count, meta_count = struct.unpack("<IIQQ", stream.read(24))
        if magic != 0x46554747 or version != 3:
            raise ValueError("Expected GGUF v3")
        alignment = 32
        for _ in range(meta_count):
            key = read_string(stream)
            kind = struct.unpack("<I", stream.read(4))[0]
            if key == "general.alignment":
                alignment = struct.unpack("<I", stream.read(4))[0]
            else:
                skip_value(stream, kind)
        metadata_end = stream.tell()
        stream.seek(0)
        prefix = stream.read(metadata_end)
        tensors, source_types, source_offsets = [], [], []
        for _ in range(count):
            name = read_string(stream)
            rank = struct.unpack("<I", stream.read(4))[0]
            dims = struct.unpack("<" + "Q" * rank, stream.read(rank * 8))
            kind, offset = struct.unpack("<IQ", stream.read(12))
            if kind not in (0, 1, 8):
                raise ValueError(f"{name}: unsupported type {kind}")
            tensors.append((name, dims, None))
            source_types.append(kind)
            source_offsets.append(offset)
        data_start = (stream.tell() + alignment - 1) // alignment * alignment
        if alignment != 32:
            raise ValueError("Reference converter currently requires alignment 32")
        table, offsets = descriptors(tensors, [dtype] * count)
        with open(destination, "wb") as output:
            output.write(prefix + table)
            output.write(b"\0" * ((-output.tell()) % alignment))
            output_start = output.tell()
            for (name, dims, _), kind, offset, target in zip(tensors, source_types, source_offsets, offsets):
                stream.seek(data_start + offset)
                output.seek(output_start + target)
                remaining = int(np.prod(dims))
                while remaining:
                    elements = min(remaining, 262144)
                    if kind == 8:
                        block_type = np.dtype([("d", "<f2"), ("q", "i1", (32,))])
                        blocks = np.frombuffer(stream.read(elements // 32 * 34), dtype=block_type)
                        values = (blocks["d"].astype(np.float32)[:, None] * blocks["q"]).reshape(-1)
                    else:
                        values = np.frombuffer(stream.read(elements * (4 if kind == 0 else 2)),
                                               dtype="<f4" if kind == 0 else "<f2")
                    output.write(values.astype("<f4" if dtype == 0 else "<f2").tobytes())
                    remaining -= elements
                print(json.dumps({"tensor": name, "elements": int(np.prod(dims))}), flush=True)


def tiny(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for arch in ("llama", "qwen2"):
        for dtype in (0, 1, 8):
            rng = np.random.default_rng(20261024)
            metadata = {"general.architecture": arch, f"{arch}.embedding_length": 32,
                        f"{arch}.block_count": 2, f"{arch}.feed_forward_length": 64,
                        f"{arch}.attention.head_count": 4, f"{arch}.attention.head_count_kv": 2,
                        f"{arch}.context_length": 32, f"{arch}.vocab_size": 64,
                        f"{arch}.rope.freq_base": 10000.0, f"{arch}.rope.dimension_count": 8,
                        f"{arch}.attention.layer_norm_rms_epsilon": 1e-5}
            meta = bytearray()
            for key, value in metadata.items():
                kind = 8 if isinstance(value, str) else 6 if isinstance(value, float) else 4
                meta += string(key) + struct.pack("<I", kind)
                meta += string(value) if kind == 8 else struct.pack("<f" if kind == 6 else "<I", value)
            tensors, types = [], []

            def add(name, dims, norm=False):
                values = np.ones(int(np.prod(dims)), np.float32) if norm else rng.normal(
                    0, 0.05, int(np.prod(dims))).astype(np.float32)
                tensors.append((name, dims, values))
                types.append(0 if len(dims) == 1 else dtype)

            add("token_embd.weight", (32, 64))
            add("output_norm.weight", (32,), True)
            if arch == "qwen2":
                add("output.weight", (32, 64))
            for layer in range(2):
                p = f"blk.{layer}."
                add(p + "attn_norm.weight", (32,), True)
                add(p + "ffn_norm.weight", (32,), True)
                for name, cols, rows in (("attn_q", 32, 32), ("attn_k", 32, 16),
                                        ("attn_v", 32, 16), ("attn_output", 32, 32),
                                        ("ffn_gate", 32, 64), ("ffn_up", 32, 64), ("ffn_down", 64, 32)):
                    add(p + name + ".weight", (cols, rows))
                    if arch == "qwen2":
                        add(p + name + ".bias", (rows,))
            table, offsets = descriptors(tensors, types)
            path = directory / f"tiny-{arch}-{['F32', 'F16'][dtype] if dtype < 2 else 'Q8_0'}.gguf"
            with open(path, "wb") as out:
                out.write(struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata)) + meta + table)
                out.write(b"\0" * (-out.tell() % 32))
                base = out.tell()
                for (_, _, values), kind, offset in zip(tensors, types, offsets):
                    out.seek(base + offset)
                    if kind == 8:
                        blocks = values.reshape(-1, 32)
                        scale = (np.max(np.abs(blocks), axis=1) / 127).astype("<f2")
                        quantized = np.rint(blocks / scale.astype(np.float32)[:, None]).clip(-127, 127).astype("i1")
                        for d, q in zip(scale, quantized):
                            out.write(d.tobytes() + q.tobytes())
                    else:
                        out.write(values.astype("<f4" if kind == 0 else "<f2").tobytes())
            print(path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiny", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=["F32", "F16"], default="F32")
    args = parser.parse_args()
    if args.tiny:
        tiny(args.tiny)
    elif args.source and args.output:
        expand(args.source, args.output, 0 if args.dtype == "F32" else 1)
    else:
        parser.error("Use --tiny DIR or --source FILE --output FILE")

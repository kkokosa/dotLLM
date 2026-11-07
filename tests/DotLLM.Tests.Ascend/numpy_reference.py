#!/usr/bin/env python3
"""Independent dense FP32 math over expanded GGUF weights; emits teacher-forced logits.

No DotLLM kernels are used. Matrices follow GGUF row order and architecture-specific
RoPE pairing. Run with OPENBLAS_NUM_THREADS=4 for reproducible CPU resource use.
"""
import argparse
import json
import struct
import numpy as np
from gguf_reference import read_string, skip_value


def load(path):
    metadata, tensors = {}, {}
    with open(path, "rb") as stream:
        magic, version, count, kvs = struct.unpack("<IIQQ", stream.read(24))
        assert magic == 0x46554747 and version == 3
        for _ in range(kvs):
            key, kind = read_string(stream), struct.unpack("<I", stream.read(4))[0]
            if key.startswith(("llama.", "qwen2.")) or key == "general.architecture":
                if kind == 8:
                    value = read_string(stream)
                elif kind in (4, 6):
                    value = struct.unpack("<I" if kind == 4 else "<f", stream.read(4))[0]
                else:
                    skip_value(stream, kind)
                    continue
                metadata[key] = value
            else:
                skip_value(stream, kind)
        descriptors = []
        for _ in range(count):
            name = read_string(stream)
            rank = struct.unpack("<I", stream.read(4))[0]
            dims = struct.unpack("<" + "Q" * rank, stream.read(8 * rank))
            kind, offset = struct.unpack("<IQ", stream.read(12))
            assert kind == 0, "Expand all tensors to F32 first"
            descriptors.append((name, dims, offset))
        data_start = (stream.tell() + 31) // 32 * 32
    for name, dims, offset in descriptors:
        tensors[name] = np.memmap(path, mode="r", dtype="<f4", offset=data_start + offset, shape=dims[::-1])
    return metadata, tensors


def forward(tokens, metadata, weights):
    arch = metadata["general.architecture"]
    def cfg(key):
        return metadata[arch + "." + key]
    heads, kvheads = cfg("attention.head_count"), cfg("attention.head_count_kv")
    width, layers = cfg("embedding_length"), cfg("block_count")
    dim = width // heads
    epsilon = cfg("attention.layer_norm_rms_epsilon")
    theta = cfg("rope.freq_base")
    neox = arch == "qwen2"
    length = len(tokens)
    hidden = np.array(weights["token_embd.weight"][tokens], dtype=np.float32)
    frequencies = np.power(np.float32(theta), -2 * np.arange(dim // 2, dtype=np.float32) / dim)
    angles = np.arange(length, dtype=np.float32)[:, None] * frequencies[None, :]
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    mask = np.triu(np.full((length, length), -np.inf, np.float32), 1)

    def norm(x, w):
        return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(epsilon))) * w

    def rotate(x, headcount):
        x = x.reshape(length, headcount, dim)
        left, right = (x[..., :dim // 2], x[..., dim // 2:]) if neox else (x[..., ::2], x[..., 1::2])
        a, b = left * cos - right * sin, left * sin + right * cos
        result = np.empty_like(x)
        if neox:
            result[..., :dim // 2], result[..., dim // 2:] = a, b
        else:
            result[..., ::2], result[..., 1::2] = a, b
        return result

    for layer in range(layers):
        p = f"blk.{layer}."
        def project(x, name):
            y = x @ weights[p + name + ".weight"].T
            bias = weights.get(p + name + ".bias")
            return y if bias is None else y + bias
        normalized = norm(hidden, weights[p + "attn_norm.weight"])
        q = rotate(project(normalized, "attn_q"), heads).transpose(1, 0, 2)
        k = rotate(project(normalized, "attn_k"), kvheads).transpose(1, 0, 2)
        v = project(normalized, "attn_v").reshape(length, kvheads, dim).transpose(1, 0, 2)
        k = np.repeat(k, heads // kvheads, axis=0)
        v = np.repeat(v, heads // kvheads, axis=0)
        scores = (q @ k.transpose(0, 2, 1)) / np.float32(np.sqrt(dim)) + mask
        scores -= np.max(scores, axis=-1, keepdims=True)
        probability = np.exp(scores)
        probability /= probability.sum(axis=-1, keepdims=True)
        attention = (probability @ v).transpose(1, 0, 2).reshape(length, width)
        hidden = hidden + project(attention, "attn_output")
        normalized = norm(hidden, weights[p + "ffn_norm.weight"])
        gate, up = project(normalized, "ffn_gate"), project(normalized, "ffn_up")
        activated = gate / (np.float32(1) + np.exp(-gate)) * up
        hidden = hidden + project(activated, "ffn_down")
    last = norm(hidden[-1:], weights["output_norm.weight"])
    return (last @ weights.get("output.weight", weights["token_embd.weight"]).T)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", help="JSON token cases from --emit-text")
    args = parser.parse_args()
    meta, weight = load(args.model)
    arch = meta["general.architecture"]
    vocab = meta[arch + ".vocab_size"] if arch + ".vocab_size" in meta else weight["token_embd.weight"].shape[0]
    if args.cases:
        with open(args.cases) as stream:
            cases = json.load(stream)
    else:
        steps = min(32, meta[arch + ".context_length"] - 8)
        tokens = list(range(1, 9)) + [(i * 7919 + 17) % vocab for i in range(8, 8 + steps)]
        cases = [{"Name": f"teacher_{length}", "Tokens": tokens[:length]} for length in range(8, len(tokens) + 1)]
    with open(args.output, "wb") as output:
        for case in cases:
            logits = forward(case["Tokens"], meta, weight)
            output.write(logits.astype("<f4").tobytes())
            print(json.dumps({"reference": "independent_numpy_fp32", "case": case["Name"],
                              "length": len(case["Tokens"]), "top1": int(logits.argmax()),
                              "min": float(logits.min()), "max": float(logits.max())}), flush=True)

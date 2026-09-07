import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EVAL_ROOT = ROOT / "evaluation"
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))

from benchmarks.da3.bench.datasets.eth3d import ETH3D
from benchmarks.da3.da3bench_utils import sample_frames
from saf3r.models.official_vggt.models.vggt import VGGT
from saf3r.utils.data_utils import load_and_process_images_vggt


NUM_GLOBAL_LAYERS = 24


def parse_layers(value):
    layers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))

    layers = sorted(set(layers))
    invalid = [layer for layer in layers if layer < 0 or layer >= NUM_GLOBAL_LAYERS]
    if invalid:
        raise ValueError(f"Invalid global layer indices: {invalid}")
    return layers


def make_query_indices(num_tokens, sample_count):
    if sample_count is None or sample_count <= 0 or sample_count >= num_tokens:
        return torch.arange(num_tokens, dtype=torch.long)
    return torch.linspace(0, num_tokens - 1, steps=sample_count).round().long()


def make_uniform_indices(num_tokens, key_count, device):
    return torch.arange(key_count, device=device, dtype=torch.float32).mul_(
        float(num_tokens) / float(key_count)
    ).floor_().long()


def empty_raw_stats():
    return {
        "query_count": 0,
        "selected_total": 0,
        "similarity_sum": 0.0,
        "similarity_sq_sum": 0.0,
    }


def add_raw_stats(dst, src):
    for key, value in src.items():
        dst[key] += value


def pairwise_value_similarity_from_indices(v_norm, key_idx):
    """Return per-head/query mean off-diagonal cosine similarity.

    v_norm is [H, N, D] and key_idx is [H, Q, K]. Values are assumed to be
    L2-normalized along D. For K selected values, the average pairwise cosine
    over distinct pairs can be computed from ||sum_i v_i||^2.
    """
    head_count, query_count, key_count = key_idx.shape
    value_dim = v_norm.shape[-1]
    v_expand = v_norm[:, None, :, :].expand(-1, query_count, -1, -1)
    selected = torch.gather(
        v_expand,
        dim=2,
        index=key_idx[..., None].expand(head_count, query_count, key_count, value_dim),
    )
    summed = selected.sum(dim=2, dtype=torch.float32)
    sum_norm_sq = (summed * summed).sum(dim=-1)
    if key_count <= 1:
        return torch.zeros_like(sum_norm_sq)
    return (sum_norm_sq - float(key_count)) / float(key_count * (key_count - 1))


def raw_from_similarity(similarity, selected_per_query, repeat=1):
    sim = similarity.detach().float().cpu()
    repeat = int(repeat)
    return {
        "query_count": int(sim.numel() * repeat),
        "selected_total": int(sim.numel() * repeat * selected_per_query),
        "similarity_sum": float(sim.sum().item() * repeat),
        "similarity_sq_sum": float((sim * sim).sum().item() * repeat),
    }


def finalize_raw(raw):
    count = raw["query_count"]
    mean = raw["similarity_sum"] / count if count else None
    var = None
    std = None
    if count:
        var = max(raw["similarity_sq_sum"] / count - mean * mean, 0.0)
        if var < 1e-12:
            var = 0.0
        std = math.sqrt(var)
    return {
        "query_count": int(count),
        "selected_keys_per_query": (
            raw["selected_total"] / count if count else None
        ),
        "mean_pairwise_value_cosine": mean,
        "std_pairwise_value_cosine": std,
        "selected_total": int(raw["selected_total"]),
    }


class StatsAccumulator:
    def __init__(self):
        self.raw = defaultdict(empty_raw_stats)

    def add(self, scene, layer, head, strategy, raw):
        key = (scene, int(layer), int(head), strategy)
        add_raw_stats(self.raw[key], raw)

    def add_raw_dict(self, raw_dict):
        for packed_key, stats in raw_dict.items():
            scene, layer, head, strategy = packed_key.split("|", 3)
            self.add(scene, int(layer), int(head), strategy, stats)

    def to_raw_dict(self):
        return {
            f"{scene}|{layer}|{head}|{strategy}": stats
            for (scene, layer, head, strategy), stats in sorted(self.raw.items())
        }

    def rows(self, scene_filter=None):
        rows = []
        for (scene, layer, head, strategy), raw in sorted(self.raw.items()):
            if scene_filter is not None and scene != scene_filter:
                continue
            row = {
                "scene": scene,
                "layer": layer,
                "head": head,
                "strategy": strategy,
            }
            row.update(finalize_raw(raw))
            rows.append(row)
        return rows

    def aggregate(self, by):
        buckets = defaultdict(empty_raw_stats)
        for (scene, layer, head, strategy), raw in self.raw.items():
            if by == "layer_head":
                key = ("ALL", layer, head, strategy)
            elif by == "layer":
                key = ("ALL", layer, -1, strategy)
            elif by == "all":
                key = ("ALL", -1, -1, strategy)
            else:
                raise ValueError(f"Unsupported aggregate: {by}")
            add_raw_stats(buckets[key], raw)

        rows = []
        for (scene, layer, head, strategy), raw in sorted(buckets.items()):
            row = {
                "scene": scene,
                "layer": layer,
                "head": head,
                "strategy": strategy,
            }
            row.update(finalize_raw(raw))
            rows.append(row)
        return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene",
        "layer",
        "head",
        "strategy",
        "query_count",
        "selected_keys_per_query",
        "mean_pairwise_value_cosine",
        "std_pairwise_value_cosine",
        "selected_total",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_outputs(accumulator, metadata, output_json, output_dir):
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "metadata": metadata,
        "raw": accumulator.to_raw_dict(),
    }
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)

    write_csv(accumulator.rows(), output_dir / "scene_layer_head_value_similarity.csv")
    write_csv(accumulator.aggregate("layer_head"), output_dir / "layer_head_value_similarity.csv")
    write_csv(accumulator.aggregate("layer"), output_dir / "layer_value_similarity.csv")
    write_csv(accumulator.aggregate("all"), output_dir / "overall_value_similarity.csv")


class VGGTAttentionValueSimilarityAnalyzer:
    def __init__(self, layers, keep_ratio, query_indices, query_chunk_size, accumulator, scene):
        self.layers = set(layers)
        self.keep_ratio = float(keep_ratio)
        self.query_indices = query_indices
        self.query_chunk_size = int(query_chunk_size)
        self.accumulator = accumulator
        self.scene = scene

    def hook(self, layer_idx, module, input_args, input_kwargs, output):
        if layer_idx not in self.layers:
            return

        x = input_args[0]
        pos = input_kwargs.get("pos", None)
        bsz, num_tokens, _ = x.shape
        if bsz != 1:
            raise ValueError(f"Expected batch size 1, got {bsz}")

        key_count = max(1, min(num_tokens, int(math.ceil(num_tokens * self.keep_ratio))))
        print(
            f"[scene {self.scene}][layer {layer_idx:02d}] "
            f"queries={self.query_indices.numel()}, keys={num_tokens}, top={key_count}",
            flush=True,
        )

        with torch.inference_mode():
            qkv = module.qkv(x).reshape(
                bsz,
                num_tokens,
                3,
                module.num_heads,
                module.head_dim,
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            if module.rope is not None:
                q = module.rope(q, pos)
                k = module.rope(k, pos)

            q = q.squeeze(0).float()
            k_t = k.squeeze(0).float().transpose(-1, -2)
            v_norm = F.normalize(v.squeeze(0).float(), p=2, dim=-1, eps=1e-6)
            scale = 1.0 / math.sqrt(module.head_dim)

            uniform_idx_1d = make_uniform_indices(num_tokens, key_count, v_norm.device)
            uniform_idx = uniform_idx_1d[None, None, :].expand(module.num_heads, 1, key_count)
            uniform_sim = pairwise_value_similarity_from_indices(v_norm, uniform_idx)
            for head in range(module.num_heads):
                self.accumulator.add(
                    self.scene,
                    layer_idx,
                    head,
                    "uniform_10pct",
                    raw_from_similarity(uniform_sim[head], key_count, repeat=self.query_indices.numel()),
                )
            del uniform_idx, uniform_idx_1d, uniform_sim

            query_indices = self.query_indices.to(device=x.device, dtype=torch.long)
            for start in range(0, query_indices.numel(), self.query_chunk_size):
                end = min(start + self.query_chunk_size, query_indices.numel())
                q_chunk = q[:, query_indices[start:end], :]
                scores = torch.matmul(q_chunk, k_t) * scale
                top_idx = torch.topk(scores, k=key_count, dim=-1, largest=True, sorted=False).indices
                top_sim = pairwise_value_similarity_from_indices(v_norm, top_idx)
                for head in range(module.num_heads):
                    self.accumulator.add(
                        self.scene,
                        layer_idx,
                        head,
                        "attention_top10pct",
                        raw_from_similarity(top_sim[head], key_count),
                    )
                del q_chunk, scores, top_idx, top_sim

            del qkv, q, k, v, k_t, v_norm


def load_model(ckpt_path, device):
    model = VGGT(save_intermediates=False)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def default_scenes():
    return [
        "courtyard",
        "electro",
        "kicker",
        "pipes",
        "relief",
        "delivery_area",
        "facade",
        "office",
        "playground",
        "relief_2",
        "terrains",
    ]


def analyze(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    layers = parse_layers(args.layers)
    scenes = args.scenes or default_scenes()

    dataset = ETH3D()
    dataset.data_root = args.data_root
    model = load_model(args.ckpt_path, device)

    accumulator = StatsAccumulator()
    metadata = {
        "model": "VGGT",
        "dataset": "ETH3D",
        "data_root": args.data_root,
        "ckpt_path": args.ckpt_path,
        "scenes": scenes,
        "max_frames": args.max_frames,
        "sample_mode": args.sample_mode,
        "layers": layers,
        "keep_ratio": args.keep_ratio,
        "query_sample_count": args.query_sample_count,
        "query_chunk_size": args.query_chunk_size,
        "strategies": ["attention_top10pct", "uniform_10pct"],
        "notes": [
            "All tokens are used as both queries and keys unless query_sample_count > 0.",
            "Special tokens and first-frame tokens are not handled separately.",
            "attention_top10pct ranks keys by raw QK logits, equivalent to ranking by attention weights.",
            "The reported value similarity is mean off-diagonal pairwise cosine among selected L2-normalized values.",
            "uniform_10pct uses evenly spaced key indices and is query-independent.",
        ],
        "completed_scenes": [],
    }

    for scene in scenes:
        scene_start = time.perf_counter()
        scene_data = dataset.get_data(scene)
        _, scene_data = sample_frames(
            scene_data,
            scene,
            max_frames=args.max_frames,
            sample_mode=args.sample_mode,
            seed=args.seed,
        )
        images = load_and_process_images_vggt(scene_data.image_files).to(device)
        num_frames = images.shape[0]
        num_tokens = num_frames * ((images.shape[-2] // 14) * (images.shape[-1] // 14) + 5)
        query_indices = make_query_indices(num_tokens, args.query_sample_count)

        print(
            f"[scene {scene}] frames={num_frames}, tokens={num_tokens}, "
            f"query_count={query_indices.numel()}, keep={math.ceil(num_tokens * args.keep_ratio)}",
            flush=True,
        )

        analyzer = VGGTAttentionValueSimilarityAnalyzer(
            layers=layers,
            keep_ratio=args.keep_ratio,
            query_indices=query_indices,
            query_chunk_size=args.query_chunk_size,
            accumulator=accumulator,
            scene=scene,
        )

        hooks = []
        for layer in layers:
            block = model.aggregator.global_blocks[layer]
            hooks.append(
                block.attn.register_forward_hook(
                    lambda module, input_args, input_kwargs, output, layer=layer: analyzer.hook(
                        layer, module, input_args, input_kwargs, output
                    ),
                    with_kwargs=True,
                )
            )

        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=dtype, enabled=device.type == "cuda"
        ):
            model.aggregator(images.unsqueeze(0))

        for hook in hooks:
            hook.remove()
        del images, analyzer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        metadata["completed_scenes"].append(scene)
        save_outputs(accumulator, metadata, args.output_json, args.output_dir)
        print(f"[scene {scene}] done in {time.perf_counter() - scene_start:.1f}s", flush=True)

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV dir: {args.output_dir}")


def merge(args):
    accumulator = StatsAccumulator()
    metadata = {
        "merged_from": args.merge,
        "notes": ["Merged raw sums from partial value-similarity analyses."],
    }
    completed = []
    for path in args.merge:
        with open(path) as f:
            payload = json.load(f)
        if "model" not in metadata:
            metadata.update(payload.get("metadata", {}))
            metadata["merged_from"] = args.merge
        completed.extend(payload.get("metadata", {}).get("completed_scenes", []))
        accumulator.add_raw_dict(payload["raw"])

    metadata["completed_scenes"] = sorted(set(completed))
    save_outputs(accumulator, metadata, args.output_json, args.output_dir)
    print(f"Saved merged JSON: {args.output_json}")
    print(f"Saved merged CSV dir: {args.output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze VGGT value similarity of selected attention keys.")
    parser.add_argument("--data-root", default="/ssd_data/mmc_lyxiang/dataset/eth3d")
    parser.add_argument("--ckpt-path", default="/data/mmc_lyxiang/3D/vggt/ckpt/model.pt")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--sample-mode", default="uniform", choices=["uniform", "random"])
    parser.add_argument("--layers", default="0-23")
    parser.add_argument("--keep-ratio", type=float, default=0.10)
    parser.add_argument(
        "--query-sample-count",
        type=int,
        default=0,
        help="Use <=0 for all tokens as queries; positive values use evenly sampled queries for smoke tests.",
    )
    parser.add_argument("--query-chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merge", nargs="*", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    if args.keep_ratio <= 0 or args.keep_ratio > 1:
        raise ValueError(f"keep-ratio must be in (0, 1], got {args.keep_ratio}")
    if args.query_chunk_size <= 0:
        raise ValueError(f"query-chunk-size must be positive, got {args.query_chunk_size}")

    if args.merge:
        merge(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()

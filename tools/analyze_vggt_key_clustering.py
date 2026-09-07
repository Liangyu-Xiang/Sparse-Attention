import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


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


NUM_SPECIAL_TOKENS = 5
PATCH_SIZE = 14
NUM_GLOBAL_LAYERS = 24
NUM_BLOCK_BINS = 4


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


def make_patch_distance_matrix(p_h, p_w):
    rows = np.arange(p_h, dtype=np.float32)
    cols = np.arange(p_w, dtype=np.float32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    coords = np.stack([rr.reshape(-1), cc.reshape(-1)], axis=1)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1)).astype(np.float32)


def make_patch_blocks(p_h, p_w):
    rows = np.arange(p_h)
    cols = np.arange(p_w)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    block_r = np.minimum((rr * NUM_BLOCK_BINS) // p_h, NUM_BLOCK_BINS - 1)
    block_c = np.minimum((cc * NUM_BLOCK_BINS) // p_w, NUM_BLOCK_BINS - 1)
    return (block_r * NUM_BLOCK_BINS + block_c).reshape(-1).astype(np.int64)


def make_token_indices(num_frames, patches_per_frame, device):
    tokens_per_frame = patches_per_frame + NUM_SPECIAL_TOKENS
    image_token_abs = []
    for frame_idx in range(num_frames):
        base = frame_idx * tokens_per_frame + NUM_SPECIAL_TOKENS
        image_token_abs.extend(range(base, base + patches_per_frame))
    image_token_abs = torch.tensor(image_token_abs, device=device, dtype=torch.long)
    return image_token_abs


def make_query_indices(num_image_tokens, sample_count):
    if sample_count is None or sample_count <= 0 or sample_count >= num_image_tokens:
        return np.arange(num_image_tokens, dtype=np.int64)
    return np.linspace(0, num_image_tokens - 1, num=sample_count).round().astype(np.int64)


def empty_raw_stats():
    return {
        "query_count": 0,
        "selected_total": 0,
        "spatial_nn_sum": 0.0,
        "spatial_nn_norm_sum": 0.0,
        "spatial_nn_count": 0,
        "block_coverage_sum_all": 0.0,
        "block_coverage_count_all": 0,
        "block_coverage_sum_hit": 0.0,
        "block_coverage_count_hit": 0,
        "frame_hit_count": 0,
        "frame_total": 0,
        "temporal_nn_sum": 0.0,
        "temporal_nn_count": 0,
    }


def add_raw_stats(dst, src):
    for key, value in src.items():
        dst[key] += value


def compute_selection_stats(selected, num_frames, patches_per_frame, patch_blocks, patch_dist):
    """Compute clustering stats for selected image-key indices.

    selected is [Q, K] in flattened image-token coordinates:
    key = frame_idx * patches_per_frame + patch_idx.
    """
    selected = np.asarray(selected, dtype=np.int64)
    if selected.ndim != 2:
        raise ValueError(f"Expected selected shape [Q, K], got {selected.shape}")

    q_count, key_count = selected.shape
    diag = max(float(patch_dist.max()), 1.0)

    raw = empty_raw_stats()
    raw["query_count"] = int(q_count)
    raw["selected_total"] = int(q_count * key_count)
    raw["frame_total"] = int(q_count * num_frames)
    raw["block_coverage_count_all"] = int(q_count * num_frames)

    if q_count == 0 or key_count == 0:
        return raw

    frames = selected // patches_per_frame
    patches = selected % patches_per_frame
    q_ids = np.repeat(np.arange(q_count, dtype=np.int64), key_count)
    frames_flat = frames.reshape(-1)
    patches_flat = patches.reshape(-1)

    # 4x4 block occupancy per query-frame.
    blocks_flat = patch_blocks[patches_flat]
    q_frame = q_ids * num_frames + frames_flat
    occ = np.zeros((q_count * num_frames, NUM_BLOCK_BINS * NUM_BLOCK_BINS), dtype=bool)
    occ[q_frame, blocks_flat] = True
    block_count = occ.sum(axis=1)
    hit = block_count > 0
    block_ratio = block_count.astype(np.float64) / float(NUM_BLOCK_BINS * NUM_BLOCK_BINS)
    raw["block_coverage_sum_all"] = float(block_ratio.sum())
    raw["frame_hit_count"] = int(hit.sum())
    raw["block_coverage_count_hit"] = int(hit.sum())
    if hit.any():
        raw["block_coverage_sum_hit"] = float(block_ratio[hit].sum())

    # Spatial nearest-neighbor distance within each query-frame.
    order = np.argsort(q_frame * patches_per_frame + patches_flat, kind="mergesort")
    groups_sorted = q_frame[order]
    patches_sorted = patches_flat[order]
    if groups_sorted.size > 0:
        bounds = np.concatenate(
            [
                np.array([0], dtype=np.int64),
                np.flatnonzero(groups_sorted[1:] != groups_sorted[:-1]).astype(np.int64) + 1,
                np.array([groups_sorted.size], dtype=np.int64),
            ]
        )
        spatial_sum = 0.0
        spatial_count = 0
        for start, end in zip(bounds[:-1], bounds[1:]):
            n = int(end - start)
            if n < 2:
                continue
            p = patches_sorted[start:end]
            d = patch_dist[np.ix_(p, p)].copy()
            d[np.arange(n), np.arange(n)] = np.inf
            nearest = d.min(axis=1)
            spatial_sum += float(nearest.sum())
            raw["spatial_nn_norm_sum"] += float((nearest / diag).sum())
            spatial_count += n
        raw["spatial_nn_sum"] = spatial_sum
        raw["spatial_nn_count"] = spatial_count

    # Temporal nearest-neighbor distance at the same patch coordinate.
    same_pixel_group = q_ids * patches_per_frame + patches_flat
    order = np.argsort(same_pixel_group * num_frames + frames_flat, kind="mergesort")
    groups_sorted = same_pixel_group[order]
    frames_sorted = frames_flat[order]
    total = groups_sorted.size
    if total > 1:
        nearest = np.full(total, num_frames + 1, dtype=np.int64)
        same_prev = groups_sorted[1:] == groups_sorted[:-1]
        dt = frames_sorted[1:] - frames_sorted[:-1]
        nearest[1:] = np.where(same_prev, np.minimum(nearest[1:], dt), nearest[1:])
        nearest[:-1] = np.where(same_prev, np.minimum(nearest[:-1], dt), nearest[:-1])
        valid = nearest <= num_frames
        raw["temporal_nn_sum"] = float(nearest[valid].sum())
        raw["temporal_nn_count"] = int(valid.sum())

    return raw


def finalize_stats(raw):
    spatial_mean = None
    spatial_norm = None
    if raw["spatial_nn_count"] > 0:
        spatial_mean = raw["spatial_nn_sum"] / raw["spatial_nn_count"]
        spatial_norm = raw["spatial_nn_norm_sum"] / raw["spatial_nn_count"]

    temporal_mean = None
    if raw["temporal_nn_count"] > 0:
        temporal_mean = raw["temporal_nn_sum"] / raw["temporal_nn_count"]

    return {
        "query_count": int(raw["query_count"]),
        "selected_total": int(raw["selected_total"]),
        "selected_keys_per_query": (
            raw["selected_total"] / raw["query_count"] if raw["query_count"] > 0 else None
        ),
        "spatial_nn_mean_patch": spatial_mean,
        "spatial_nn_mean_norm": spatial_norm,
        "spatial_nn_valid_ratio": (
            raw["spatial_nn_count"] / raw["selected_total"] if raw["selected_total"] > 0 else None
        ),
        "block_coverage_all": (
            raw["block_coverage_sum_all"] / raw["block_coverage_count_all"]
            if raw["block_coverage_count_all"] > 0
            else None
        ),
        "block_coverage_hit": (
            raw["block_coverage_sum_hit"] / raw["block_coverage_count_hit"]
            if raw["block_coverage_count_hit"] > 0
            else None
        ),
        "frame_hit_rate": (
            raw["frame_hit_count"] / raw["frame_total"] if raw["frame_total"] > 0 else None
        ),
        "temporal_nn_mean_frames": temporal_mean,
        "temporal_match_rate": (
            raw["temporal_nn_count"] / raw["selected_total"] if raw["selected_total"] > 0 else None
        ),
    }


class StatsAccumulator:
    def __init__(self):
        self.raw = defaultdict(empty_raw_stats)

    def add(self, layer, ratio, strategy, stats):
        key = (str(layer), f"{float(ratio):g}", strategy)
        add_raw_stats(self.raw[key], stats)

    def add_raw_dict(self, raw_dict):
        for layer, by_ratio in raw_dict.items():
            for ratio, by_strategy in by_ratio.items():
                for strategy, stats in by_strategy.items():
                    self.add(layer, ratio, strategy, stats)

    def to_nested_raw(self):
        out = {}
        for (layer, ratio, strategy), stats in sorted(
            self.raw.items(), key=lambda item: (int(item[0][0]), float(item[0][1]), item[0][2])
        ):
            out.setdefault(layer, {}).setdefault(ratio, {})[strategy] = stats
        return out

    def to_rows(self):
        rows = []
        for (layer, ratio, strategy), raw in sorted(
            self.raw.items(), key=lambda item: (int(item[0][0]), float(item[0][1]), item[0][2])
        ):
            row = {
                "layer": int(layer),
                "ratio": float(ratio),
                "strategy": strategy,
            }
            row.update(finalize_stats(raw))
            rows.append(row)
        return rows


def write_csv(rows, path):
    import csv

    fieldnames = [
        "layer",
        "ratio",
        "strategy",
        "query_count",
        "selected_keys_per_query",
        "spatial_nn_mean_patch",
        "spatial_nn_mean_norm",
        "spatial_nn_valid_ratio",
        "block_coverage_all",
        "block_coverage_hit",
        "frame_hit_rate",
        "temporal_nn_mean_frames",
        "temporal_match_rate",
        "selected_total",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_outputs(accumulator, metadata, output_json, output_csv):
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = accumulator.to_rows()
    payload = {
        "metadata": metadata,
        "raw": accumulator.to_nested_raw(),
        "rows": rows,
    }
    with open(output_json, "w") as f:
        json.dump(payload, f, indent=2)
    write_csv(rows, output_csv)


class VGGTHighKeyAnalyzer:
    def __init__(
        self,
        layers,
        ratios,
        query_image_idx,
        image_token_abs,
        num_frames,
        p_h,
        p_w,
        query_chunk_size,
        accumulator,
    ):
        self.layers = set(layers)
        self.ratios = sorted(float(ratio) for ratio in ratios)
        self.query_image_idx = query_image_idx
        self.image_token_abs = image_token_abs
        self.query_token_abs = image_token_abs[
            torch.as_tensor(query_image_idx, device=image_token_abs.device, dtype=torch.long)
        ]
        self.num_frames = num_frames
        self.p_h = p_h
        self.p_w = p_w
        self.patches_per_frame = p_h * p_w
        self.num_image_tokens = num_frames * self.patches_per_frame
        self.query_chunk_size = query_chunk_size
        self.accumulator = accumulator
        self.patch_blocks = make_patch_blocks(p_h, p_w)
        self.patch_dist = make_patch_distance_matrix(p_h, p_w)
        self.k_by_ratio = {
            ratio: max(1, min(self.num_image_tokens, int(math.ceil(self.num_image_tokens * ratio / 100.0))))
            for ratio in self.ratios
        }
        self.max_k = max(self.k_by_ratio.values())

    def add_uniform_baseline(self, seed):
        generator = torch.Generator(device=self.image_token_abs.device)
        generator.manual_seed(int(seed))
        q_count = len(self.query_image_idx)
        random_scores = torch.rand(
            (q_count, self.num_image_tokens),
            device=self.image_token_abs.device,
            generator=generator,
        )
        uniform_idx = torch.topk(random_scores, k=self.max_k, dim=-1, largest=True, sorted=True).indices
        uniform_idx = uniform_idx.cpu().numpy()
        del random_scores

        stats_by_ratio = {}
        for ratio, key_count in self.k_by_ratio.items():
            stats_by_ratio[ratio] = compute_selection_stats(
                uniform_idx[:, :key_count],
                self.num_frames,
                self.patches_per_frame,
                self.patch_blocks,
                self.patch_dist,
            )

        for layer in self.layers:
            for ratio, stats in stats_by_ratio.items():
                self.accumulator.add(layer, ratio, "uniform_random", stats)

    def hook(self, layer_idx, module, input_args, input_kwargs, output):
        if layer_idx not in self.layers:
            return

        x = input_args[0]
        pos = input_kwargs.get("pos", None)
        bsz, num_tokens, dim = x.shape
        if bsz != 1:
            raise ValueError(f"Expected batch size 1, got {bsz}")

        print(
            f"[layer {layer_idx:02d}] scoring {len(self.query_image_idx)} sampled queries "
            f"against {self.num_image_tokens} image keys",
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
            q, k, _ = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)
            if module.rope is not None:
                q = module.rope(q, pos)
                k = module.rope(k, pos)

            k_all = k.float()
            for start in range(0, len(self.query_token_abs), self.query_chunk_size):
                end = min(start + self.query_chunk_size, len(self.query_token_abs))
                q_sel = q[:, :, self.query_token_abs[start:end], :].float()
                scores = torch.matmul(q_sel, k_all.transpose(-1, -2)) / math.sqrt(module.head_dim)
                weights = torch.softmax(scores, dim=-1)
                image_weights = weights[:, :, :, self.image_token_abs]
                mean_weights = image_weights.mean(dim=1).squeeze(0)
                top_idx = torch.topk(
                    mean_weights,
                    k=self.max_k,
                    dim=-1,
                    largest=True,
                    sorted=True,
                ).indices.cpu().numpy()

                for ratio, key_count in self.k_by_ratio.items():
                    stats = compute_selection_stats(
                        top_idx[:, :key_count],
                        self.num_frames,
                        self.patches_per_frame,
                        self.patch_blocks,
                        self.patch_dist,
                    )
                    self.accumulator.add(layer_idx, ratio, "attention_top", stats)

                del q_sel, scores, weights, image_weights, mean_weights

            del qkv, q, k, k_all


def load_model(ckpt_path, device):
    model = VGGT(save_intermediates=False)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    return model


def analyze_scenes(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    layers = parse_layers(args.layers)

    dataset = ETH3D()
    dataset.data_root = args.data_root
    scenes = args.scenes or [
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
        "ratios": args.ratios,
        "query_sample_count": args.query_sample_count,
        "query_chunk_size": args.query_chunk_size,
        "strategies": ["attention_top", "uniform_random"],
        "notes": [
            "Only image-token queries and image-token keys are used for spatial/temporal stats.",
            "attention_top ranks image keys by mean attention probability across heads.",
            "uniform_random samples the same number of image keys uniformly without replacement.",
        ],
        "patch_grids": [],
        "completed_scenes": [],
    }

    for scene_id, scene in enumerate(scenes):
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
        p_h = images.shape[-2] // PATCH_SIZE
        p_w = images.shape[-1] // PATCH_SIZE
        patches_per_frame = p_h * p_w
        num_image_tokens = num_frames * patches_per_frame

        grid = [int(p_h), int(p_w)]
        if grid not in metadata["patch_grids"]:
            metadata["patch_grids"].append(grid)

        query_image_idx = make_query_indices(num_image_tokens, args.query_sample_count)
        image_token_abs = make_token_indices(num_frames, patches_per_frame, device)
        analyzer = VGGTHighKeyAnalyzer(
            layers=layers,
            ratios=args.ratios,
            query_image_idx=query_image_idx,
            image_token_abs=image_token_abs,
            num_frames=num_frames,
            p_h=p_h,
            p_w=p_w,
            query_chunk_size=args.query_chunk_size,
            accumulator=accumulator,
        )
        analyzer.add_uniform_baseline(seed=args.seed + 1009 * scene_id + 17 * num_image_tokens)

        hooks = []
        for layer in layers:
            block = model.aggregator.global_blocks[layer]
            hooks.append(block.attn.register_forward_hook(
                lambda module, input_args, input_kwargs, output, layer=layer: analyzer.hook(
                    layer, module, input_args, input_kwargs, output
                ),
                with_kwargs=True,
            ))

        print(
            f"[scene {scene}] frames={num_frames}, grid={p_h}x{p_w}, "
            f"image_keys={num_image_tokens}, query_samples={len(query_image_idx)}",
            flush=True,
        )
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            model.aggregator(images.unsqueeze(0))

        for hook in hooks:
            hook.remove()
        del images, analyzer, image_token_abs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        metadata["completed_scenes"].append(scene)
        save_outputs(accumulator, metadata, args.output_json, args.output_csv)
        print(f"[scene {scene}] done in {time.perf_counter() - scene_start:.1f}s", flush=True)

    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV:  {args.output_csv}")


def merge_outputs(args):
    accumulator = StatsAccumulator()
    metadata = {
        "merged_from": args.merge,
        "notes": ["Merged raw sums from partial key-clustering analyses."],
    }

    for path in args.merge:
        with open(path) as f:
            payload = json.load(f)
        if "model" not in metadata:
            metadata.update(payload.get("metadata", {}))
            metadata["merged_from"] = args.merge
        accumulator.add_raw_dict(payload["raw"])

    save_outputs(accumulator, metadata, args.output_json, args.output_csv)
    print(f"Saved merged JSON: {args.output_json}")
    print(f"Saved merged CSV:  {args.output_csv}")


def build_parser():
    parser = argparse.ArgumentParser(description="Analyze VGGT high-attention key clustering.")
    parser.add_argument("--data-root", default="/ssd_data/mmc_lyxiang/dataset/eth3d")
    parser.add_argument("--ckpt-path", default="/data/mmc_lyxiang/3D/vggt/ckpt/model.pt")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--sample-mode", default="uniform", choices=["uniform", "random"])
    parser.add_argument("--layers", default="0-23")
    parser.add_argument("--ratios", nargs="+", type=float, default=[10.0])
    parser.add_argument(
        "--query-sample-count",
        type=int,
        default=64,
        help="Number of image-token queries sampled per scene. Use 0 or a negative value for all queries.",
    )
    parser.add_argument("--query-chunk-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--merge", nargs="*", default=None)
    return parser


def main():
    args = build_parser().parse_args()
    if args.merge:
        merge_outputs(args)
    else:
        analyze_scenes(args)


if __name__ == "__main__":
    main()

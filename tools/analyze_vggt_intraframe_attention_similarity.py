import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

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
NUM_ATTENTION_HEADS = 16
EPS = 1e-12


def parse_int_list(value, max_exclusive, name):
    out = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(part))
    out = sorted(dict.fromkeys(out))
    invalid = [item for item in out if item < 0 or item >= max_exclusive]
    if invalid:
        raise ValueError(f"Invalid {name}: {invalid}")
    return out


def make_query_positions(patches_per_frame, patch_query_sample_count, include_special_queries, device):
    patch_positions = torch.arange(
        NUM_SPECIAL_TOKENS,
        NUM_SPECIAL_TOKENS + patches_per_frame,
        device=device,
        dtype=torch.long,
    )
    if 0 < patch_query_sample_count < patches_per_frame:
        sampled = torch.linspace(
            0,
            patches_per_frame - 1,
            steps=patch_query_sample_count,
            device=device,
        ).round().long()
        patch_positions = patch_positions[sampled]

    if not include_special_queries:
        return patch_positions

    special_positions = torch.arange(NUM_SPECIAL_TOKENS, device=device, dtype=torch.long)
    return torch.cat([special_positions, patch_positions], dim=0)


def make_empty_raw():
    return {
        "rows": 0,
        "cosine_sum": 0.0,
        "tv_sum": 0.0,
        "js_sum": 0.0,
        "topk_overlap_sum": 0.0,
        "top1_agreement_sum": 0.0,
        "left_entropy_sum": 0.0,
        "right_entropy_sum": 0.0,
    }


def add_raw(dst, src):
    for key, value in src.items():
        dst[key] += value


class SimilarityAccumulator:
    def __init__(self, topk):
        self.topk = int(topk)
        self.raw = defaultdict(make_empty_raw)

    def add_tensor(self, scene, layer, heads, comparison, left, right):
        if left.numel() == 0:
            return
        left = left.float()
        right = right.float()
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch for {comparison}: {left.shape} vs {right.shape}")

        key_count = left.shape[-1]
        topk = max(1, min(self.topk, key_count))

        dot = (left * right).sum(dim=-1)
        left_norm = left.pow(2).sum(dim=-1).sqrt()
        right_norm = right.pow(2).sum(dim=-1).sqrt()
        cosine = dot / (left_norm * right_norm).clamp_min(EPS)
        tv = 0.5 * (left - right).abs().sum(dim=-1)

        left_safe = left.clamp_min(EPS)
        right_safe = right.clamp_min(EPS)
        mid = (0.5 * (left + right)).clamp_min(EPS)
        js = 0.5 * (left_safe * (left_safe.log() - mid.log())).sum(dim=-1)
        js = js + 0.5 * (right_safe * (right_safe.log() - mid.log())).sum(dim=-1)

        left_entropy = -(left_safe * left_safe.log()).sum(dim=-1)
        right_entropy = -(right_safe * right_safe.log()).sum(dim=-1)
        top1 = (left.argmax(dim=-1) == right.argmax(dim=-1)).float()

        left_top = left.topk(k=topk, dim=-1, largest=True, sorted=False).indices
        right_top = right.topk(k=topk, dim=-1, largest=True, sorted=False).indices
        overlap = (left_top[..., :, None] == right_top[..., None, :]).any(dim=-1).float()
        overlap = overlap.sum(dim=-1) / float(topk)

        per_head = {
            "rows": torch.full((left.shape[0],), left.shape[1] * left.shape[2], device=left.device),
            "cosine_sum": cosine.sum(dim=(1, 2)),
            "tv_sum": tv.sum(dim=(1, 2)),
            "js_sum": js.sum(dim=(1, 2)),
            "topk_overlap_sum": overlap.sum(dim=(1, 2)),
            "top1_agreement_sum": top1.sum(dim=(1, 2)),
            "left_entropy_sum": left_entropy.sum(dim=(1, 2)),
            "right_entropy_sum": right_entropy.sum(dim=(1, 2)),
        }

        for head_pos, head in enumerate(heads):
            row_key = (scene, int(layer), int(head), comparison)
            raw = self.raw[row_key]
            for metric_name, values in per_head.items():
                raw[metric_name] += float(values[head_pos].item())

    def add_raw_dict(self, raw_dict):
        for key, raw in raw_dict.items():
            scene, layer, head, comparison = key.split("|", 3)
            add_raw(self.raw[(scene, int(layer), int(head), comparison)], raw)

    def to_raw_dict(self):
        return {
            f"{scene}|{layer}|{head}|{comparison}": raw
            for (scene, layer, head, comparison), raw in sorted(
                self.raw.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])
            )
        }

    def to_rows(self, include_all=True):
        rows = []
        merged = defaultdict(make_empty_raw)

        for (scene, layer, head, comparison), raw in sorted(
            self.raw.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])
        ):
            rows.append(self._make_row(scene, layer, head, comparison, raw))
            if include_all:
                add_raw(merged[("ALL", layer, head, comparison)], raw)

        if include_all:
            for (scene, layer, head, comparison), raw in sorted(
                merged.items(), key=lambda item: (item[0][1], item[0][2], item[0][3])
            ):
                rows.append(self._make_row(scene, layer, head, comparison, raw))

        return rows

    @staticmethod
    def _make_row(scene, layer, head, comparison, raw):
        rows = int(raw["rows"])
        denom = rows if rows > 0 else 1
        return {
            "scene": scene,
            "layer": int(layer),
            "head": int(head),
            "comparison": comparison,
            "rows": rows,
            "cosine": raw["cosine_sum"] / denom,
            "total_variation": raw["tv_sum"] / denom,
            "js_divergence": raw["js_sum"] / denom,
            "topk_overlap": raw["topk_overlap_sum"] / denom,
            "top1_agreement": raw["top1_agreement_sum"] / denom,
            "left_entropy": raw["left_entropy_sum"] / denom,
            "right_entropy": raw["right_entropy_sum"] / denom,
        }


class IntraframeSimilarityHook:
    def __init__(self, layers, heads, query_positions, num_frames, tokens_per_frame, accumulator):
        self.layers = set(layers)
        self.heads = heads
        self.head_index = None
        self.query_positions = query_positions
        self.num_frames = int(num_frames)
        self.tokens_per_frame = int(tokens_per_frame)
        self.accumulator = accumulator
        self.frame_attn_by_layer = {}

    def _select_heads(self, tensor):
        if self.head_index is None:
            self.head_index = torch.tensor(self.heads, device=tensor.device, dtype=torch.long)
        return tensor.index_select(1, self.head_index)

    def _qk(self, module, x, pos):
        bsz, n_tokens, _ = x.shape
        qkv = (
            module.qkv(x)
            .reshape(bsz, n_tokens, 3, module.num_heads, module.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, _ = qkv.unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)
        if module.rope is not None:
            q = module.rope(q, pos)
            k = module.rope(k, pos)
        q = self._select_heads(q).float()
        k = self._select_heads(k).float()
        return q, k

    def frame_hook(self, scene, layer, module, input_args, input_kwargs):
        if layer not in self.layers:
            return

        x = input_args[0]
        pos = input_kwargs.get("pos", None)
        if pos is None and len(input_args) > 1:
            pos = input_args[1]
        if x.shape[0] != self.num_frames or x.shape[1] != self.tokens_per_frame:
            raise ValueError(
                f"Unexpected frame attention input shape {tuple(x.shape)}; "
                f"expected [{self.num_frames}, {self.tokens_per_frame}, C]"
            )

        q, k = self._qk(module, x, pos)
        q = q[:, :, self.query_positions, :].permute(1, 0, 2, 3).contiguous()
        k = k.permute(1, 0, 2, 3).contiguous()
        scores = torch.einsum("hsqd,hskd->hsqk", q, k) / math.sqrt(module.head_dim)
        frame_attn = torch.softmax(scores, dim=-1).to(torch.float16)
        self.frame_attn_by_layer[layer] = frame_attn

    def global_hook(self, scene, layer, module, input_args, input_kwargs):
        if layer not in self.layers:
            return

        x = input_args[0]
        pos = input_kwargs.get("pos", None)
        if pos is None and len(input_args) > 1:
            pos = input_args[1]
        if x.shape[0] != 1 or x.shape[1] != self.num_frames * self.tokens_per_frame:
            raise ValueError(
                f"Unexpected global attention input shape {tuple(x.shape)}; "
                f"expected [1, {self.num_frames * self.tokens_per_frame}, C]"
            )

        q, k = self._qk(module, x, pos)
        q = q.squeeze(0).reshape(len(self.heads), self.num_frames, self.tokens_per_frame, module.head_dim)
        k = k.squeeze(0).reshape(len(self.heads), self.num_frames, self.tokens_per_frame, module.head_dim)
        q = q[:, :, self.query_positions, :].contiguous()
        scores = torch.einsum("hsqd,hskd->hsqk", q, k) / math.sqrt(module.head_dim)
        global_intraframe = torch.softmax(scores, dim=-1)

        frame_attn = self.frame_attn_by_layer.pop(layer)
        self.accumulator.add_tensor(
            scene,
            layer,
            self.heads,
            "global_intraframe_vs_previous_frame_block",
            global_intraframe,
            frame_attn,
        )

        if self.num_frames > 1:
            self.accumulator.add_tensor(
                scene,
                layer,
                self.heads,
                "global_intraframe_adjacent_temporal_frames",
                global_intraframe[:, 1:, :, :],
                global_intraframe[:, :-1, :, :],
            )
            self.accumulator.add_tensor(
                scene,
                layer,
                self.heads,
                "frame_block_adjacent_temporal_frames",
                frame_attn[:, 1:, :, :],
                frame_attn[:, :-1, :, :],
            )


def load_model(ckpt_path, device):
    model = VGGT(save_intermediates=False)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def write_csv(rows, output_csv):
    fieldnames = [
        "scene",
        "layer",
        "head",
        "comparison",
        "rows",
        "cosine",
        "total_variation",
        "js_divergence",
        "topk_overlap",
        "top1_agreement",
        "left_entropy",
        "right_entropy",
    ]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(accumulator, metadata, output_json, output_csv):
    output_json = Path(output_json)
    output_csv = Path(output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "raw": accumulator.to_raw_dict(),
        "rows": accumulator.to_rows(include_all=True),
    }
    output_json.write_text(json.dumps(payload, indent=2))
    write_csv(payload["rows"], output_csv)


def analyze(args):
    layers = parse_int_list(args.layers, NUM_GLOBAL_LAYERS, "layers")
    heads = parse_int_list(args.heads, NUM_ATTENTION_HEADS, "heads")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

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
    accumulator = SimilarityAccumulator(topk=args.topk)
    metadata = {
        "model": "VGGT",
        "dataset": "ETH3D",
        "data_root": str(Path(args.data_root).resolve()),
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "scenes": scenes,
        "completed_scenes": [],
        "max_frames": int(args.max_frames),
        "sample_mode": args.sample_mode,
        "layers": layers,
        "heads": heads,
        "patch_query_sample_count": int(args.patch_query_sample_count),
        "include_special_queries": bool(args.include_special_queries),
        "topk": int(args.topk),
        "comparisons": [
            "global_intraframe_vs_previous_frame_block",
            "global_intraframe_adjacent_temporal_frames",
            "frame_block_adjacent_temporal_frames",
        ],
        "notes": [
            "The primary comparison masks each global block to same-frame keys and compares its row-wise softmax distribution to the immediately preceding frame block.",
            "Rows compare the same query token position and same key token coordinate inside each frame.",
            "Temporal comparisons compare frame t against frame t-1 after aligning query/key coordinates.",
        ],
    }

    for scene_id, scene in enumerate(scenes):
        started = time.perf_counter()
        scene_data = dataset.get_data(scene)
        _, scene_data = sample_frames(
            scene_data,
            scene,
            max_frames=args.max_frames,
            sample_mode=args.sample_mode,
            seed=args.seed + scene_id,
        )
        images = load_and_process_images_vggt(scene_data.image_files).to(device)
        num_frames = int(images.shape[0])
        patch_h = int(images.shape[-2] // PATCH_SIZE)
        patch_w = int(images.shape[-1] // PATCH_SIZE)
        patches_per_frame = patch_h * patch_w
        tokens_per_frame = NUM_SPECIAL_TOKENS + patches_per_frame
        query_positions = make_query_positions(
            patches_per_frame,
            args.patch_query_sample_count,
            args.include_special_queries,
            device,
        )

        metadata.setdefault("patch_grids", [])
        grid = [patch_h, patch_w]
        if grid not in metadata["patch_grids"]:
            metadata["patch_grids"].append(grid)

        hooks = IntraframeSimilarityHook(
            layers=layers,
            heads=heads,
            query_positions=query_positions,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            accumulator=accumulator,
        )
        handles = []
        for layer in layers:
            handles.append(
                model.aggregator.frame_blocks[layer].attn.register_forward_pre_hook(
                    lambda module, input_args, input_kwargs, layer=layer: hooks.frame_hook(
                        scene, layer, module, input_args, input_kwargs
                    ),
                    with_kwargs=True,
                )
            )
            handles.append(
                model.aggregator.global_blocks[layer].attn.register_forward_pre_hook(
                    lambda module, input_args, input_kwargs, layer=layer: hooks.global_hook(
                        scene, layer, module, input_args, input_kwargs
                    ),
                    with_kwargs=True,
                )
            )

        print(
            f"[scene {scene}] frames={num_frames}, grid={patch_h}x{patch_w}, "
            f"query_positions={query_positions.numel()}, layers={len(layers)}, heads={len(heads)}",
            flush=True,
        )
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
            model.aggregator(images.unsqueeze(0))

        for handle in handles:
            handle.remove()

        metadata["completed_scenes"].append(scene)
        save_outputs(accumulator, metadata, args.output_json, args.output_csv)
        print(f"[scene {scene}] done in {time.perf_counter() - started:.1f}s", flush=True)

        del images, hooks, handles, query_positions
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_outputs(accumulator, metadata, args.output_json, args.output_csv)
    print(f"Saved JSON: {args.output_json}", flush=True)
    print(f"Saved CSV:  {args.output_csv}", flush=True)


def merge(args):
    accumulator = SimilarityAccumulator(topk=args.topk)
    metadata = {
        "merged_from": [str(Path(path).resolve()) for path in args.merge],
        "notes": ["Merged raw sums from worker outputs."],
    }
    for path in args.merge:
        with open(path) as f:
            payload = json.load(f)
        if "model" not in metadata:
            metadata.update(payload.get("metadata", {}))
            metadata["merged_from"] = [str(Path(item).resolve()) for item in args.merge]
        accumulator.add_raw_dict(payload["raw"])

    save_outputs(accumulator, metadata, args.output_json, args.output_csv)
    print(f"Saved merged JSON: {args.output_json}", flush=True)
    print(f"Saved merged CSV:  {args.output_csv}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Compare VGGT intra-frame attention weights.")
    parser.add_argument("--data-root", default="/ssd_data/mmc_lyxiang/dataset/eth3d")
    parser.add_argument("--ckpt-path", default="/data/mmc_lyxiang/3D/vggt/ckpt/model.pt")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--sample-mode", default="uniform", choices=["uniform", "random"])
    parser.add_argument("--layers", default="0-23")
    parser.add_argument("--heads", default="0-15")
    parser.add_argument("--patch-query-sample-count", type=int, default=64)
    parser.add_argument("--include-special-queries", action="store_true")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--merge", nargs="*", default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.merge:
        merge(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()

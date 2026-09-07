import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


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


def parse_int_list(value):
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
    return sorted(dict.fromkeys(out))


def ensure_valid_layers(layers):
    invalid = [layer for layer in layers if layer < 0 or layer >= NUM_GLOBAL_LAYERS]
    if invalid:
        raise ValueError(f"Invalid VGGT global layer indices: {invalid}")


def attention_to_png(npy_path, png_path, scale):
    matrix = np.load(npy_path, mmap_mode="r")
    n_tokens = matrix.shape[0]
    image = np.empty(matrix.shape, dtype=np.uint8)

    if scale <= 0:
        scale = 1.0
    denom = math.log1p(scale * n_tokens)
    if denom <= 0:
        denom = 1.0

    chunk = 256
    for start in range(0, n_tokens, chunk):
        end = min(start + chunk, n_tokens)
        rows = np.asarray(matrix[start:end], dtype=np.float32)
        rows = np.log1p(rows * n_tokens) / denom
        np.clip(rows, 0.0, 1.0, out=rows)
        image[start:end] = (rows * 255.0 + 0.5).astype(np.uint8)

    Image.fromarray(image, mode="L").save(png_path, compress_level=1)


class GlobalAttentionDumper:
    def __init__(self, layers, heads, out_dir, query_chunk_size):
        self.layers = set(layers)
        self.heads = list(heads)
        self.out_dir = Path(out_dir)
        self.query_chunk_size = query_chunk_size
        self.records = []

    def hook(self, layer_idx, module, input_args, input_kwargs):
        if layer_idx not in self.layers:
            return

        x = input_args[0]
        pos = input_kwargs.get("pos", None)
        if pos is None and len(input_args) > 1:
            pos = input_args[1]

        batch_size, n_tokens, channels = x.shape
        if batch_size != 1:
            raise ValueError(f"Expected batch size 1, got {batch_size}")

        invalid_heads = [head for head in self.heads if head < 0 or head >= module.num_heads]
        if invalid_heads:
            raise ValueError(f"Invalid heads {invalid_heads}; module has {module.num_heads} heads")

        layer_dir = self.out_dir / f"layer{layer_idx:02d}"
        matrix_dir = layer_dir / "matrices"
        heatmap_dir = layer_dir / "heatmaps"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        print(
            f"[capture] layer={layer_idx} heads={self.heads} "
            f"tokens={n_tokens} channels={channels}",
            flush=True,
        )

        with torch.no_grad():
            qkv = (
                module.qkv(x)
                .reshape(batch_size, n_tokens, 3, module.num_heads, module.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, _ = qkv.unbind(0)
            q, k = module.q_norm(q), module.k_norm(k)

            if module.rope is not None:
                q = module.rope(q, pos)
                k = module.rope(k, pos)

            head_index = torch.tensor(self.heads, dtype=torch.long, device=x.device)
            q = q[0].index_select(0, head_index).float().contiguous()
            k = k[0].index_select(0, head_index).float().contiguous()
            k_t = k.transpose(-1, -2).contiguous()

            matrix_paths = []
            memmaps = []
            max_values = np.zeros(len(self.heads), dtype=np.float64)
            for head in self.heads:
                matrix_path = matrix_dir / f"layer{layer_idx:02d}_head{head:02d}_attention.npy"
                matrix = np.lib.format.open_memmap(
                    matrix_path,
                    mode="w+",
                    dtype=np.float16,
                    shape=(n_tokens, n_tokens),
                )
                matrix_paths.append(matrix_path)
                memmaps.append(matrix)

            for start in range(0, n_tokens, self.query_chunk_size):
                end = min(start + self.query_chunk_size, n_tokens)
                scores = torch.matmul(q[:, start:end], k_t) * module.scale
                attn = torch.softmax(scores, dim=-1)
                attn_cpu = attn.to(torch.float16).cpu().numpy()

                for head_pos, matrix in enumerate(memmaps):
                    matrix[start:end, :] = attn_cpu[head_pos]
                    max_values[head_pos] = max(max_values[head_pos], float(attn_cpu[head_pos].max()))

                del scores, attn, attn_cpu

            for matrix in memmaps:
                matrix.flush()

            del qkv, q, k, k_t
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for head, matrix_path, max_value in zip(self.heads, matrix_paths, max_values):
            png_path = heatmap_dir / f"layer{layer_idx:02d}_head{head:02d}_attention_log.png"
            attention_to_png(matrix_path, png_path, max_value)
            self.records.append(
                {
                    "layer": int(layer_idx),
                    "head": int(head),
                    "matrix_path": str(matrix_path.resolve()),
                    "heatmap_path": str(png_path.resolve()),
                    "shape": [int(n_tokens), int(n_tokens)],
                    "dtype": "float16",
                    "png_mapping": "uint8(log1p(attention * total_tokens) / log1p(max_attention * total_tokens))",
                    "max_attention": float(max_value),
                }
            )

        print(f"[capture] layer={layer_idx} done in {time.perf_counter() - started:.1f}s", flush=True)


def write_index_html(out_dir, records, metadata):
    out_dir = Path(out_dir)
    html_path = out_dir / "index.html"
    panels = []
    for record in sorted(records, key=lambda r: (r["layer"], r["head"])):
        rel = Path(record["heatmap_path"]).relative_to(out_dir)
        panels.append(
            "\n".join(
                [
                    '<section class="panel">',
                    f'  <h2>Layer {record["layer"]}, Head {record["head"]}</h2>',
                    f'  <img src="{rel.as_posix()}" alt="Layer {record["layer"]} head {record["head"]} full token attention matrix">',
                    f'  <p>{record["shape"][0]} x {record["shape"][1]} tokens, max={record["max_attention"]:.6g}</p>',
                    "</section>",
                ]
            )
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VGGT Global Attention Matrices</title>
  <style>
    body {{
      margin: 24px;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f7f7f4;
      color: #1f2428;
    }}
    header {{
      max-width: 1080px;
      margin-bottom: 24px;
    }}
    h1 {{
      font-size: 24px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    .meta {{
      margin: 0;
      color: #53606a;
      font-size: 14px;
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 24px;
    }}
    .panel {{
      min-width: 0;
    }}
    .panel h2 {{
      font-size: 15px;
      margin: 0 0 8px;
    }}
    .panel p {{
      margin: 8px 0 0;
      font-size: 12px;
      color: #53606a;
    }}
    img {{
      width: 100%;
      max-height: 760px;
      object-fit: contain;
      border: 1px solid #cfd6dc;
      background: #000;
      image-rendering: pixelated;
    }}
  </style>
</head>
<body>
  <header>
    <h1>VGGT Global Attention Matrices</h1>
    <p class="meta">Scene {metadata["scene"]}; frame indices {metadata["frame_indices"]}; token order is frame-major with 5 special tokens then 37x37 patch tokens per frame. Each PNG is generated from the full token matrix; exact float16 matrices are stored beside it.</p>
  </header>
  <main class="grid">
{chr(10).join(panels)}
  </main>
</body>
</html>
"""
    html_path.write_text(html)
    return html_path


def main(args):
    layers = parse_int_list(args.layers)
    heads = parse_int_list(args.heads)
    ensure_valid_layers(layers)
    if len(heads) == 0:
        raise ValueError("At least one head is required")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ETH3D()
    dataset.data_root = args.data_root
    scene_data = dataset.get_data(args.scene)
    frame_indices = [args.start_index + i * args.frame_step for i in range(args.num_frames)]
    if frame_indices[-1] >= len(scene_data.image_files):
        raise ValueError(
            f"Scene {args.scene} has {len(scene_data.image_files)} usable frames, "
            f"but requested indices {frame_indices}"
        )

    _, scene_data = sample_frames(
        scene_data,
        args.scene,
        indices=frame_indices,
        sample_mode="uniform",
        max_frames=args.num_frames,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )

    model = VGGT(save_intermediates=False)
    model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu"))
    model = model.to(device).eval()

    images = load_and_process_images_vggt(scene_data.image_files).to(device)
    patch_h = images.shape[-2] // PATCH_SIZE
    patch_w = images.shape[-1] // PATCH_SIZE
    tokens_per_frame = NUM_SPECIAL_TOKENS + patch_h * patch_w
    total_tokens = args.num_frames * tokens_per_frame

    metadata = {
        "model": "VGGT",
        "dataset": "ETH3D",
        "scene": args.scene,
        "data_root": str(Path(args.data_root).resolve()),
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "frame_indices": frame_indices,
        "image_files": [str(Path(path).resolve()) for path in scene_data.image_files],
        "image_basenames": [Path(path).name for path in scene_data.image_files],
        "num_frames": int(args.num_frames),
        "frame_step": int(args.frame_step),
        "layers": layers,
        "heads": heads,
        "patch_grid": [int(patch_h), int(patch_w)],
        "num_special_tokens_per_frame": NUM_SPECIAL_TOKENS,
        "tokens_per_frame": int(tokens_per_frame),
        "total_tokens": int(total_tokens),
        "token_order": "frame-major: for each frame, 5 special tokens followed by row-major patch tokens",
    }

    dumper = GlobalAttentionDumper(
        layers=layers,
        heads=heads,
        out_dir=output_dir,
        query_chunk_size=args.query_chunk_size,
    )

    hooks = []
    for layer in layers:
        block = model.aggregator.global_blocks[layer]
        hooks.append(
            block.attn.register_forward_pre_hook(
                lambda module, input_args, input_kwargs, layer=layer: dumper.hook(
                    layer, module, input_args, input_kwargs
                ),
                with_kwargs=True,
            )
        )

    started = time.perf_counter()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=dtype, enabled=device.type == "cuda"):
        model.aggregator(images.unsqueeze(0))

    for hook in hooks:
        hook.remove()

    metadata["elapsed_seconds"] = time.perf_counter() - started
    metadata["records"] = dumper.records

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2))
    frames_path = output_dir / "selected_frames.txt"
    frames_path.write_text(
        "\n".join(
            f"{idx}\t{Path(path).name}\t{Path(path).resolve()}"
            for idx, path in zip(frame_indices, scene_data.image_files)
        )
        + "\n"
    )
    index_path = write_index_html(output_dir, dumper.records, metadata)

    print(f"Saved manifest: {manifest_path}", flush=True)
    print(f"Saved frames:   {frames_path}", flush=True)
    print(f"Saved index:    {index_path}", flush=True)
    print(f"Saved {len(dumper.records)} full attention matrices.", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Dump VGGT full global attention matrices for selected layers and heads.")
    parser.add_argument("--data-root", default="/ssd_data/mmc_lyxiang/dataset/eth3d")
    parser.add_argument("--ckpt-path", default="/data/mmc_lyxiang/3D/vggt/ckpt/model.pt")
    parser.add_argument("--scene", default="facade")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--layers", default="3,7,13,20")
    parser.add_argument("--heads", default="0,1,2,3")
    parser.add_argument("--query-chunk-size", type=int, default=128)
    parser.add_argument("--output-dir", default="tmp_attention_vis/vggt_facade_8f_stride10_layers3_7_13_20_heads0_3")
    parser.add_argument("--cpu", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())

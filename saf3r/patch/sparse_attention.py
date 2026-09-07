import json
import math
import os
import time

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.functional as F

from ..triton.sparse_topk_attention import (
    fused_sparse_topk_attention, fused_sparse_topk_attention_2
)


def sparse_attention(
    q, k, v, est_topk_idx,
    cfg, p_h, p_w, n_spc, head_idx
):
    """
    Sparse attention interface that selects and applies one of the supported
    sparse-attention patterns to a given head.

    Args:
        q: (1, 1, N, D)
        k: (1, 1, N, D)
        v: (1, 1, N, Dv)
        est_topk_idx: (num_img_tokens, topk) estimated top-k indices for each image token
        cfg: dict containing the sparse attention configuration
        p_h: number of tokens along height
        p_w: number of tokens along width
        n_spc: number of special tokens
        head_idx: index of the current head
    Returns:
        out: (1, 1, N, Dv)
    """
    num_toks = q.shape[2]
    num_toks_per_img = p_h * p_w + n_spc
    num_imgs = num_toks // num_toks_per_img
    spec_idx, img_idx = get_separate_indices(num_imgs, num_toks_per_img, n_spc)

    # select form one of the following modes
    mode = cfg["mode"]
    if mode == "skip":
        return torch.zeros_like(v)
    
    elif mode == "broadcast_first":
        out = broadcast_anchorframe_attention(q, k, v, num_toks_per_img)
    
    elif mode == "all_to_first":
        out = all_to_first(q, k, v, num_toks_per_img)
    
    elif mode == "global_to_frame":
        out = global_to_frame_attention(q, k, v, num_toks_per_img)

    elif mode == "q_probe_topk":
        # prioritize stride (i.e., topk = N / stride)
        if "stride" in cfg:
            topk = max(1, num_toks // cfg["stride"])
        else:
            topk = min(cfg.get("topk", 1024), num_toks)
        q_sample_ratio = cfg.get("q_sample_ratio", 1)

        include_as = cfg.get("include_as", False)
        if include_as:
            anchor_idx = torch.arange(num_toks_per_img)
            indices = torch.cat([spec_idx, anchor_idx])
        else:
            indices = None

        out = q_probe_topk_attention(q, k, v, topk, q_sample_ratio, indices=indices)
    
    elif mode == "dino_topk":
        dino_sp2sp_stride = cfg.get("dino_sp2sp_stride", 1)
        out = dino_topk_attention(q, k, v, spec_idx, img_idx, est_topk_idx, dino_sp2sp_stride)

    elif mode == "random_keys":
        out = random_key_attention(q, k, v, spec_idx, img_idx, cfg, p_h, p_w, head_idx)

    elif mode == "block2x2_keys":
        out = block_grid_attention(q, k, v, spec_idx, img_idx, cfg, p_h, p_w, head_idx)

    elif mode == "ranked_middle_keys":
        out = ranked_middle_key_attention(q, k, v, spec_idx, img_idx, cfg, p_h, p_w, head_idx)

    elif mode == "block_novelty_keys":
        out = block_novelty_key_attention(q, k, v, spec_idx, img_idx, cfg, p_h, p_w, head_idx)

    elif mode == "block_coverage_redundancy_keys":
        out = block_coverage_redundancy_attention(q, k, v, spec_idx, img_idx, cfg, p_h, p_w, head_idx)

    elif mode == "block3x3_violation_keys":
        out = block3x3_violation_guided_attention(
            q, k, v, spec_idx, img_idx, cfg, p_h, p_w, head_idx, est_topk_idx
        )

    elif mode == "hier_split_prune_keys":
        out = hierarchical_split_prune_attention(
            q, k, v, spec_idx, img_idx, cfg, p_h, p_w, head_idx
        )

    elif mode == "fixed_keys":
        out = fixed_key_attention(q, k, v, cfg, head_idx)

    elif mode == "query_topk_keys":
        out = query_topk_key_attention(q, k, v, cfg)

    elif mode == "oracle_region_keys":
        out = oracle_region_attention(q, k, v, cfg, p_h, p_w, n_spc, head_idx)

    elif mode == "stride":
        stride = cfg.get("stride", 4)
        shift = cfg.get("shift", False)
        start_pos = head_idx % stride if shift else 0

        include_as = cfg.get("include_as", True)
        if include_as:
            anchor_idx = torch.arange(num_toks_per_img)
            indices = torch.cat([spec_idx, anchor_idx])
        else:
            indices = None

        out = stride_attention(q, k, v, stride, start_pos, indices=indices)

    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return out


def _make_generator(device, seed):
    generator_device = device if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(int(seed))
    return generator


def _replace_with_random_neighbors(selected_idx, p_h, p_w, neighbor_size, generator):
    if neighbor_size <= 1:
        return selected_idx

    device = selected_idx.device
    patches_per_img = p_h * p_w
    win_h = min(neighbor_size, p_h)
    win_w = min(neighbor_size, p_w)
    frame_idx = selected_idx // patches_per_img
    patch_idx = selected_idx % patches_per_img
    row = patch_idx // p_w
    col = patch_idx % p_w

    start_row = (row - neighbor_size // 2).clamp(0, p_h - win_h)
    start_col = (col - neighbor_size // 2).clamp(0, p_w - win_w)
    row_offsets = torch.arange(win_h, device=device, dtype=torch.long)
    col_offsets = torch.arange(win_w, device=device, dtype=torch.long)
    row_offsets, col_offsets = torch.meshgrid(row_offsets, col_offsets, indexing="ij")
    row_offsets = row_offsets.reshape(1, -1)
    col_offsets = col_offsets.reshape(1, -1)

    cand_row = start_row[:, None] + row_offsets
    cand_col = start_col[:, None] + col_offsets
    candidates = frame_idx[:, None] * patches_per_img + cand_row * p_w + cand_col

    is_other = candidates != selected_idx[:, None]
    scores = torch.rand(candidates.shape, device=device, generator=generator)
    scores = scores.masked_fill(~is_other, -1)
    choice = scores.argmax(dim=1)
    replaced = candidates.gather(1, choice[:, None]).squeeze(1)

    # Degenerate case: a 1-token image has no other key in its neighborhood.
    has_other = is_other.any(dim=1)
    return torch.where(has_other, replaced, selected_idx)


def _deduplicate_and_fill(indices, target_count, universe_size, generator):
    device = indices.device
    unique_indices = torch.unique(indices, sorted=False)
    if unique_indices.numel() >= target_count:
        return unique_indices[:target_count]

    used = torch.zeros(universe_size, dtype=torch.bool, device=device)
    used[unique_indices] = True
    perm = torch.randperm(universe_size, device=device, generator=generator)
    fill = perm[~used[perm]][: target_count - unique_indices.numel()]
    return torch.cat([unique_indices, fill], dim=0)


def random_key_attention(q, k, v, spec_tok_idx, img_tok_idx, cfg, p_h, p_w, head_idx=0):
    """
    Randomly select a global subset of image keys for all image-token queries.

    Special-token queries still attend to all keys. Image-token queries attend to
    the sampled image keys, optionally plus all special keys. If neighbor
    replacement is enabled, each sampled image key is replaced by one random
    key from an image-plane neighborhood in the same frame.
    """
    assert q.shape[0] == 1

    device = q.device
    spec_tok_idx = spec_tok_idx.to(device=device, dtype=torch.long)
    img_tok_idx = img_tok_idx.to(device=device, dtype=torch.long)

    num_img_tokens = img_tok_idx.shape[0]
    keep_ratio = float(cfg.get("keep_ratio", 1.0))
    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    keep_count = max(1, min(num_img_tokens, int(round(num_img_tokens * keep_ratio))))
    layer_idx = int(cfg.get("layer_idx", 0))
    seed = int(cfg.get("seed", 0))
    seed = seed + 1000003 * layer_idx + 1009 * int(head_idx) + 37 * keep_count + 17 * num_img_tokens
    generator = _make_generator(device, seed)

    selected_img_idx = torch.randperm(num_img_tokens, device=device, generator=generator)[:keep_count]

    if bool(cfg.get("replace_with_neighbors", False)):
        neighbor_size = int(cfg.get("neighbor_size", 0))
        if neighbor_size <= 1:
            raise ValueError("neighbor_size must be > 1 when replace_with_neighbors=True")
        selected_img_idx = _replace_with_random_neighbors(
            selected_img_idx, p_h, p_w, neighbor_size, generator
        )
        selected_img_idx = _deduplicate_and_fill(
            selected_img_idx, keep_count, num_img_tokens, generator
        )

    selected_img_idx = torch.sort(selected_img_idx).values
    key_idx = img_tok_idx[selected_img_idx]
    if bool(cfg.get("include_special_keys", True)):
        key_idx = torch.cat([spec_tok_idx, key_idx], dim=0)

    # Special tokens keep full attention so camera/register communication is not sparsified.
    q_spec = q[:, :, spec_tok_idx, :]
    out_spec = F.scaled_dot_product_attention(q_spec, k, v)

    q_img = q[:, :, img_tok_idx, :]
    k_sel = k[:, :, key_idx, :]
    v_sel = v[:, :, key_idx, :]
    out_img = F.scaled_dot_product_attention(q_img, k_sel, v_sel)

    out = torch.empty_like(v)
    out[:, :, spec_tok_idx, :] = out_spec
    out[:, :, img_tok_idx, :] = out_img
    return out


def _make_patch_blocks(num_imgs, p_h, p_w, block_size, device):
    block_idx, _, _ = _make_patch_blocks_and_centers(num_imgs, p_h, p_w, block_size, device)
    return block_idx


def _make_patch_blocks_and_centers(num_imgs, p_h, p_w, block_size, device):
    patches_per_img = p_h * p_w
    blocks = []
    centers = []
    center_pos = []
    for img_id in range(num_imgs):
        img_base = img_id * patches_per_img
        for row in range(0, p_h, block_size):
            row_end = min(row + block_size, p_h)
            for col in range(0, p_w, block_size):
                col_end = min(col + block_size, p_w)
                block = []
                center_r = row + (row_end - row - 1) // 2
                center_c = col + (col_end - col - 1) // 2
                center = img_base + center_r * p_w + center_c
                for rr in range(row, row_end):
                    for cc in range(col, col_end):
                        block.append(img_base + rr * p_w + cc)
                centers.append(center)
                center_pos.append(block.index(center))
                blocks.append(block)

    max_block_tokens = block_size * block_size
    block_idx = torch.full(
        (len(blocks), max_block_tokens),
        -1,
        device=device,
        dtype=torch.long,
    )
    for block_id, block in enumerate(blocks):
        block_idx[block_id, :len(block)] = torch.tensor(block, device=device, dtype=torch.long)
    center_idx = torch.tensor(centers, device=device, dtype=torch.long)
    center_pos = torch.tensor(center_pos, device=device, dtype=torch.long)
    return block_idx, center_idx, center_pos


def _select_one_key_per_block(q, k, img_tok_idx, block_idx, selection, cfg, head_idx):
    B, H, _, D = q.shape
    assert B == 1

    device = q.device
    valid_mask = block_idx >= 0
    safe_block_idx = block_idx.clamp_min(0)

    if selection == "random":
        layer_idx = int(cfg.get("layer_idx", 0))
        seed = int(cfg.get("seed", 0))
        seed = seed + 1000003 * layer_idx + 1009 * int(head_idx) + 17 * img_tok_idx.shape[0]
        generator = _make_generator(device, seed)
        scores = torch.rand((H, block_idx.shape[0], block_idx.shape[1]), device=device, generator=generator)
        scores = scores.masked_fill(~valid_mask[None, :, :], -1)
        choice = scores.argmax(dim=-1)
        return safe_block_idx[None, :, :].expand(H, -1, -1).gather(2, choice[..., None]).squeeze(-1)

    if selection not in {"max", "min"}:
        raise ValueError(f"Unsupported block selection: {selection}")

    num_img_tokens = img_tok_idx.shape[0]
    q_sample_count = min(int(cfg.get("q_sample_count", 256)), num_img_tokens)
    if q_sample_count <= 0:
        raise ValueError(f"q_sample_count must be positive, got {q_sample_count}")

    if q_sample_count == num_img_tokens:
        sample_idx = torch.arange(num_img_tokens, device=device, dtype=torch.long)
    else:
        sample_idx = torch.linspace(
            0,
            num_img_tokens - 1,
            steps=q_sample_count,
            device=device,
        ).round().long()

    q_img = q[:, :, img_tok_idx, :]
    q_sample = q_img[:, :, sample_idx, :]
    scores = torch.matmul(q_sample.float(), k.transpose(-1, -2).float()) / math.sqrt(D)
    weights = torch.softmax(scores, dim=-1)
    key_scores = weights[:, :, :, img_tok_idx].mean(dim=2).squeeze(0)  # [H, num_img_tokens]

    block_scores = key_scores[:, safe_block_idx]  # [H, num_blocks, block_area]
    if selection == "max":
        block_scores = block_scores.masked_fill(~valid_mask[None, :, :], -math.inf)
        choice = block_scores.argmax(dim=-1)
    else:
        block_scores = block_scores.masked_fill(~valid_mask[None, :, :], math.inf)
        choice = block_scores.argmin(dim=-1)

    return safe_block_idx[None, :, :].expand(H, -1, -1).gather(2, choice[..., None]).squeeze(-1)


def _gather_kv_per_head(k, v, key_idx):
    B, H, K = key_idx.shape
    k_idx = key_idx[..., None].expand(B, H, K, k.shape[-1])
    v_idx = key_idx[..., None].expand(B, H, K, v.shape[-1])
    return torch.gather(k, dim=2, index=k_idx), torch.gather(v, dim=2, index=v_idx)


def _pool_image_blocks(features, block_idx, valid_mask):
    H, _, D = features.shape
    num_blocks, block_area = block_idx.shape
    safe_idx = block_idx.clamp_min(0).reshape(-1)
    gathered = features[:, safe_idx, :].reshape(H, num_blocks, block_area, D)
    weights = valid_mask.to(dtype=features.dtype, device=features.device)[None, :, :, None]
    counts = valid_mask.sum(dim=1).clamp_min(1).to(dtype=features.dtype, device=features.device)
    return (gathered * weights).sum(dim=2) / counts[None, :, None]


def _masked_minmax(scores, mask):
    masked_scores = scores.masked_fill(~mask, math.inf)
    min_v = masked_scores.amin(dim=-1, keepdim=True)
    masked_scores = scores.masked_fill(~mask, -math.inf)
    max_v = masked_scores.amax(dim=-1, keepdim=True)
    denom = (max_v - min_v).clamp_min(1e-6)
    norm = (scores - min_v) / denom
    return norm.masked_fill(~mask, 0.0)


def _scatter_random_block_tokens(
    token_scores,
    selected_blocks,
    block_idx,
    valid_mask,
    per_block_keep,
    generator,
    base_scores,
):
    if selected_blocks.numel() == 0:
        return

    device = token_scores.device
    H, Qb, Kb = selected_blocks.shape
    block_area = block_idx.shape[1]
    token_idx = block_idx[selected_blocks]
    token_valid = valid_mask[selected_blocks]

    rand = torch.rand((H, Qb, Kb, block_area), device=device, generator=generator)
    rand = rand.masked_fill(~token_valid, -1.0)
    order = rand.argsort(dim=-1, descending=True)
    rank_values = torch.arange(block_area, device=device, dtype=torch.long)
    rank_values = rank_values.view(1, 1, 1, block_area).expand_as(order)
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, rank_values)

    choose = (rank < per_block_keep[..., None]) & token_valid
    if not choose.any():
        return

    h_idx = torch.arange(H, device=device)[:, None, None, None].expand_as(token_idx)
    q_idx = torch.arange(Qb, device=device)[None, :, None, None].expand_as(token_idx)
    score_values = base_scores[..., None].expand_as(rand) + rand * 1e-3
    token_scores[h_idx[choose], q_idx[choose], token_idx[choose]] = score_values[choose]


def block_grid_attention(q, k, v, spec_tok_idx, img_tok_idx, cfg, p_h, p_w, head_idx=0):
    """
    Keep keys according to 2D patch-grid blocks. `all` keeps every key, while
    `max`, `random`, and `min` keep one representative image key per block.
    """
    assert q.shape[0] == 1

    selection = cfg.get("selection", "all")
    block_size = int(cfg.get("block_size", 2))
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    if selection == "all":
        return F.scaled_dot_product_attention(q, k, v)

    device = q.device
    spec_tok_idx = spec_tok_idx.to(device=device, dtype=torch.long)
    img_tok_idx = img_tok_idx.to(device=device, dtype=torch.long)

    num_imgs = img_tok_idx.numel() // (p_h * p_w)
    block_idx = _make_patch_blocks(num_imgs, p_h, p_w, block_size, device)
    selected_img_idx = _select_one_key_per_block(
        q, k, img_tok_idx, block_idx, selection, cfg, head_idx
    )

    B, H, _, _ = q.shape
    selected_tok_idx = img_tok_idx[selected_img_idx]
    if bool(cfg.get("include_special_keys", True)):
        spec_idx_per_head = spec_tok_idx[None, :].expand(H, -1)
        key_idx = torch.cat([spec_idx_per_head, selected_tok_idx], dim=1)
    else:
        key_idx = selected_tok_idx
    key_idx = key_idx[None, :, :].expand(B, -1, -1)

    q_spec = q[:, :, spec_tok_idx, :]
    out_spec = F.scaled_dot_product_attention(q_spec, k, v)

    q_img = q[:, :, img_tok_idx, :]
    k_sel, v_sel = _gather_kv_per_head(k, v, key_idx)
    out_img = F.scaled_dot_product_attention(q_img, k_sel, v_sel)

    out = torch.empty_like(v)
    out[:, :, spec_tok_idx, :] = out_spec
    out[:, :, img_tok_idx, :] = out_img
    return out


def _sample_query_indices(num_tokens, sample_count, device):
    sample_count = min(int(sample_count), num_tokens)
    if sample_count <= 0:
        raise ValueError(f"q_sample_count must be positive, got {sample_count}")

    if sample_count == num_tokens:
        return torch.arange(num_tokens, device=device, dtype=torch.long)

    return torch.linspace(
        0,
        num_tokens - 1,
        steps=sample_count,
        device=device,
    ).round().long()


def _estimate_image_key_attention_scores(q, k, img_tok_idx, q_sample_count):
    D = q.shape[-1]
    q_img = q[:, :, img_tok_idx, :]
    sample_idx = _sample_query_indices(img_tok_idx.numel(), q_sample_count, q.device)
    q_sample = q_img[:, :, sample_idx, :]

    scores = torch.matmul(q_sample.float(), k.transpose(-1, -2).float()) / math.sqrt(D)
    weights = torch.softmax(scores, dim=-1)
    return weights[:, :, :, img_tok_idx].mean(dim=2).squeeze(0)  # [H, num_img_tokens]


def _split_ranked_key_pool(key_scores, cfg):
    num_img_tokens = key_scores.shape[1]
    top_ratio = float(cfg.get("top_ratio", 0.1))
    final_ratio = float(cfg.get("final_ratio", 0.2))
    drop_bottom_ratio = float(cfg.get("drop_bottom_ratio", 0.5))
    pool_ratio = 1.0 - drop_bottom_ratio

    if not (0 < top_ratio <= final_ratio <= pool_ratio <= 1):
        raise ValueError(
            "Expected 0 < top_ratio <= final_ratio <= 1 - drop_bottom_ratio <= 1, "
            f"got top={top_ratio}, final={final_ratio}, drop_bottom={drop_bottom_ratio}"
        )

    top_count = max(1, min(num_img_tokens, int(round(num_img_tokens * top_ratio))))
    final_count = max(top_count, min(num_img_tokens, int(round(num_img_tokens * final_ratio))))
    pool_count = max(final_count, min(num_img_tokens, int(round(num_img_tokens * pool_ratio))))
    middle_select_count = min(final_count - top_count, pool_count - top_count)

    pool_scores, pool_idx = torch.topk(
        key_scores,
        k=pool_count,
        dim=1,
        largest=True,
        sorted=True,
    )
    top_idx = pool_idx[:, :top_count]
    mid_idx = pool_idx[:, top_count:]
    mid_scores = pool_scores[:, top_count:]
    return top_idx, mid_idx, mid_scores, middle_select_count


def _select_random_middle(mid_idx, select_count, cfg, head_idx):
    if select_count <= 0:
        return mid_idx[:, :0]

    seed = int(cfg.get("seed", 0))
    layer_idx = int(cfg.get("layer_idx", 0))
    seed = seed + 1000003 * layer_idx + 1009 * int(head_idx) + 17 * mid_idx.shape[1]
    generator = _make_generator(mid_idx.device, seed)
    scores = torch.rand(mid_idx.shape, device=mid_idx.device, generator=generator)
    choice = torch.topk(scores, k=select_count, dim=1, largest=True, sorted=False).indices
    return mid_idx.gather(1, choice)


def _spatial_eligible_middle_mask(top_idx, mid_idx, num_img_tokens, p_h, p_w, radius):
    if radius < 0:
        raise ValueError(f"spatial_radius must be non-negative, got {radius}")

    H, _ = top_idx.shape
    patches_per_img = p_h * p_w
    blocked = torch.zeros((H, num_img_tokens), dtype=torch.bool, device=top_idx.device)

    frame_idx = top_idx // patches_per_img
    patch_idx = top_idx % patches_per_img
    row = patch_idx // p_w
    col = patch_idx % p_w
    head_idx = torch.arange(H, device=top_idx.device)[:, None].expand_as(top_idx)

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr = row + dr
            nc = col + dc
            valid = (nr >= 0) & (nr < p_h) & (nc >= 0) & (nc < p_w)
            if not valid.any():
                continue
            neighbor_idx = frame_idx * patches_per_img + nr.clamp(0, p_h - 1) * p_w + nc.clamp(0, p_w - 1)
            blocked[head_idx[valid], neighbor_idx[valid]] = True

    return ~blocked.gather(1, mid_idx)


def _select_spatial_middle(
    mid_idx, mid_scores, top_idx, select_count, cfg, num_img_tokens, p_h, p_w,
    random_select, head_idx,
):
    if select_count <= 0:
        return mid_idx[:, :0]

    radius = int(cfg.get("spatial_radius", 2))
    eligible = _spatial_eligible_middle_mask(top_idx, mid_idx, num_img_tokens, p_h, p_w, radius)
    eligible_count = int(eligible.sum(dim=1).min().item())
    select_count = min(select_count, eligible_count)
    if select_count <= 0:
        return mid_idx[:, :0]

    if random_select:
        seed = int(cfg.get("seed", 0))
        layer_idx = int(cfg.get("layer_idx", 0))
        seed = seed + 1000003 * layer_idx + 1009 * int(head_idx) + 31 * radius + 17 * mid_idx.shape[1]
        generator = _make_generator(mid_idx.device, seed)
        scores = torch.rand(mid_idx.shape, device=mid_idx.device, generator=generator)
        scores = scores.masked_fill(~eligible, -1)
    else:
        scores = mid_scores.masked_fill(~eligible, -math.inf)

    choice = torch.topk(scores, k=select_count, dim=1, largest=True, sorted=False).indices
    return mid_idx.gather(1, choice)


def _make_spatiotemporal_neighbor_indices(center_idx, num_imgs, p_h, p_w, temporal_radius, spatial_radius):
    device = center_idx.device
    patches_per_img = p_h * p_w
    frame_idx = center_idx // patches_per_img
    patch_idx = center_idx % patches_per_img
    row = patch_idx // p_w
    col = patch_idx % p_w

    dt = torch.arange(-temporal_radius, temporal_radius + 1, device=device, dtype=torch.long)
    dr = torch.arange(-spatial_radius, spatial_radius + 1, device=device, dtype=torch.long)
    dc = torch.arange(-spatial_radius, spatial_radius + 1, device=device, dtype=torch.long)
    dt, dr, dc = torch.meshgrid(dt, dr, dc, indexing="ij")
    dt = dt.reshape(1, 1, -1)
    dr = dr.reshape(1, 1, -1)
    dc = dc.reshape(1, 1, -1)

    nf = frame_idx[:, :, None] + dt
    nr = row[:, :, None] + dr
    nc = col[:, :, None] + dc
    valid = (nf >= 0) & (nf < num_imgs) & (nr >= 0) & (nr < p_h) & (nc >= 0) & (nc < p_w)

    neighbor_idx = (
        nf.clamp(0, num_imgs - 1) * patches_per_img
        + nr.clamp(0, p_h - 1) * p_w
        + nc.clamp(0, p_w - 1)
    )
    return neighbor_idx, valid


def _query_conditioned_dissimilarity_scores(q, k, img_tok_idx, top_idx, mid_idx, cfg, p_h, p_w):
    device = q.device
    H = q.shape[1]
    D = q.shape[-1]
    num_img_tokens = img_tok_idx.numel()
    num_imgs = num_img_tokens // (p_h * p_w)

    q_img = q[:, :, img_tok_idx, :].squeeze(0).float()
    k_img = k[:, :, img_tok_idx, :].squeeze(0).float()

    c_q = torch.bmm(q_img.transpose(1, 2), q_img) / num_img_tokens
    k_ctx = torch.bmm(k_img, c_q)
    k_norm = (k_ctx * k_img).sum(dim=-1).clamp_min(1e-12).sqrt()

    top_mask = torch.zeros((H, num_img_tokens), dtype=torch.bool, device=device)
    top_mask.scatter_(1, top_idx, True)

    temporal_radius = int(cfg.get("temporal_radius", 4))
    spatial_radius = int(cfg.get("spatial_radius", 2))
    chunk_size = int(cfg.get("ctx_chunk_size", 256))
    if chunk_size <= 0:
        raise ValueError(f"ctx_chunk_size must be positive, got {chunk_size}")

    max_sims = []
    for start in range(0, mid_idx.shape[1], chunk_size):
        center_idx = mid_idx[:, start:start + chunk_size]
        neighbor_idx, valid = _make_spatiotemporal_neighbor_indices(
            center_idx, num_imgs, p_h, p_w, temporal_radius, spatial_radius
        )
        Hc, C, L = neighbor_idx.shape
        flat_neighbor_idx = neighbor_idx.reshape(Hc, C * L)
        top_neighbor = top_mask.gather(1, flat_neighbor_idx).reshape(Hc, C, L) & valid

        center_ctx = k_ctx.gather(1, center_idx[:, :, None].expand(-1, -1, D))
        center_norm = k_norm.gather(1, center_idx)
        neighbor_k = k_img.gather(
            1,
            flat_neighbor_idx[:, :, None].expand(-1, -1, D),
        ).reshape(Hc, C, L, D)
        neighbor_norm = k_norm.gather(1, flat_neighbor_idx).reshape(Hc, C, L)

        numerator = (center_ctx[:, :, None, :] * neighbor_k).sum(dim=-1)
        denominator = (center_norm[:, :, None] * neighbor_norm).clamp_min(1e-12)
        sim = numerator / denominator
        sim = sim.masked_fill(~top_neighbor, -2.0)
        max_sim = sim.max(dim=-1).values
        max_sim = torch.where(top_neighbor.any(dim=-1), max_sim, torch.full_like(max_sim, -2.0))
        max_sims.append(max_sim)

    return torch.cat(max_sims, dim=1)


def _select_ctx_dissimilar_middle(q, k, img_tok_idx, top_idx, mid_idx, select_count, cfg, p_h, p_w):
    if select_count <= 0:
        return mid_idx[:, :0]

    select_count = min(select_count, mid_idx.shape[1])
    max_sim = _query_conditioned_dissimilarity_scores(
        q, k, img_tok_idx, top_idx, mid_idx, cfg, p_h, p_w
    )
    choice = torch.topk(-max_sim, k=select_count, dim=1, largest=True, sorted=False).indices
    return mid_idx.gather(1, choice)


def ranked_middle_key_attention(q, k, v, spec_tok_idx, img_tok_idx, cfg, p_h, p_w, head_idx=0):
    """
    Keep top-attended image keys, drop the bottom half, then select more keys
    from the middle-attended pool according to the configured strategy.
    """
    assert q.shape[0] == 1

    device = q.device
    spec_tok_idx = spec_tok_idx.to(device=device, dtype=torch.long)
    img_tok_idx = img_tok_idx.to(device=device, dtype=torch.long)

    key_scores = _estimate_image_key_attention_scores(
        q, k, img_tok_idx, int(cfg.get("q_sample_count", 256))
    )
    top_idx, mid_idx, mid_scores, middle_select_count = _split_ranked_key_pool(key_scores, cfg)

    selection = cfg.get("selection", "middle_random")
    if selection == "middle_random":
        mid_selected = _select_random_middle(mid_idx, middle_select_count, cfg, head_idx)
    elif selection == "middle_attention":
        mid_selected = mid_idx[:, :middle_select_count]
    elif selection == "spatial_random":
        mid_selected = _select_spatial_middle(
            mid_idx, mid_scores, top_idx, middle_select_count, cfg, img_tok_idx.numel(), p_h, p_w,
            random_select=True, head_idx=head_idx,
        )
    elif selection == "spatial_attention":
        mid_selected = _select_spatial_middle(
            mid_idx, mid_scores, top_idx, middle_select_count, cfg, img_tok_idx.numel(), p_h, p_w,
            random_select=False, head_idx=head_idx,
        )
    elif selection == "ctx_dissimilar":
        mid_selected = _select_ctx_dissimilar_middle(
            q, k, img_tok_idx, top_idx, mid_idx, middle_select_count, cfg, p_h, p_w
        )
    else:
        raise ValueError(f"Unsupported ranked_middle selection: {selection}")

    selected_img_idx = torch.cat([top_idx, mid_selected], dim=1)
    selected_img_idx = torch.sort(selected_img_idx, dim=1).values

    B, H, _, _ = q.shape
    selected_tok_idx = img_tok_idx[selected_img_idx]
    if bool(cfg.get("include_special_keys", True)):
        spec_idx_per_head = spec_tok_idx[None, :].expand(H, -1)
        key_idx = torch.cat([spec_idx_per_head, selected_tok_idx], dim=1)
    else:
        key_idx = selected_tok_idx
    key_idx = key_idx[None, :, :].expand(B, -1, -1)

    q_spec = q[:, :, spec_tok_idx, :]
    out_spec = F.scaled_dot_product_attention(q_spec, k, v)

    q_img = q[:, :, img_tok_idx, :]
    k_sel, v_sel = _gather_kv_per_head(k, v, key_idx)
    out_img = F.scaled_dot_product_attention(q_img, k_sel, v_sel)

    out = torch.empty_like(v)
    out[:, :, spec_tok_idx, :] = out_spec
    out[:, :, img_tok_idx, :] = out_img
    return out


def _random_token_from_blocks(selected_blocks, block_idx, valid_mask, generator):
    if selected_blocks.numel() == 0:
        return selected_blocks

    device = selected_blocks.device
    block_area = block_idx.shape[1]
    token_idx = block_idx[selected_blocks]
    token_valid = valid_mask[selected_blocks]
    rand = torch.rand((*selected_blocks.shape, block_area), device=device, generator=generator)
    rand = rand.masked_fill(~token_valid, -1.0)
    choice = rand.argmax(dim=-1)
    return token_idx.gather(-1, choice[..., None]).squeeze(-1)


def _make_image_token_coords(num_imgs, p_h, p_w, cfg, device):
    patches_per_img = p_h * p_w
    patch_idx = torch.arange(patches_per_img, device=device, dtype=torch.float32)
    row = (patch_idx // p_w).repeat(num_imgs)
    col = (patch_idx % p_w).repeat(num_imgs)
    frame = torch.arange(num_imgs, device=device, dtype=torch.float32).repeat_interleave(patches_per_img)

    if bool(cfg.get("normalize_distance", True)):
        frame = frame / max(num_imgs - 1, 1)
        row = row / max(p_h - 1, 1)
        col = col / max(p_w - 1, 1)
    else:
        patch_size = float(cfg.get("patch_size", 14))
        row = (row + 0.5) * patch_size
        col = (col + 0.5) * patch_size

    temporal_weight = float(cfg.get("temporal_distance_weight", 1.0))
    spatial_weight = float(cfg.get("spatial_distance_weight", 1.0))
    return torch.stack(
        [frame * temporal_weight, row * spatial_weight, col * spatial_weight],
        dim=1,
    )


def _pool_block_coords(token_coords, block_idx, valid_mask):
    safe_idx = block_idx.clamp_min(0).reshape(-1)
    gathered = token_coords[safe_idx].reshape(block_idx.shape[0], block_idx.shape[1], 3)
    weights = valid_mask.to(dtype=token_coords.dtype, device=token_coords.device)[:, :, None]
    counts = valid_mask.sum(dim=1).clamp_min(1).to(dtype=token_coords.dtype, device=token_coords.device)
    return (gathered * weights).sum(dim=1) / counts[:, None]


def _min_distance_to_anchors(block_coords, anchor_coords, candidate_mask, block_chunk_size):
    H, Qb, A, _ = anchor_coords.shape
    num_blocks = block_coords.shape[0]
    scores = torch.empty((H, Qb, num_blocks), device=block_coords.device, dtype=torch.float32)
    anchor_sq = (anchor_coords * anchor_coords).sum(dim=-1)

    for start in range(0, num_blocks, block_chunk_size):
        end = min(start + block_chunk_size, num_blocks)
        coords = block_coords[start:end]
        coord_sq = (coords * coords).sum(dim=-1)
        dot = torch.einsum("bd,hqad->hqba", coords, anchor_coords)
        dist_sq = coord_sq[None, None, :, None] + anchor_sq[:, :, None, :] - 2.0 * dot
        dist = dist_sq.clamp_min(0.0).amin(dim=-1).sqrt()
        scores[:, :, start:end] = dist

    if candidate_mask.dim() == 1:
        mask = candidate_mask[None, None, :]
    else:
        mask = candidate_mask
    return scores.masked_fill(~mask, -math.inf)


def _coverage_expand_blocks(
    high_blocks,
    high_keys,
    block_coords,
    token_coords,
    num_target_blocks,
    cfg,
):
    H, Qb, high_count = high_blocks.shape
    num_blocks = block_coords.shape[0]
    extra_count = min(num_blocks, int(num_target_blocks)) - high_count
    if extra_count <= 0:
        return high_blocks

    remaining_mask = torch.ones((H, Qb, num_blocks), device=high_blocks.device, dtype=torch.bool)
    remaining_mask.scatter_(2, high_blocks, False)

    anchor_count = min(high_count, int(cfg.get("distance_anchor_count", 16)))
    if anchor_count <= 0:
        raise ValueError(f"distance_anchor_count must be positive, got {anchor_count}")

    if anchor_count == high_count:
        anchor_keys = high_keys
    else:
        anchor_pos = torch.linspace(
            0,
            high_count - 1,
            steps=anchor_count,
            device=high_blocks.device,
        ).round().long()
        anchor_keys = high_keys.gather(
            2,
            anchor_pos[None, None, :].expand(H, Qb, -1),
        )
    anchor_coords = token_coords[anchor_keys]

    block_chunk_size = int(cfg.get("distance_block_chunk_size", 4096))
    if block_chunk_size <= 0:
        raise ValueError(f"distance_block_chunk_size must be positive, got {block_chunk_size}")

    dist = _min_distance_to_anchors(
        block_coords,
        anchor_coords,
        remaining_mask,
        block_chunk_size,
    )
    extra_blocks = torch.topk(
        dist,
        k=extra_count,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    return torch.cat([high_blocks, extra_blocks], dim=-1)


def _prune_redundant_value_keys(selected_keys, v_img_norm, final_count):
    H, Qb, selected_count = selected_keys.shape
    final_count = min(int(final_count), selected_count)
    if final_count >= selected_count:
        return selected_keys

    value_dim = v_img_norm.shape[-1]
    values = v_img_norm[:, None, :, :].expand(-1, Qb, -1, -1)
    selected_values = torch.gather(
        values,
        dim=2,
        index=selected_keys[..., None].expand(H, Qb, selected_count, value_dim),
    )

    value_sum = selected_values.sum(dim=2, dtype=torch.float32)
    redundancy = (selected_values * value_sum[:, :, None, :]).sum(dim=-1) - 1.0
    redundancy = redundancy / max(selected_count - 1, 1)
    keep_idx = torch.topk(
        -redundancy,
        k=final_count,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    return selected_keys.gather(2, keep_idx)


def block_coverage_redundancy_attention(q, k, v, spec_tok_idx, img_tok_idx, cfg, p_h, p_w, head_idx=0):
    """
    Three-stage block sparse attention.

    Patch tokens are partitioned into 2x2 spatiotemporal blocks per frame.
    Each query block first keeps one random key from the top-attended blocks,
    expands coverage by adding far-away candidate blocks, then removes the
    most value-redundant selected keys by their mean pairwise cosine score.
    """
    assert q.shape[0] == 1

    keep_ratio = float(cfg.get("keep_ratio", 0.1))
    coverage_ratio = float(cfg.get("coverage_ratio", 0.2))
    top_block_ratio = float(cfg.get("top_block_ratio", 0.2))
    if not (0 < keep_ratio <= coverage_ratio <= 1):
        raise ValueError(
            f"Expected 0 < keep_ratio <= coverage_ratio <= 1, got "
            f"keep={keep_ratio}, coverage={coverage_ratio}"
        )
    if top_block_ratio <= 0 or top_block_ratio > 1:
        raise ValueError(f"top_block_ratio must be in (0, 1], got {top_block_ratio}")

    block_size = int(cfg.get("block_size", 2))
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    query_block_chunk_size = int(cfg.get("query_block_chunk_size", 8))
    if query_block_chunk_size <= 0:
        raise ValueError(f"query_block_chunk_size must be positive, got {query_block_chunk_size}")

    device = q.device
    spec_tok_idx = spec_tok_idx.to(device=device, dtype=torch.long)
    img_tok_idx = img_tok_idx.to(device=device, dtype=torch.long)

    B, H, N, D = q.shape
    patches_per_img = p_h * p_w
    num_img_tokens = img_tok_idx.numel()
    num_imgs = num_img_tokens // patches_per_img
    final_key_count = max(1, min(num_img_tokens, int(round(num_img_tokens * keep_ratio))))
    coverage_key_count = max(
        final_key_count,
        min(num_img_tokens, int(round(num_img_tokens * coverage_ratio))),
    )

    if final_key_count >= num_img_tokens and bool(cfg.get("include_special_keys", False)):
        return F.scaled_dot_product_attention(q, k, v)

    layer_idx = int(cfg.get("layer_idx", 0))
    seed = int(cfg.get("seed", 0))
    seed = (
        seed
        + 1000003 * layer_idx
        + 1009 * int(head_idx)
        + 37 * final_key_count
        + 17 * num_img_tokens
    )
    generator = _make_generator(device, seed)

    block_idx = _make_patch_blocks(num_imgs, p_h, p_w, block_size, device)
    valid_mask = block_idx >= 0
    safe_block_idx = block_idx.clamp_min(0)
    num_blocks = block_idx.shape[0]
    high_block_count = max(1, min(num_blocks, int(round(num_blocks * top_block_ratio))))
    coverage_block_count = max(high_block_count, min(num_blocks, coverage_key_count))
    final_block_count = min(coverage_block_count, final_key_count)

    q_img = q[:, :, img_tok_idx, :].squeeze(0)
    k_img = k[:, :, img_tok_idx, :].squeeze(0)
    q_blocks = _pool_image_blocks(q_img.float(), safe_block_idx, valid_mask)
    k_blocks = _pool_image_blocks(k_img.float(), safe_block_idx, valid_mask)

    token_coords = _make_image_token_coords(num_imgs, p_h, p_w, cfg, device)
    block_coords = _pool_block_coords(token_coords, safe_block_idx, valid_mask)
    v_img_norm = F.normalize(v[:, :, img_tok_idx, :].squeeze(0).float(), p=2, dim=-1, eps=1e-6)

    out = torch.empty_like(v)
    q_spec = q[:, :, spec_tok_idx, :]
    out[:, :, spec_tok_idx, :] = F.scaled_dot_product_attention(q_spec, k, v)

    k_full = k.squeeze(0).float()
    v_full = v.squeeze(0).float()
    k_full_t = k_full.transpose(-1, -2)
    scale = 1.0 / math.sqrt(D)
    include_special_keys = bool(cfg.get("include_special_keys", False))

    for start in range(0, num_blocks, query_block_chunk_size):
        end = min(start + query_block_chunk_size, num_blocks)
        q_chunk_blocks = end - start
        q_block = q_blocks[:, start:end, :]

        block_logits = torch.matmul(q_block, k_blocks.transpose(-1, -2)) * scale
        block_weights = torch.softmax(block_logits, dim=-1)
        high_blocks = torch.topk(
            block_weights,
            k=high_block_count,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        high_keys = _random_token_from_blocks(high_blocks, safe_block_idx, valid_mask, generator)

        coverage_blocks = _coverage_expand_blocks(
            high_blocks,
            high_keys,
            block_coords,
            token_coords,
            coverage_block_count,
            cfg,
        )
        coverage_keys = _random_token_from_blocks(coverage_blocks, safe_block_idx, valid_mask, generator)
        final_keys = _prune_redundant_value_keys(coverage_keys, v_img_norm, final_block_count)

        selected_img = torch.zeros(
            (H, q_chunk_blocks, num_img_tokens),
            device=device,
            dtype=torch.bool,
        )
        selected_img.scatter_(2, final_keys, True)

        query_tokens = safe_block_idx[start:end]
        query_valid = valid_mask[start:end]
        q_parent = torch.arange(q_chunk_blocks, device=device)[:, None].expand_as(query_tokens)
        q_parent = q_parent[query_valid]
        q_img_idx = query_tokens[query_valid]
        q_full_idx = img_tok_idx[q_img_idx]
        q_tok = q[:, :, q_full_idx, :].squeeze(0).float()

        scores = torch.matmul(q_tok, k_full_t) * scale
        key_mask = torch.zeros((H, q_tok.shape[1], N), dtype=torch.bool, device=device)
        if include_special_keys:
            key_mask[:, :, spec_tok_idx] = True
        key_mask[:, :, img_tok_idx] = selected_img[:, q_parent, :]
        scores.masked_fill_(~key_mask, -math.inf)
        attn = torch.softmax(scores, dim=-1)
        out_tok = torch.matmul(attn, v_full)
        out[:, :, q_full_idx, :] = out_tok.unsqueeze(0).to(dtype=v.dtype)

        del block_logits, block_weights, high_blocks, high_keys
        del coverage_blocks, coverage_keys, final_keys, selected_img
        del scores, key_mask, attn, out_tok

    return out


def block_novelty_key_attention(q, k, v, spec_tok_idx, img_tok_idx, cfg, p_h, p_w, head_idx=0):
    """
    Query-block-conditioned key selection.

    Image tokens are grouped into per-frame 2D patch blocks. For each query
    block, we score key blocks from pooled Q/K, keep random half tokens from the
    highest-attention blocks, then fill the key budget from blocks weighted by
    both coarse attention and novelty against the high-attention block set.
    """
    assert q.shape[0] == 1

    keep_ratio = float(cfg.get("keep_ratio", 0.2))
    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    block_size = int(cfg.get("block_size", 2))
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    top_block_ratio = float(cfg.get("top_block_ratio", 0.1))
    if top_block_ratio <= 0 or top_block_ratio > 1:
        raise ValueError(f"top_block_ratio must be in (0, 1], got {top_block_ratio}")

    high_block_token_ratio = float(cfg.get("high_block_token_ratio", 0.5))
    if high_block_token_ratio <= 0 or high_block_token_ratio > 1:
        raise ValueError(
            f"high_block_token_ratio must be in (0, 1], got {high_block_token_ratio}"
        )

    fill_tokens_per_block = int(cfg.get("fill_tokens_per_block", 2))
    if fill_tokens_per_block <= 0:
        raise ValueError(f"fill_tokens_per_block must be positive, got {fill_tokens_per_block}")

    query_block_chunk_size = int(cfg.get("query_block_chunk_size", 16))
    if query_block_chunk_size <= 0:
        raise ValueError(
            f"query_block_chunk_size must be positive, got {query_block_chunk_size}"
        )

    attn_weight = float(cfg.get("attn_weight", 0.5))
    novelty_weight = float(cfg.get("novelty_weight", 0.5))
    if attn_weight < 0 or novelty_weight < 0 or attn_weight + novelty_weight <= 0:
        raise ValueError(
            f"Invalid score weights: attn_weight={attn_weight}, novelty_weight={novelty_weight}"
        )

    device = q.device
    spec_tok_idx = spec_tok_idx.to(device=device, dtype=torch.long)
    img_tok_idx = img_tok_idx.to(device=device, dtype=torch.long)

    B, H, N, D = q.shape
    _, _, _, Dv = v.shape
    patches_per_img = p_h * p_w
    num_img_tokens = img_tok_idx.numel()
    num_imgs = num_img_tokens // patches_per_img
    spec_count = spec_tok_idx.numel()
    mandatory_total = spec_count + patches_per_img
    target_total = max(mandatory_total, min(N, int(round(N * keep_ratio))))
    image_target_count = min(num_img_tokens, max(patches_per_img, target_total - spec_count))
    optional_target_count = image_target_count - patches_per_img

    if target_total >= N:
        return F.scaled_dot_product_attention(q, k, v)

    layer_idx = int(cfg.get("layer_idx", 0))
    seed = int(cfg.get("seed", 0))
    seed = seed + 1000003 * layer_idx + 7919 * N + 37 * int(round(keep_ratio * 1000))
    generator = _make_generator(device, seed)

    block_idx = _make_patch_blocks(num_imgs, p_h, p_w, block_size, device)
    valid_mask = block_idx >= 0
    safe_block_idx = block_idx.clamp_min(0)
    block_counts = valid_mask.sum(dim=1)
    num_blocks, block_area = block_idx.shape
    blocks_per_img = math.ceil(p_h / block_size) * math.ceil(p_w / block_size)
    candidate_block_mask = torch.ones(num_blocks, dtype=torch.bool, device=device)
    candidate_block_mask[:blocks_per_img] = False
    candidate_block_count = int(candidate_block_mask.sum().item())

    if candidate_block_count <= 0 or optional_target_count <= 0:
        key_idx = torch.cat([spec_tok_idx, img_tok_idx[:patches_per_img]], dim=0)
        k_sel = k[:, :, key_idx, :]
        v_sel = v[:, :, key_idx, :]

        q_spec = q[:, :, spec_tok_idx, :]
        out_spec = F.scaled_dot_product_attention(q_spec, k, v)
        q_img = q[:, :, img_tok_idx, :]
        out_img = F.scaled_dot_product_attention(q_img, k_sel, v_sel)

        out = torch.empty_like(v)
        out[:, :, spec_tok_idx, :] = out_spec
        out[:, :, img_tok_idx, :] = out_img
        return out

    q_img = q[:, :, img_tok_idx, :].squeeze(0)
    k_img = k[:, :, img_tok_idx, :].squeeze(0)
    q_blocks = _pool_image_blocks(q_img.float(), safe_block_idx, valid_mask)
    k_blocks = _pool_image_blocks(k_img.float(), safe_block_idx, valid_mask)

    k_norm = F.normalize(k_img.float(), p=2, dim=-1)
    k_norm_blocks = _pool_image_blocks(k_norm, safe_block_idx, valid_mask)
    block_counts_f = block_counts.to(device=device, dtype=torch.float32)

    out = torch.empty_like(v)
    q_spec = q[:, :, spec_tok_idx, :]
    out[:, :, spec_tok_idx, :] = F.scaled_dot_product_attention(q_spec, k, v)

    k_full = k.squeeze(0).float()
    v_full = v.squeeze(0).float()
    k_full_t = k_full.transpose(-1, -2)
    scale = 1.0 / math.sqrt(D)

    high_block_count = max(1, min(candidate_block_count, int(round(candidate_block_count * top_block_ratio))))
    max_extra_blocks = min(candidate_block_count, int(math.ceil(optional_target_count / 2.0)) + 16)
    first_frame_img_mask = torch.zeros(num_img_tokens, dtype=torch.bool, device=device)
    first_frame_img_mask[:patches_per_img] = True

    for start in range(0, num_blocks, query_block_chunk_size):
        end = min(start + query_block_chunk_size, num_blocks)
        q_block = q_blocks[:, start:end, :]
        q_chunk_blocks = end - start

        block_logits = torch.matmul(q_block, k_blocks.transpose(-1, -2)) * scale
        candidate_mask = candidate_block_mask[None, None, :]
        block_weights = torch.softmax(
            block_logits.masked_fill(~candidate_mask, -math.inf),
            dim=-1,
        )

        high_blocks = torch.topk(
            block_weights,
            k=high_block_count,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices

        token_scores = torch.full(
            (H, q_chunk_blocks, num_img_tokens),
            -math.inf,
            dtype=torch.float32,
            device=device,
        )

        high_counts = block_counts[high_blocks]
        high_keep = torch.ceil(high_counts.float() * high_block_token_ratio).long().clamp_min(1)
        high_base = torch.full(
            high_blocks.shape,
            2.0,
            dtype=torch.float32,
            device=device,
        )
        _scatter_random_block_tokens(
            token_scores,
            high_blocks,
            safe_block_idx,
            valid_mask,
            high_keep,
            generator,
            high_base,
        )

        remaining_mask = candidate_block_mask[None, None, :].expand(H, q_chunk_blocks, -1).clone()
        remaining_mask.scatter_(2, high_blocks, False)

        flat_high_blocks = high_blocks.reshape(H, -1)
        high_vecs = k_norm_blocks.gather(
            1,
            flat_high_blocks[:, :, None].expand(-1, -1, D),
        ).reshape(H, q_chunk_blocks, high_block_count, D)
        high_weights = block_counts_f[high_blocks][..., None]
        high_mean = (high_vecs * high_weights).sum(dim=2)
        high_mean = high_mean / high_weights.sum(dim=2).clamp_min(1.0)
        redundancy = torch.matmul(
            high_mean[:, :, None, :],
            k_norm_blocks.transpose(-1, -2)[:, None, :, :],
        ).squeeze(2)
        novelty = 1.0 - redundancy

        attn_norm = _masked_minmax(block_weights, remaining_mask)
        novelty_norm = _masked_minmax(novelty, remaining_mask)
        block_score = (
            attn_weight * attn_norm + novelty_weight * novelty_norm
        ) / (attn_weight + novelty_weight)
        block_score = block_score.masked_fill(~remaining_mask, -math.inf)

        extra_blocks = torch.topk(
            block_score,
            k=max_extra_blocks,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        extra_base = block_score.gather(2, extra_blocks)
        extra_valid = torch.isfinite(extra_base)
        extra_counts = block_counts[extra_blocks]
        extra_keep = torch.minimum(
            torch.full_like(extra_counts, fill_tokens_per_block),
            extra_counts,
        )
        extra_keep = torch.where(extra_valid, extra_keep, torch.zeros_like(extra_keep))
        _scatter_random_block_tokens(
            token_scores,
            extra_blocks,
            safe_block_idx,
            valid_mask,
            extra_keep,
            generator,
            extra_base.clamp_min(0.0),
        )

        valid_optional = torch.isfinite(token_scores).sum(dim=-1)
        if int(valid_optional.min().item()) < optional_target_count:
            fill_mask = (~first_frame_img_mask)[None, None, :]
            fill_scores = -1.0 + torch.rand(
                token_scores.shape,
                device=device,
                generator=generator,
                dtype=token_scores.dtype,
            ) * 1e-3
            token_scores = torch.where(
                (~torch.isfinite(token_scores)) & fill_mask,
                fill_scores,
                token_scores,
            )

        selected_img = first_frame_img_mask[None, None, :].expand(H, q_chunk_blocks, -1).clone()
        optional_idx = torch.topk(
            token_scores,
            k=optional_target_count,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        selected_img.scatter_(2, optional_idx, True)

        query_tokens = safe_block_idx[start:end]
        query_valid = valid_mask[start:end]
        q_parent = torch.arange(q_chunk_blocks, device=device)[:, None].expand_as(query_tokens)
        q_parent = q_parent[query_valid]
        q_img_idx = query_tokens[query_valid]
        q_full_idx = img_tok_idx[q_img_idx]
        q_tok = q[:, :, q_full_idx, :].squeeze(0).float()

        scores = torch.matmul(q_tok, k_full_t) * scale
        key_mask = torch.zeros((H, q_tok.shape[1], N), dtype=torch.bool, device=device)
        key_mask[:, :, spec_tok_idx] = True
        key_mask[:, :, img_tok_idx] = selected_img[:, q_parent, :]
        scores.masked_fill_(~key_mask, -math.inf)
        attn = torch.softmax(scores, dim=-1)
        out_tok = torch.matmul(attn, v_full)
        out[:, :, q_full_idx, :] = out_tok.unsqueeze(0).to(dtype=v.dtype)

        del block_logits, block_weights, token_scores, selected_img, scores, key_mask, attn

    return out


def _zscore_last_dim(x, eps=1e-6):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def _anchor_special_union_count(num_toks, num_toks_per_img, num_imgs, num_special_tokens):
    indices = set(range(num_toks_per_img))
    for img_idx in range(num_imgs):
        base = img_idx * num_toks_per_img
        indices.update(range(base, min(base + num_special_tokens, num_toks)))
    return len(indices)


def _stride_anchor_special_union_count(
    num_toks,
    num_toks_per_img,
    num_imgs,
    num_special_tokens,
    stride,
    start_pos,
):
    indices = set(range(start_pos, num_toks, stride))
    indices.update(range(num_toks_per_img))
    for img_idx in range(num_imgs):
        base = img_idx * num_toks_per_img
        indices.update(range(base, min(base + num_special_tokens, num_toks)))
    return len(indices)


def estimate_saf3r_pairs_from_cfg(
    cfg,
    num_toks,
    num_toks_per_img,
    num_imgs,
    num_special_toks,
    num_img_toks,
    num_special_tokens,
    head_idx=0,
    est_topk_idx=None,
):
    mode = cfg.get("mode", "full")
    if mode == "full":
        return num_toks * num_toks
    if mode == "skip":
        return 0
    if mode == "broadcast_first":
        return num_toks_per_img * num_toks_per_img
    if mode in {"all_to_first", "global_to_frame"}:
        return num_toks * num_toks_per_img
    if mode == "q_probe_topk":
        if "stride" in cfg:
            key_count = max(1, num_toks // int(cfg["stride"]))
        else:
            key_count = min(int(cfg.get("topk", 1024)), num_toks)
        if cfg.get("include_as", False):
            key_count = min(
                num_toks,
                key_count
                + _anchor_special_union_count(
                    num_toks, num_toks_per_img, num_imgs, num_special_tokens
                ),
            )
        return num_toks * key_count
    if mode == "dino_topk":
        if est_topk_idx is not None and est_topk_idx.numel() > 0:
            topk = int(est_topk_idx.shape[-1])
        else:
            topk = int(cfg.get("effective_topk", cfg.get("topk", num_img_toks)))
        topk = min(topk, num_img_toks)
        sp2sp_stride = int(cfg.get("dino_sp2sp_stride", 1))
        special_key_count = (num_toks + sp2sp_stride - 1) // sp2sp_stride
        return num_special_toks * special_key_count + num_img_toks * topk
    if mode == "stride":
        stride = int(cfg.get("stride", 4))
        start_pos = int(head_idx) % stride if cfg.get("shift", False) else 0
        key_count = (num_toks - start_pos + stride - 1) // stride
        if cfg.get("include_as", True):
            key_count = _stride_anchor_special_union_count(
                num_toks,
                num_toks_per_img,
                num_imgs,
                num_special_tokens,
                stride,
                start_pos,
            )
        return num_toks * key_count
    if mode in {"fixed_keys", "query_topk_keys", "random_keys"}:
        keep_ratio = float(cfg.get("keep_ratio", 1.0))
        key_count = max(1, min(num_toks, int(round(num_toks * keep_ratio))))
        return num_toks * key_count
    if mode == "block2x2_keys":
        if cfg.get("selection", "all") == "all":
            return num_toks * num_toks
        block_size = int(cfg.get("block_size", 2))
        blocks_per_img = ((num_toks_per_img - num_special_tokens + block_size - 1) // block_size)
        key_count = num_imgs * blocks_per_img * blocks_per_img
        if cfg.get("include_special_keys", True):
            key_count += num_special_toks
        return num_special_toks * num_toks + num_img_toks * key_count
    return num_toks * num_toks


def _block3x3_refinement_budget(
    cfg,
    num_toks,
    num_toks_per_img,
    num_imgs,
    num_special_toks,
    num_img_toks,
    num_special_tokens,
    num_blocks,
    refinable_count,
    candidate_count,
    head_idx,
    est_topk_idx,
):
    include_special_keys = bool(cfg.get("include_special_keys", True))
    budget_mode = cfg.get("budget_mode", "saf3r")

    if budget_mode == "saf3r" and cfg.get("saf3r_budget_cfg", None) is not None:
        target_pairs = estimate_saf3r_pairs_from_cfg(
            cfg["saf3r_budget_cfg"],
            num_toks,
            num_toks_per_img,
            num_imgs,
            num_special_toks,
            num_img_toks,
            num_special_tokens,
            head_idx=head_idx,
            est_topk_idx=est_topk_idx,
        )
        img_query_pairs = max(0, target_pairs - num_special_toks * num_toks)
        target_keys = int(round(img_query_pairs / max(num_img_toks, 1)))
        image_target_count = target_keys - num_special_toks if include_special_keys else target_keys
    else:
        keep_ratio = float(cfg.get("keep_ratio", 1.0))
        target_keys = max(1, min(num_toks, int(round(num_toks * keep_ratio))))
        image_target_count = target_keys - num_special_toks if include_special_keys else target_keys

    image_target_count = max(num_blocks, image_target_count)
    image_target_count = min(num_img_toks, num_blocks + refinable_count, image_target_count)
    saf3r_refine_count = max(0, image_target_count - num_blocks)

    configured_count = None
    if cfg.get("refinement_blocks", None) is not None:
        configured_count = int(cfg["refinement_blocks"])
    elif cfg.get("refinement_ratio", None) is not None:
        configured_count = int(round(num_blocks * float(cfg["refinement_ratio"])))

    if configured_count is None:
        refine_count = saf3r_refine_count
    else:
        refine_count = max(0, configured_count)
        if bool(cfg.get("enforce_saf3r_budget", True)):
            refine_count = min(refine_count, saf3r_refine_count)

    refine_count = max(0, min(refine_count, refinable_count, candidate_count))
    return refine_count


def _block3x3_target_image_count(
    cfg,
    num_toks,
    num_toks_per_img,
    num_imgs,
    num_special_toks,
    num_img_toks,
    num_special_tokens,
    head_idx,
    est_topk_idx,
):
    include_special_keys = bool(cfg.get("include_special_keys", True))
    budget_mode = cfg.get("budget_mode", "saf3r")

    if budget_mode == "saf3r" and cfg.get("saf3r_budget_cfg", None) is not None:
        target_pairs = estimate_saf3r_pairs_from_cfg(
            cfg["saf3r_budget_cfg"],
            num_toks,
            num_toks_per_img,
            num_imgs,
            num_special_toks,
            num_img_toks,
            num_special_tokens,
            head_idx=head_idx,
            est_topk_idx=est_topk_idx,
        )
        img_query_pairs = max(0, target_pairs - num_special_toks * num_toks)
        target_keys = int(round(img_query_pairs / max(num_img_toks, 1)))
        image_target_count = target_keys - num_special_toks if include_special_keys else target_keys
    else:
        keep_ratio = float(cfg.get("keep_ratio", 1.0))
        target_keys = max(1, min(num_toks, int(round(num_toks * keep_ratio))))
        image_target_count = target_keys - num_special_toks if include_special_keys else target_keys

    return max(1, min(num_img_toks, image_target_count))


def _configured_refine_count(cfg, reference_count):
    if cfg.get("refinement_blocks", None) is not None:
        return max(0, int(cfg["refinement_blocks"]))
    if cfg.get("refinement_ratio", None) is not None:
        return max(0, int(round(reference_count * float(cfg["refinement_ratio"]))))
    return None


def _uniform_block_indices(num_blocks, count, device):
    count = max(1, min(int(count), num_blocks))
    if count == num_blocks:
        return torch.arange(num_blocks, device=device, dtype=torch.long)
    return torch.linspace(0, num_blocks - 1, steps=count, device=device).round().long()


def _estimate_refinable_block_mask(num_imgs, p_h, p_w, block_size):
    mask = []
    for _ in range(num_imgs):
        for row in range(0, p_h, block_size):
            row_end = min(row + block_size, p_h)
            for col in range(0, p_w, block_size):
                col_end = min(col + block_size, p_w)
                mask.append((row_end - row) * (col_end - col) >= 2)
    return torch.tensor(mask, dtype=torch.bool)


def _tighten_budgeted_counts_for_candidates(
    base_count,
    refine_count,
    target_image_count,
    num_blocks,
    refinable_mask,
    candidate_ratio,
    device,
):
    base_blocks = _uniform_block_indices(num_blocks, base_count, device)
    while True:
        base_refinable_count = int(refinable_mask[base_blocks].sum().item())
        candidate_count = (
            0
            if base_refinable_count == 0
            else max(1, min(base_refinable_count, int(round(base_refinable_count * candidate_ratio))))
        )
        if refine_count <= candidate_count:
            return base_count, refine_count, base_blocks, candidate_count
        refine_count = candidate_count
        base_count = min(target_image_count - refine_count, num_blocks)
        base_blocks = _uniform_block_indices(num_blocks, base_count, device)


def _block3x3_budgeted_counts(
    cfg,
    num_toks,
    num_toks_per_img,
    num_imgs,
    num_special_toks,
    num_img_toks,
    num_special_tokens,
    num_blocks,
    refinable_count,
    candidate_count,
    head_idx,
    est_topk_idx,
):
    target_count = _block3x3_target_image_count(
        cfg,
        num_toks,
        num_toks_per_img,
        num_imgs,
        num_special_toks,
        num_img_toks,
        num_special_tokens,
        head_idx,
        est_topk_idx,
    )
    target_count = min(target_count, num_blocks + refinable_count)

    allocation_scope = cfg.get("allocation_scope", "budgeted_uniform")
    if allocation_scope == "all_blocks":
        target_count = max(num_blocks, target_count)
        refine_count = _block3x3_refinement_budget(
            cfg,
            num_toks,
            num_toks_per_img,
            num_imgs,
            num_special_toks,
            num_img_toks,
            num_special_tokens,
            num_blocks,
            refinable_count,
            candidate_count,
            head_idx,
            est_topk_idx,
        )
        return num_blocks, refine_count
    if allocation_scope != "budgeted_uniform":
        raise ValueError(f"Unsupported block3x3 allocation_scope: {allocation_scope}")

    configured_count = _configured_refine_count(cfg, target_count)
    if configured_count is None:
        refine_count = max(0, target_count - num_blocks)
    else:
        refine_count = configured_count
    if cfg.get("refinement_strategy", "violation") == "none":
        refine_count = 0

    min_refine_count = 0 if cfg.get("refinement_strategy", "violation") == "none" else max(0, target_count - num_blocks)
    refine_count = max(refine_count, min_refine_count)
    refine_count = min(refine_count, target_count // 2, refinable_count, candidate_count)
    base_count = target_count - refine_count
    base_count = min(base_count, num_blocks)
    return base_count, refine_count


def estimate_block3x3_violation_kept_pairs(
    cfg,
    num_toks,
    num_toks_per_img,
    num_imgs,
    num_special_toks,
    num_img_toks,
    num_special_tokens,
    p_h=None,
    p_w=None,
    head_idx=0,
    est_topk_idx=None,
):
    block_size = int(cfg.get("block_size", 3))
    if p_h is None or p_w is None:
        p = num_toks_per_img - num_special_tokens
        p_w = int(math.sqrt(p))
        p_h = p_w
        if p_h * p_w != p:
            p_h = p
            p_w = 1

    blocks_per_img = ((p_h + block_size - 1) // block_size) * (
        (p_w + block_size - 1) // block_size
    )
    num_blocks = num_imgs * blocks_per_img

    refinable_mask = _estimate_refinable_block_mask(num_imgs, p_h, p_w, block_size)
    refinable_count = int(refinable_mask.sum().item())

    candidate_ratio = float(cfg.get("candidate_ratio", 0.3))
    candidate_count = (
        0
        if refinable_count == 0
        else max(1, min(refinable_count, int(round(num_blocks * candidate_ratio))))
    )
    base_count, refine_count = _block3x3_budgeted_counts(
        cfg,
        num_toks,
        num_toks_per_img,
        num_imgs,
        num_special_toks,
        num_img_toks,
        num_special_tokens,
        num_blocks,
        refinable_count,
        candidate_count,
        head_idx,
        est_topk_idx,
    )
    if cfg.get("allocation_scope", "budgeted_uniform") == "budgeted_uniform":
        base_count, refine_count, _, _ = _tighten_budgeted_counts_for_candidates(
            base_count,
            refine_count,
            base_count + refine_count,
            num_blocks,
            refinable_mask,
            candidate_ratio,
            torch.device("cpu"),
        )

    key_count = base_count + refine_count
    if bool(cfg.get("include_special_keys", True)):
        key_count += num_special_toks
    return num_special_toks * num_toks + num_img_toks * key_count


def _center_block_attention(q, k, v, spec_tok_idx, img_tok_idx, center_img_idx, include_special_keys):
    B, H, _, _ = q.shape
    device = q.device
    center_tok_idx = img_tok_idx[center_img_idx]
    center_tok_idx = center_tok_idx[None, :].expand(H, -1)
    if include_special_keys:
        spec_idx_per_head = spec_tok_idx[None, :].expand(H, -1)
        key_idx = torch.cat([spec_idx_per_head, center_tok_idx], dim=1)
    else:
        key_idx = center_tok_idx
    key_idx = key_idx[None, :, :].expand(B, -1, -1)

    out = torch.empty_like(v)
    if spec_tok_idx.numel() > 0:
        q_spec = q[:, :, spec_tok_idx, :]
        out[:, :, spec_tok_idx, :] = F.scaled_dot_product_attention(q_spec, k, v)

    q_img = q[:, :, img_tok_idx, :]
    k_sel, v_sel = _gather_kv_per_head(k, v, key_idx)
    out[:, :, img_tok_idx, :] = F.scaled_dot_product_attention(q_img, k_sel, v_sel)
    return out


def _gather_query_kv(source, key_idx):
    H, Q, Ksel = key_idx.shape
    dim = source.shape[-1]
    expanded = source[:, None, :, :].expand(H, Q, -1, dim)
    gather_idx = key_idx[..., None].expand(H, Q, Ksel, dim)
    return torch.gather(expanded, dim=2, index=gather_idx)


def _hier_add_tree_node(
    frame_id,
    r0,
    r1,
    c0,
    c1,
    parent,
    p_w,
    patches_per_img,
    frames,
    parents,
    lefts,
    rights,
    sizes,
    reps,
    regions,
):
    node_id = len(frames)
    center_r = (r0 + r1 - 1) // 2
    center_c = (c0 + c1 - 1) // 2
    rep = frame_id * patches_per_img + center_r * p_w + center_c

    frames.append(frame_id)
    parents.append(parent)
    lefts.append(-1)
    rights.append(-1)
    sizes.append((r1 - r0) * (c1 - c0))
    reps.append(rep)
    regions.append((frame_id, r0, r1, c0, c1))

    if sizes[-1] > 1:
        height = r1 - r0
        width = c1 - c0
        if height >= width and height > 1:
            mid = (r0 + r1) // 2
            left_child = _hier_add_tree_node(
                frame_id, r0, mid, c0, c1, node_id, p_w, patches_per_img,
                frames, parents, lefts, rights, sizes, reps, regions,
            )
            right_child = _hier_add_tree_node(
                frame_id, mid, r1, c0, c1, node_id, p_w, patches_per_img,
                frames, parents, lefts, rights, sizes, reps, regions,
            )
        else:
            mid = (c0 + c1) // 2
            left_child = _hier_add_tree_node(
                frame_id, r0, r1, c0, mid, node_id, p_w, patches_per_img,
                frames, parents, lefts, rights, sizes, reps, regions,
            )
            right_child = _hier_add_tree_node(
                frame_id, r0, r1, mid, c1, node_id, p_w, patches_per_img,
                frames, parents, lefts, rights, sizes, reps, regions,
            )
        lefts[node_id] = left_child
        rights[node_id] = right_child

    return node_id


def _build_hierarchical_patch_tree(num_imgs, p_h, p_w, device):
    patches_per_img = p_h * p_w
    frames = []
    parents = []
    lefts = []
    rights = []
    sizes = []
    reps = []
    regions = []
    roots = []

    for frame_id in range(num_imgs):
        roots.append(
            _hier_add_tree_node(
                frame_id, 0, p_h, 0, p_w, -1, p_w, patches_per_img,
                frames, parents, lefts, rights, sizes, reps, regions,
            )
        )

    return {
        "roots": torch.tensor(roots, device=device, dtype=torch.long),
        "frame": torch.tensor(frames, device=device, dtype=torch.long),
        "parent": torch.tensor(parents, device=device, dtype=torch.long),
        "left": torch.tensor(lefts, device=device, dtype=torch.long),
        "right": torch.tensor(rights, device=device, dtype=torch.long),
        "size": torch.tensor(sizes, device=device, dtype=torch.long),
        "rep": torch.tensor(reps, device=device, dtype=torch.long),
        "region": torch.tensor(regions, device=device, dtype=torch.long),
    }


def _hier_patch_budget(cfg, num_imgs, num_img_toks):
    if cfg.get("patch_budget", None) is not None:
        budget = int(cfg["patch_budget"])
    else:
        keep_ratio = float(cfg.get("keep_ratio", 0.1))
        budget = int(round(num_img_toks * keep_ratio))
    if budget <= 0:
        raise ValueError(f"patch_budget must be positive, got {budget}")
    if budget < num_imgs:
        if bool(cfg.get("strict_patch_budget", False)):
            raise ValueError(
                f"patch_budget={budget} is smaller than num_frames={num_imgs}; "
                "root-per-frame warm start would violate the fixed-budget setup."
            )
        budget = num_imgs
    return max(1, min(num_img_toks, budget))


def _hier_uniform_warm_start(tree, patch_budget, num_imgs):
    roots = [int(x) for x in tree["roots"].detach().cpu().tolist()]
    left = tree["left"].detach().cpu().tolist()
    size = tree["size"].detach().cpu().tolist()
    right = tree["right"].detach().cpu().tolist()

    per_frame = [[root] for root in roots]
    active_count = num_imgs
    while active_count < patch_budget:
        changed = False
        for frame_id in range(num_imgs):
            if active_count >= patch_budget:
                break
            nodes = per_frame[frame_id]
            split_pos = None
            split_key = None
            for pos, node_id in enumerate(nodes):
                if left[node_id] < 0:
                    continue
                key = (size[node_id], -pos)
                if split_key is None or key > split_key:
                    split_pos = pos
                    split_key = key
            if split_pos is None:
                continue
            node_id = nodes.pop(split_pos)
            nodes.extend([left[node_id], right[node_id]])
            active_count += 1
            changed = True
        if not changed:
            break

    init_nodes = [node for nodes in per_frame for node in nodes]
    return torch.tensor(init_nodes[:patch_budget], device=tree["roots"].device, dtype=torch.long)


def estimate_hier_split_prune_kept_pairs(
    cfg,
    num_toks,
    num_toks_per_img,
    num_imgs,
    num_special_toks,
    num_img_toks,
):
    patch_budget = _hier_patch_budget(cfg, num_imgs, num_img_toks)
    key_count = patch_budget
    if bool(cfg.get("include_special_keys", True)):
        key_count += num_special_toks
    return num_special_toks * num_toks + num_img_toks * key_count


def _hier_attention_from_nodes(
    q_tok,
    k_heads,
    v_heads,
    spec_tok_idx,
    img_tok_idx,
    node_ids,
    node_rep,
    node_log_size,
    scale,
    use_multiplicity_correction,
    include_special_keys,
):
    rep_idx = node_rep[node_ids]
    key_idx = img_tok_idx[rep_idx]
    k_sel = _gather_query_kv(k_heads, key_idx)
    v_sel = _gather_query_kv(v_heads, key_idx)
    scores = (q_tok[:, :, None, :] * k_sel).sum(dim=-1) * scale
    if use_multiplicity_correction:
        scores = scores + node_log_size[node_ids].to(dtype=scores.dtype)

    if include_special_keys and spec_tok_idx.numel() > 0:
        H, Q, _ = node_ids.shape
        spec_idx = spec_tok_idx[None, None, :].expand(H, Q, -1)
        k_spec = _gather_query_kv(k_heads, spec_idx)
        v_spec = _gather_query_kv(v_heads, spec_idx)
        spec_scores = (q_tok[:, :, None, :] * k_spec).sum(dim=-1) * scale
        scores = torch.cat([spec_scores, scores], dim=-1)
        v_sel = torch.cat([v_spec, v_sel], dim=-2)

    attn = torch.softmax(scores, dim=-1)
    return (attn[..., None] * v_sel).sum(dim=-2)


def _hier_node_logits_values(
    q_tok,
    k_heads,
    v_heads,
    img_tok_idx,
    node_ids,
    node_rep,
    node_log_size,
    scale,
    use_multiplicity_correction,
):
    safe_nodes = node_ids.clamp_min(0)
    rep_idx = node_rep[safe_nodes]
    key_idx = img_tok_idx[rep_idx]
    k_sel = _gather_query_kv(k_heads, key_idx)
    v_sel = _gather_query_kv(v_heads, key_idx)
    logits = (q_tok[:, :, None, :] * k_sel).sum(dim=-1) * scale
    if use_multiplicity_correction:
        logits = logits + node_log_size[safe_nodes].to(dtype=logits.dtype)
    return logits, v_sel


def _hier_relative_l2(candidate_out, current_out, eps):
    return (candidate_out - current_out).norm(dim=-1) / current_out.norm(dim=-1).clamp_min(eps)


def _hier_active_positions(frontier_active, patch_budget):
    scores = frontier_active.to(dtype=torch.float32)
    pos = torch.topk(scores, k=patch_budget, dim=-1, largest=True, sorted=False).indices
    return pos


def _hier_gather_active_nodes(frontier_nodes, frontier_active, patch_budget):
    active_pos = _hier_active_positions(frontier_active, patch_budget)
    active_nodes = frontier_nodes.gather(2, active_pos)
    return active_nodes, active_pos


def _hier_trial_remove(nodes, remove_pos):
    return torch.cat([nodes[:, :, :remove_pos], nodes[:, :, remove_pos + 1:]], dim=2)


def _hier_select_random_index(valid_mask, generator):
    scores = torch.rand(valid_mask.shape, device=valid_mask.device, generator=generator)
    scores = scores.masked_fill(~valid_mask, -math.inf)
    values, idx = scores.max(dim=-1)
    return idx, torch.isfinite(values)


def _hier_compute_scores(
    q_tok,
    k_heads,
    v_heads,
    spec_tok_idx,
    img_tok_idx,
    active_nodes,
    frontier_nodes,
    frontier_active,
    node_rep,
    node_log_size,
    node_left,
    node_right,
    scale,
    use_multiplicity_correction,
    include_special_keys,
    eps,
    still_refining,
):
    H, Q, patch_budget = active_nodes.shape
    patch_logits, patch_values = _hier_node_logits_values(
        q_tok,
        k_heads,
        v_heads,
        img_tok_idx,
        active_nodes,
        node_rep,
        node_log_size,
        scale,
        use_multiplicity_correction,
    )

    logits_for_softmax = patch_logits
    values_for_output = patch_values
    if include_special_keys and spec_tok_idx.numel() > 0:
        spec_idx = spec_tok_idx[None, None, :].expand(H, Q, -1)
        k_spec = _gather_query_kv(k_heads, spec_idx)
        v_spec = _gather_query_kv(v_heads, spec_idx)
        spec_logits = (q_tok[:, :, None, :] * k_spec).sum(dim=-1) * scale
        logits_for_softmax = torch.cat([spec_logits, patch_logits], dim=-1)
        values_for_output = torch.cat([v_spec, patch_values], dim=-2)

    weights = torch.softmax(logits_for_softmax, dim=-1)
    current_out = (weights[..., None] * values_for_output).sum(dim=-2)
    log_z = torch.logsumexp(logits_for_softmax, dim=-1)
    patch_weights = weights[:, :, -patch_budget:]

    denom_prune = (1.0 - patch_weights).clamp_min(eps)
    prune_out = (
        current_out[:, :, None, :]
        - patch_weights[..., None] * patch_values
    ) / denom_prune[..., None]
    prune_cost = _hier_relative_l2(prune_out, current_out[:, :, None, :], eps)
    if patch_budget <= 1:
        prune_cost.fill_(math.inf)

    left = node_left[active_nodes]
    right = node_right[active_nodes]
    split_valid = (left >= 0) & (right >= 0) & still_refining[:, :, None]
    child_nodes = torch.stack([left.clamp_min(0), right.clamp_min(0)], dim=-1)
    child_logits, child_values = _hier_node_logits_values(
        q_tok,
        k_heads,
        v_heads,
        img_tok_idx,
        child_nodes.reshape(H, Q, patch_budget * 2),
        node_rep,
        node_log_size,
        scale,
        use_multiplicity_correction,
    )
    child_logits = child_logits.reshape(H, Q, patch_budget, 2)
    child_values = child_values.reshape(H, Q, patch_budget, 2, v_heads.shape[-1])
    child_rel = torch.exp((child_logits - log_z[:, :, None, None]).clamp(max=80.0))
    child_num = (child_rel[..., None] * child_values).sum(dim=-2)
    split_denom = (1.0 - patch_weights + child_rel.sum(dim=-1)).clamp_min(eps)
    split_out = (
        current_out[:, :, None, :]
        - patch_weights[..., None] * patch_values
        + child_num
    ) / split_denom[..., None]
    split_gain = _hier_relative_l2(split_out, current_out[:, :, None, :], eps)
    split_gain = split_gain.masked_fill(~split_valid, -math.inf)

    inactive_valid = (frontier_nodes >= 0) & (~frontier_active) & still_refining[:, :, None]
    activate_logits, activate_values = _hier_node_logits_values(
        q_tok,
        k_heads,
        v_heads,
        img_tok_idx,
        frontier_nodes,
        node_rep,
        node_log_size,
        scale,
        use_multiplicity_correction,
    )
    activate_rel = torch.exp((activate_logits - log_z[:, :, None]).clamp(max=80.0))
    activate_denom = (1.0 + activate_rel).clamp_min(eps)
    activate_out = (
        current_out[:, :, None, :]
        + activate_rel[..., None] * activate_values
    ) / activate_denom[..., None]
    activate_gain = _hier_relative_l2(activate_out, current_out[:, :, None, :], eps)
    activate_gain = activate_gain.masked_fill(~inactive_valid, -math.inf)

    return split_gain, activate_gain, prune_cost


def _hier_frame_counts(node_frame, active_nodes, num_imgs):
    frame_ids = node_frame[active_nodes]
    counts = torch.zeros(
        (*active_nodes.shape[:2], num_imgs),
        dtype=torch.float32,
        device=active_nodes.device,
    )
    counts.scatter_add_(2, frame_ids, torch.ones_like(frame_ids, dtype=torch.float32))
    return counts


def _hier_write_stats(
    cfg,
    tree,
    head_idx,
    head_count,
    strategy,
    patch_budget,
    num_imgs,
    num_img_toks,
    init_nodes,
    query_count_per_head,
    final_frame_sum,
    final_frame_min,
    final_frame_max,
    round_success_sum,
    split_frame_counts,
    activate_frame_counts,
    prune_frame_counts,
    round_query_count,
    round_frame_sum,
    round_accept_count,
    round_split_frame_counts,
    round_activate_frame_counts,
    round_prune_frame_counts,
    round_gain_sum,
    round_prune_cost_sum,
    elapsed,
):
    stats_dir = cfg.get("stats_dir", None)
    if not stats_dir:
        return

    os.makedirs(stats_dir, exist_ok=True)
    node_frame_cpu = tree["frame"].detach().cpu()
    init_counts = torch.bincount(node_frame_cpu[init_nodes.detach().cpu()], minlength=num_imgs)
    denom = max(1, int(query_count_per_head))
    heads = list(range(int(head_idx), int(head_idx) + int(head_count)))
    payload = {
        "time": time.time(),
        "pid": os.getpid(),
        "dataset": os.environ.get("SAF3R_CURRENT_DATASET"),
        "scene": os.environ.get("SAF3R_CURRENT_SCENE"),
        "layer": int(cfg.get("layer_idx", -1)),
        "heads": heads,
        "mode": "hier_split_prune_keys",
        "strategy": strategy,
        "patch_budget": int(patch_budget),
        "max_rounds": int(cfg.get("max_rounds", 0)),
        "gamma": float(cfg.get("gamma", 1.0)),
        "use_multiplicity_correction": bool(cfg.get("use_multiplicity_correction", True)),
        "num_frames": int(num_imgs),
        "num_image_tokens": int(num_img_toks),
        "patch_key_retention_ratio": float(patch_budget / max(num_img_toks, 1)),
        "uniform_init_frame_counts": [int(x) for x in init_counts.tolist()],
        "query_count_per_head": int(query_count_per_head),
        "elapsed_seconds": float(elapsed),
        "per_head": [],
    }

    for local_head, global_head in enumerate(heads):
        per_round = []
        for round_idx in range(round_query_count.shape[1]):
            round_denom = max(1, int(round_query_count[local_head, round_idx].item()))
            accept_denom = max(1, int(round_accept_count[local_head, round_idx].item()))
            per_round.append(
                {
                    "round": int(round_idx + 1),
                    "query_count": int(round_query_count[local_head, round_idx].item()),
                    "accepted_count": int(round_accept_count[local_head, round_idx].item()),
                    "frame_count_mean_after_round": [
                        float(x) for x in (round_frame_sum[local_head, round_idx] / round_denom).tolist()
                    ],
                    "split_frame_counts": [
                        int(x) for x in round_split_frame_counts[local_head, round_idx].tolist()
                    ],
                    "activate_frame_counts": [
                        int(x) for x in round_activate_frame_counts[local_head, round_idx].tolist()
                    ],
                    "prune_frame_counts": [
                        int(x) for x in round_prune_frame_counts[local_head, round_idx].tolist()
                    ],
                    "gain_mean": float(round_gain_sum[local_head, round_idx] / accept_denom),
                    "prune_cost_mean": float(round_prune_cost_sum[local_head, round_idx] / accept_denom),
                }
            )

        payload["per_head"].append(
            {
                "head": int(global_head),
                "final_frame_count_mean": [
                    float(x) for x in (final_frame_sum[local_head] / denom).tolist()
                ],
                "final_frame_count_min": [
                    float(x) for x in final_frame_min[local_head].tolist()
                ],
                "final_frame_count_max": [
                    float(x) for x in final_frame_max[local_head].tolist()
                ],
                "successful_refinement_rounds_mean": float(round_success_sum[local_head] / denom),
                "split_frame_counts": [int(x) for x in split_frame_counts[local_head].tolist()],
                "activate_frame_counts": [int(x) for x in activate_frame_counts[local_head].tolist()],
                "prune_frame_counts": [int(x) for x in prune_frame_counts[local_head].tolist()],
                "per_round": per_round,
            }
        )

    path = os.path.join(stats_dir, f"hier_split_prune_pid{os.getpid()}.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(payload) + "\n")


def _hier_uniform_attention(
    q,
    k,
    v,
    spec_tok_idx,
    img_tok_idx,
    cfg,
    tree,
    init_nodes,
    patch_budget,
    head_idx,
):
    B, H, _, D = q.shape
    query_chunk_size = int(cfg.get("query_chunk_size", 8))
    include_special_keys = bool(cfg.get("include_special_keys", True))
    use_multiplicity_correction = bool(cfg.get("use_multiplicity_correction", True))
    node_log_size = tree["size"].float().log()
    scale = 1.0 / math.sqrt(D)
    started = time.perf_counter()

    q_img = q[:, :, img_tok_idx, :].squeeze(0).float()
    k_heads = k.squeeze(0).float()
    v_heads = v.squeeze(0).float()
    out = torch.empty_like(v)
    if spec_tok_idx.numel() > 0:
        out[:, :, spec_tok_idx, :] = F.scaled_dot_product_attention(q[:, :, spec_tok_idx, :], k, v)

    base_nodes = init_nodes[None, None, :].expand(H, 1, patch_budget)
    for start in range(0, img_tok_idx.numel(), query_chunk_size):
        end = min(start + query_chunk_size, img_tok_idx.numel())
        q_tok = q_img[:, start:end, :]
        nodes = base_nodes.expand(H, end - start, patch_budget)
        out_tok = _hier_attention_from_nodes(
            q_tok,
            k_heads,
            v_heads,
            spec_tok_idx,
            img_tok_idx,
            nodes,
            tree["rep"],
            node_log_size,
            scale,
            use_multiplicity_correction,
            include_special_keys,
        )
        out[:, :, img_tok_idx[start:end], :] = out_tok.unsqueeze(0).to(dtype=v.dtype)

    num_imgs = int(tree["roots"].numel())
    node_frame = tree["frame"].detach().cpu()
    init_counts = torch.bincount(node_frame[init_nodes.detach().cpu()], minlength=num_imgs).to(torch.float64)
    final_frame_sum = init_counts[None, :].expand(H, -1).clone() * int(img_tok_idx.numel())
    final_frame_min = init_counts[None, :].expand(H, -1).clone()
    final_frame_max = init_counts[None, :].expand(H, -1).clone()
    empty_round_count = torch.zeros((H, 0), dtype=torch.long)
    empty_round_frame = torch.zeros((H, 0, num_imgs), dtype=torch.float64)
    empty_round_gain = torch.zeros((H, 0), dtype=torch.float64)
    zero_frame_counts = torch.zeros((H, num_imgs), dtype=torch.long)
    _hier_write_stats(
        cfg,
        tree,
        head_idx,
        H,
        "uniform",
        patch_budget,
        num_imgs,
        int(img_tok_idx.numel()),
        init_nodes,
        int(img_tok_idx.numel()),
        final_frame_sum,
        final_frame_min,
        final_frame_max,
        torch.zeros(H, dtype=torch.float64),
        zero_frame_counts,
        zero_frame_counts,
        zero_frame_counts,
        empty_round_count,
        empty_round_frame,
        empty_round_count,
        empty_round_frame.to(dtype=torch.long),
        empty_round_frame.to(dtype=torch.long),
        empty_round_frame.to(dtype=torch.long),
        empty_round_gain,
        empty_round_gain,
        time.perf_counter() - started,
    )
    return out


def _hier_topk_attention(q, k, v, spec_tok_idx, img_tok_idx, cfg, tree, init_nodes, patch_budget, p_h, p_w, head_idx):
    B, H, _, D = q.shape
    query_chunk_size = int(cfg.get("query_chunk_size", 8))
    include_special_keys = bool(cfg.get("include_special_keys", True))
    scale = 1.0 / math.sqrt(D)
    started = time.perf_counter()

    q_img = q[:, :, img_tok_idx, :].squeeze(0).float()
    k_heads = k.squeeze(0).float()
    v_heads = v.squeeze(0).float()
    k_img = k_heads[:, img_tok_idx, :]
    out = torch.empty_like(v)
    if spec_tok_idx.numel() > 0:
        out[:, :, spec_tok_idx, :] = F.scaled_dot_product_attention(q[:, :, spec_tok_idx, :], k, v)

    num_imgs = int(tree["roots"].numel())
    patches_per_img = p_h * p_w
    final_frame_sum = torch.zeros((H, num_imgs), dtype=torch.float64)
    final_frame_min = torch.full((H, num_imgs), float("inf"), dtype=torch.float64)
    final_frame_max = torch.zeros((H, num_imgs), dtype=torch.float64)

    for start in range(0, img_tok_idx.numel(), query_chunk_size):
        end = min(start + query_chunk_size, img_tok_idx.numel())
        q_tok = q_img[:, start:end, :]
        patch_scores = torch.matmul(q_tok, k_img.transpose(-1, -2)) * scale
        top_idx = torch.topk(patch_scores, k=patch_budget, dim=-1, largest=True, sorted=False).indices
        final_key_idx = img_tok_idx[top_idx]
        if include_special_keys and spec_tok_idx.numel() > 0:
            spec_idx = spec_tok_idx[None, None, :].expand(H, end - start, -1)
            final_key_idx = torch.cat([spec_idx, final_key_idx], dim=-1)
        k_sel = _gather_query_kv(k_heads, final_key_idx)
        v_sel = _gather_query_kv(v_heads, final_key_idx)
        scores = (q_tok[:, :, None, :] * k_sel).sum(dim=-1) * scale
        attn = torch.softmax(scores, dim=-1)
        out_tok = (attn[..., None] * v_sel).sum(dim=-2)
        out[:, :, img_tok_idx[start:end], :] = out_tok.unsqueeze(0).to(dtype=v.dtype)

        frame_ids = (top_idx // patches_per_img).to(dtype=torch.long)
        frame_counts = torch.zeros((H, end - start, num_imgs), dtype=torch.float32, device=q.device)
        frame_counts.scatter_add_(2, frame_ids, torch.ones_like(frame_ids, dtype=torch.float32))
        frame_counts = frame_counts.detach().cpu().to(torch.float64)
        final_frame_sum += frame_counts.sum(dim=1)
        final_frame_min = torch.minimum(final_frame_min, frame_counts.amin(dim=1))
        final_frame_max = torch.maximum(final_frame_max, frame_counts.amax(dim=1))

    empty_round_count = torch.zeros((H, 0), dtype=torch.long)
    empty_round_frame = torch.zeros((H, 0, num_imgs), dtype=torch.float64)
    empty_round_gain = torch.zeros((H, 0), dtype=torch.float64)
    zero_frame_counts = torch.zeros((H, num_imgs), dtype=torch.long)
    _hier_write_stats(
        cfg,
        tree,
        head_idx,
        H,
        "topk",
        patch_budget,
        num_imgs,
        int(img_tok_idx.numel()),
        init_nodes,
        int(img_tok_idx.numel()),
        final_frame_sum,
        final_frame_min,
        final_frame_max,
        torch.zeros(H, dtype=torch.float64),
        zero_frame_counts,
        zero_frame_counts,
        zero_frame_counts,
        empty_round_count,
        empty_round_frame,
        empty_round_count,
        empty_round_frame.to(dtype=torch.long),
        empty_round_frame.to(dtype=torch.long),
        empty_round_frame.to(dtype=torch.long),
        empty_round_gain,
        empty_round_gain,
        time.perf_counter() - started,
    )
    return out


def _hier_refined_attention(
    q,
    k,
    v,
    spec_tok_idx,
    img_tok_idx,
    cfg,
    tree,
    init_nodes,
    patch_budget,
    head_idx,
):
    B, H, _, D = q.shape
    strategy = cfg.get("strategy", "hierarchical")
    if strategy not in {"hierarchical", "random"}:
        raise ValueError(f"Unsupported hierarchical split-prune strategy: {strategy}")

    query_chunk_size = int(cfg.get("query_chunk_size", 8))
    max_rounds = int(cfg.get("max_rounds", 1))
    gamma = float(cfg.get("gamma", 1.0))
    eps = float(cfg.get("eps", 1e-6))
    include_special_keys = bool(cfg.get("include_special_keys", True))
    use_multiplicity_correction = bool(cfg.get("use_multiplicity_correction", True))
    if max_rounds < 0:
        raise ValueError(f"max_rounds must be non-negative, got {max_rounds}")

    node_rep = tree["rep"]
    node_frame = tree["frame"]
    node_left = tree["left"]
    node_right = tree["right"]
    node_log_size = tree["size"].float().log()
    num_imgs = int(tree["roots"].numel())
    num_img_toks = int(img_tok_idx.numel())
    scale = 1.0 / math.sqrt(D)

    seed = int(cfg.get("seed", 0))
    seed = seed + 1000003 * int(cfg.get("layer_idx", 0)) + 1009 * int(head_idx) + 37 * num_img_toks
    generator = _make_generator(q.device, seed)

    q_img = q[:, :, img_tok_idx, :].squeeze(0).float()
    k_heads = k.squeeze(0).float()
    v_heads = v.squeeze(0).float()
    out = torch.empty_like(v)
    if spec_tok_idx.numel() > 0:
        out[:, :, spec_tok_idx, :] = F.scaled_dot_product_attention(q[:, :, spec_tok_idx, :], k, v)

    query_count_per_head = 0
    final_frame_sum = torch.zeros((H, num_imgs), dtype=torch.float64)
    final_frame_min = torch.full((H, num_imgs), float("inf"), dtype=torch.float64)
    final_frame_max = torch.zeros((H, num_imgs), dtype=torch.float64)
    round_success_sum = torch.zeros(H, dtype=torch.float64)
    split_frame_counts = torch.zeros((H, num_imgs), dtype=torch.long)
    activate_frame_counts = torch.zeros((H, num_imgs), dtype=torch.long)
    prune_frame_counts = torch.zeros((H, num_imgs), dtype=torch.long)
    round_query_count = torch.zeros((H, max_rounds), dtype=torch.long)
    round_frame_sum = torch.zeros((H, max_rounds, num_imgs), dtype=torch.float64)
    round_accept_count = torch.zeros((H, max_rounds), dtype=torch.long)
    round_split_frame_counts = torch.zeros((H, max_rounds, num_imgs), dtype=torch.long)
    round_activate_frame_counts = torch.zeros((H, max_rounds, num_imgs), dtype=torch.long)
    round_prune_frame_counts = torch.zeros((H, max_rounds, num_imgs), dtype=torch.long)
    round_gain_sum = torch.zeros((H, max_rounds), dtype=torch.float64)
    round_prune_cost_sum = torch.zeros((H, max_rounds), dtype=torch.float64)

    started = time.perf_counter()
    max_frontier = patch_budget + max_rounds + 1

    for start in range(0, num_img_toks, query_chunk_size):
        end = min(start + query_chunk_size, num_img_toks)
        q_count = end - start
        q_tok = q_img[:, start:end, :]
        query_count_per_head += q_count

        frontier_nodes = torch.full(
            (H, q_count, max_frontier),
            -1,
            dtype=torch.long,
            device=q.device,
        )
        frontier_active = torch.zeros((H, q_count, max_frontier), dtype=torch.bool, device=q.device)
        frontier_nodes[:, :, :patch_budget] = init_nodes[None, None, :].expand(H, q_count, -1)
        frontier_active[:, :, :patch_budget] = True
        frontier_count = torch.full((H, q_count), patch_budget, dtype=torch.long, device=q.device)
        still_refining = torch.ones((H, q_count), dtype=torch.bool, device=q.device)
        round_counts = torch.zeros((H, q_count), dtype=torch.long, device=q.device)

        for round_idx in range(max_rounds):
            active_nodes, active_pos = _hier_gather_active_nodes(frontier_nodes, frontier_active, patch_budget)

            if strategy == "random":
                split_valid = (node_left[active_nodes] >= 0) & still_refining[:, :, None]
                if patch_budget <= 1:
                    split_valid.fill_(False)
                inactive_valid = (frontier_nodes >= 0) & (~frontier_active) & still_refining[:, :, None]
                split_choice, has_split = _hier_select_random_index(split_valid, generator)
                activate_choice, has_activate = _hier_select_random_index(inactive_valid, generator)
                split_score = torch.rand((H, q_count), device=q.device, generator=generator).masked_fill(~has_split, -math.inf)
                activate_score = torch.rand((H, q_count), device=q.device, generator=generator).masked_fill(~has_activate, -math.inf)
                use_split = split_score >= activate_score
                has_add = torch.isfinite(torch.maximum(split_score, activate_score))
                prune_mask = torch.ones((H, q_count, patch_budget), dtype=torch.bool, device=q.device)
                prune_mask &= has_add[:, :, None]
                prune_mask &= still_refining[:, :, None]
                parent_mask = F.one_hot(split_choice.clamp_min(0), num_classes=patch_budget).to(dtype=torch.bool)
                prune_mask = torch.where(use_split[:, :, None], prune_mask & ~parent_mask, prune_mask)
                prune_choice, has_prune = _hier_select_random_index(prune_mask, generator)
                accept = has_add & has_prune & still_refining
                best_gain = torch.zeros((H, q_count), device=q.device)
                prune_value = torch.zeros((H, q_count), device=q.device)
            else:
                split_gain, activate_gain, prune_cost = _hier_compute_scores(
                    q_tok,
                    k_heads,
                    v_heads,
                    spec_tok_idx,
                    img_tok_idx,
                    active_nodes,
                    frontier_nodes,
                    frontier_active,
                    node_rep,
                    node_log_size,
                    node_left,
                    node_right,
                    scale,
                    use_multiplicity_correction,
                    include_special_keys,
                    eps,
                    still_refining,
                )
                split_score, split_choice = split_gain.max(dim=-1)
                activate_score, activate_choice = activate_gain.max(dim=-1)
                use_split = split_score >= activate_score
                best_gain = torch.maximum(split_score, activate_score)

                prune_for_split = prune_cost.clone()
                parent_mask = F.one_hot(split_choice.clamp_min(0), num_classes=patch_budget).to(dtype=torch.bool)
                prune_for_split = prune_for_split.masked_fill(parent_mask, math.inf)
                split_prune_value, split_prune_choice = prune_for_split.min(dim=-1)
                activate_prune_value, activate_prune_choice = prune_cost.min(dim=-1)
                prune_value = torch.where(use_split, split_prune_value, activate_prune_value)
                prune_choice = torch.where(use_split, split_prune_choice, activate_prune_choice)
                accept = torch.isfinite(best_gain) & torch.isfinite(prune_value)
                accept &= best_gain > (gamma * prune_value)
                accept &= still_refining

            if not accept.any():
                break

            accepted_indices = accept.nonzero(as_tuple=False)
            for hq in accepted_indices:
                h = int(hq[0].item())
                q_pos = int(hq[1].item())
                round_accept_count[h, round_idx] += 1
                round_gain_sum[h, round_idx] += float(best_gain[h, q_pos].item())
                round_prune_cost_sum[h, round_idx] += float(prune_value[h, q_pos].item())
                prune_slot = int(prune_choice[h, q_pos].item())
                prune_frontier_pos = int(active_pos[h, q_pos, prune_slot].item())
                pruned_node = int(frontier_nodes[h, q_pos, prune_frontier_pos].item())

                if bool(use_split[h, q_pos].item()):
                    split_slot = int(split_choice[h, q_pos].item())
                    parent_frontier_pos = int(active_pos[h, q_pos, split_slot].item())
                    parent_node = int(frontier_nodes[h, q_pos, parent_frontier_pos].item())
                    left_child = int(node_left[parent_node].item())
                    right_child = int(node_right[parent_node].item())
                    append_pos = int(frontier_count[h, q_pos].item())

                    frontier_nodes[h, q_pos, parent_frontier_pos] = left_child
                    frontier_active[h, q_pos, parent_frontier_pos] = True
                    frontier_nodes[h, q_pos, append_pos] = right_child
                    frontier_active[h, q_pos, append_pos] = True
                    frontier_count[h, q_pos] += 1
                    split_frame_counts[h, int(node_frame[parent_node].item())] += 1
                    round_split_frame_counts[h, round_idx, int(node_frame[parent_node].item())] += 1
                else:
                    activate_frontier_pos = int(activate_choice[h, q_pos].item())
                    activated_node = int(frontier_nodes[h, q_pos, activate_frontier_pos].item())
                    frontier_active[h, q_pos, activate_frontier_pos] = True
                    activate_frame_counts[h, int(node_frame[activated_node].item())] += 1
                    round_activate_frame_counts[h, round_idx, int(node_frame[activated_node].item())] += 1

                frontier_active[h, q_pos, prune_frontier_pos] = False
                prune_frame_counts[h, int(node_frame[pruned_node].item())] += 1
                round_prune_frame_counts[h, round_idx, int(node_frame[pruned_node].item())] += 1
                round_counts[h, q_pos] += 1

            still_refining &= accept
            active_nodes_after_round, _ = _hier_gather_active_nodes(frontier_nodes, frontier_active, patch_budget)
            frame_counts_after_round = _hier_frame_counts(
                node_frame,
                active_nodes_after_round,
                num_imgs,
            ).detach().cpu().to(torch.float64)
            round_frame_sum[:, round_idx, :] += frame_counts_after_round.sum(dim=1)
            round_query_count[:, round_idx] += q_count

        active_nodes, _ = _hier_gather_active_nodes(frontier_nodes, frontier_active, patch_budget)
        out_tok = _hier_attention_from_nodes(
            q_tok,
            k_heads,
            v_heads,
            spec_tok_idx,
            img_tok_idx,
            active_nodes,
            node_rep,
            node_log_size,
            scale,
            use_multiplicity_correction,
            include_special_keys,
        )
        out[:, :, img_tok_idx[start:end], :] = out_tok.unsqueeze(0).to(dtype=v.dtype)

        frame_counts = _hier_frame_counts(node_frame, active_nodes, num_imgs).detach().cpu().to(torch.float64)
        final_frame_sum += frame_counts.sum(dim=1)
        final_frame_min = torch.minimum(final_frame_min, frame_counts.amin(dim=1))
        final_frame_max = torch.maximum(final_frame_max, frame_counts.amax(dim=1))
        round_success_sum += round_counts.detach().cpu().sum(dim=1).to(torch.float64)

    _hier_write_stats(
        cfg,
        tree,
        head_idx,
        H,
        strategy,
        patch_budget,
        num_imgs,
        num_img_toks,
        init_nodes,
        query_count_per_head,
        final_frame_sum,
        final_frame_min,
        final_frame_max,
        round_success_sum,
        split_frame_counts,
        activate_frame_counts,
        prune_frame_counts,
        round_query_count,
        round_frame_sum,
        round_accept_count,
        round_split_frame_counts,
        round_activate_frame_counts,
        round_prune_frame_counts,
        round_gain_sum,
        round_prune_cost_sum,
        time.perf_counter() - started,
    )

    return out


def hierarchical_split_prune_attention(q, k, v, spec_tok_idx, img_tok_idx, cfg, p_h, p_w, head_idx=0):
    """
    Fixed patch-key-budget hierarchical sparse attention.

    Patch queries use a query/head-specific sparse set with exactly B patch
    representatives. Special-token queries keep dense attention; special keys
    may be included for patch queries but do not count against B.
    """
    assert q.shape[0] == 1

    strategy = cfg.get("strategy", "hierarchical")
    if strategy not in {"uniform", "topk", "random", "hierarchical"}:
        raise ValueError(f"Unsupported hier_split_prune strategy: {strategy}")

    query_chunk_size = int(cfg.get("query_chunk_size", 8))
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")

    device = q.device
    spec_tok_idx = spec_tok_idx.to(device=device, dtype=torch.long)
    img_tok_idx = img_tok_idx.to(device=device, dtype=torch.long)
    num_img_toks = int(img_tok_idx.numel())
    patches_per_img = p_h * p_w
    num_imgs = num_img_toks // patches_per_img
    patch_budget = _hier_patch_budget(cfg, num_imgs, num_img_toks)

    tree = _build_hierarchical_patch_tree(num_imgs, p_h, p_w, device)
    init_nodes = _hier_uniform_warm_start(tree, patch_budget, num_imgs)
    patch_budget = int(init_nodes.numel())

    if strategy == "topk":
        return _hier_topk_attention(q, k, v, spec_tok_idx, img_tok_idx, cfg, tree, init_nodes, patch_budget, p_h, p_w, head_idx)
    if strategy == "uniform":
        return _hier_uniform_attention(q, k, v, spec_tok_idx, img_tok_idx, cfg, tree, init_nodes, patch_budget, head_idx)
    return _hier_refined_attention(
        q,
        k,
        v,
        spec_tok_idx,
        img_tok_idx,
        cfg,
        tree,
        init_nodes,
        patch_budget,
        head_idx,
    )


def _dispersion_scores(k_img, v_img, block_idx, valid_mask, dispersion_lambda, eps):
    safe_idx = block_idx.clamp_min(0).reshape(-1)
    H, _, D = k_img.shape
    num_blocks, block_area = block_idx.shape
    valid = valid_mask.to(device=k_img.device)
    counts = valid.sum(dim=1).clamp_min(1).to(dtype=torch.float32, device=k_img.device)
    weights = valid.to(dtype=torch.float32, device=k_img.device)[None, :, :, None]

    k_block = k_img[:, safe_idx, :].reshape(H, num_blocks, block_area, D).float()
    v_block = v_img[:, safe_idx, :].reshape(H, num_blocks, block_area, v_img.shape[-1]).float()

    k_mean = (k_block * weights).sum(dim=2) / counts[None, :, None]
    v_mean = (v_block * weights).sum(dim=2) / counts[None, :, None]
    k_disp = (((k_block - k_mean[:, :, None, :]).norm(dim=-1) * valid[None]).sum(dim=-1) / counts[None])
    v_disp = (((v_block - v_mean[:, :, None, :]).norm(dim=-1) * valid[None]).sum(dim=-1) / counts[None])

    k_disp = _zscore_last_dim(k_disp, eps=eps)
    v_disp = _zscore_last_dim(v_disp, eps=eps)
    return dispersion_lambda * k_disp + (1.0 - dispersion_lambda) * v_disp


def _select_random_pairs(pair_valid, generator):
    rand = torch.rand(pair_valid.shape, device=pair_valid.device, generator=generator)
    rand = rand.masked_fill(~pair_valid, -1.0)
    return rand.argmax(dim=-1)


def _select_reconstruction_pairs(ref_z, ref_v, ref_valid, g_ref, mu_ref, alpha, eps, pair_i, pair_j):
    zi = ref_z[..., pair_i]
    zj = ref_z[..., pair_j]
    pair_valid = ref_valid[..., pair_i] & ref_valid[..., pair_j]

    pair_logits = torch.stack([zi, zj], dim=-1)
    pair_logmass = torch.logsumexp(pair_logits, dim=-1)
    block_counts = ref_valid.sum(dim=-1).clamp_min(1).to(dtype=ref_z.dtype)
    pair_logmass = pair_logmass + torch.log(block_counts[..., None] / 2.0)

    pair_weight = torch.softmax(pair_logits, dim=-1)
    vi = ref_v[..., pair_i, :]
    vj = ref_v[..., pair_j, :]
    pair_mu = pair_weight[..., 0, None] * vi + pair_weight[..., 1, None] * vj

    mass_loss = (pair_logmass - g_ref[..., None]).pow(2)
    denom = mu_ref.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
    value_loss = (pair_mu - mu_ref[..., None, :]).pow(2).sum(dim=-1) / denom
    loss = alpha * mass_loss + (1.0 - alpha) * value_loss
    loss = loss.masked_fill(~pair_valid, math.inf)
    return loss.argmin(dim=-1)


def block3x3_violation_guided_attention(
    q, k, v, spec_tok_idx, img_tok_idx, cfg, p_h, p_w, head_idx=0, est_topk_idx=None
):
    """
    3x3 block sparse attention with query-specific violation-guided refinement.

    Each image-token block starts from its center key. Candidate blocks are
    chosen from per-head K/V dispersion; each query then independently refines
    the highest violation blocks by replacing the center key with two keys from
    the same 3x3 block. The final attention only scores the resulting sparse
    per-query key set.
    """
    assert q.shape[0] == 1

    block_size = int(cfg.get("block_size", 3))
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    candidate_ratio = float(cfg.get("candidate_ratio", 0.3))
    if candidate_ratio <= 0 or candidate_ratio > 1:
        raise ValueError(f"candidate_ratio must be in (0, 1], got {candidate_ratio}")

    dispersion_lambda = float(cfg.get("dispersion_lambda", cfg.get("lambda", 0.5)))
    violation_alpha = float(cfg.get("violation_alpha", cfg.get("alpha", 0.5)))
    if dispersion_lambda < 0 or dispersion_lambda > 1:
        raise ValueError(f"dispersion_lambda must be in [0, 1], got {dispersion_lambda}")
    if violation_alpha < 0 or violation_alpha > 1:
        raise ValueError(f"violation_alpha must be in [0, 1], got {violation_alpha}")

    refinement_strategy = cfg.get("refinement_strategy", "violation")
    if refinement_strategy not in {"violation", "random", "none"}:
        raise ValueError(f"Unsupported refinement_strategy: {refinement_strategy}")
    candidate_selection = cfg.get(
        "candidate_selection",
        "random" if refinement_strategy == "random" else "dispersion",
    )
    if candidate_selection not in {"dispersion", "random"}:
        raise ValueError(f"Unsupported candidate_selection: {candidate_selection}")
    pair_selection = cfg.get(
        "pair_selection",
        "random" if refinement_strategy == "random" else "optimal",
    )
    if pair_selection not in {"optimal", "random"}:
        raise ValueError(f"Unsupported pair_selection: {pair_selection}")

    query_chunk_size = int(cfg.get("query_chunk_size", 8))
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")

    eps = float(cfg.get("eps", 1e-6))
    include_special_keys = bool(cfg.get("include_special_keys", True))

    device = q.device
    spec_tok_idx = spec_tok_idx.to(device=device, dtype=torch.long)
    img_tok_idx = img_tok_idx.to(device=device, dtype=torch.long)

    B, H, N, D = q.shape
    patches_per_img = p_h * p_w
    num_img_tokens = img_tok_idx.numel()
    num_imgs = num_img_tokens // patches_per_img
    num_toks_per_img = patches_per_img + spec_tok_idx.numel() // max(num_imgs, 1)
    num_special_tokens = spec_tok_idx.numel() // max(num_imgs, 1)
    num_special_toks = spec_tok_idx.numel()

    block_idx, center_img_idx, center_pos = _make_patch_blocks_and_centers(
        num_imgs, p_h, p_w, block_size, device
    )
    valid_mask = block_idx >= 0
    safe_block_idx = block_idx.clamp_min(0)
    block_counts = valid_mask.sum(dim=1)
    refinable_mask = block_counts >= 2
    refinable_count = int(refinable_mask.sum().item())
    num_blocks, block_area = block_idx.shape

    initial_candidate_count = (
        0
        if refinable_count == 0
        else max(1, min(refinable_count, int(round(num_blocks * candidate_ratio))))
    )
    base_count, refine_count = _block3x3_budgeted_counts(
        cfg,
        N,
        num_toks_per_img,
        num_imgs,
        num_special_toks,
        num_img_tokens,
        num_special_tokens,
        num_blocks,
        refinable_count,
        initial_candidate_count,
        head_idx,
        est_topk_idx=est_topk_idx,
    )

    allocation_scope = cfg.get("allocation_scope", "budgeted_uniform")
    base_blocks = _uniform_block_indices(num_blocks, base_count, device)
    if allocation_scope == "budgeted_uniform":
        base_count, refine_count, base_blocks, candidate_count = _tighten_budgeted_counts_for_candidates(
            base_count,
            refine_count,
            base_count + refine_count,
            num_blocks,
            refinable_mask,
            candidate_ratio,
            device,
        )
    else:
        candidate_count = initial_candidate_count

    if refinement_strategy == "none" or refinable_count == 0 or refine_count <= 0:
        return _center_block_attention(
            q, k, v, spec_tok_idx, img_tok_idx, center_img_idx[base_blocks], include_special_keys
        )

    layer_idx = int(cfg.get("layer_idx", 0))
    seed = int(cfg.get("seed", 0))
    seed = seed + 1000003 * layer_idx + 1009 * int(head_idx) + 37 * N
    generator = _make_generator(device, seed)

    q_img = q[:, :, img_tok_idx, :].squeeze(0).float()
    k_heads = k.squeeze(0).float()
    v_heads = v.squeeze(0).float()
    k_img = k_heads[:, img_tok_idx, :]
    v_img = v_heads[:, img_tok_idx, :]

    candidate_mask = refinable_mask[None, :].expand(H, -1).clone()
    if allocation_scope == "budgeted_uniform":
        base_mask = torch.zeros((H, num_blocks), dtype=torch.bool, device=device)
        base_mask[:, base_blocks] = True
        candidate_mask &= base_mask
    if candidate_selection == "random":
        disp = torch.rand((H, num_blocks), device=device, generator=generator)
    else:
        disp = _dispersion_scores(
            k_img,
            v_img,
            safe_block_idx,
            valid_mask,
            dispersion_lambda,
            eps,
        )
    disp = disp.masked_fill(~candidate_mask, -math.inf)
    candidate_blocks = torch.topk(
        disp,
        k=candidate_count,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices

    cand_tokens = safe_block_idx[candidate_blocks]
    cand_valid = valid_mask[candidate_blocks]
    cand_center_pos = center_pos[candidate_blocks]
    cand_counts = block_counts[candidate_blocks].to(dtype=torch.float32, device=device)
    cand_k = _gather_query_kv(k_img, cand_tokens)
    cand_v = _gather_query_kv(v_img, cand_tokens)
    center_v = v_img.gather(
        1,
        center_img_idx[None, :, None].expand(H, -1, v_img.shape[-1]),
    )
    cand_center_v = center_v.gather(
        1,
        candidate_blocks[..., None].expand(H, candidate_count, v_img.shape[-1]),
    )

    pair_idx = torch.triu_indices(block_area, block_area, offset=1, device=device)
    pair_i, pair_j = pair_idx[0], pair_idx[1]

    out = torch.empty_like(v)
    if num_special_toks > 0:
        q_spec = q[:, :, spec_tok_idx, :]
        out[:, :, spec_tok_idx, :] = F.scaled_dot_product_attention(q_spec, k, v)

    scale = 1.0 / math.sqrt(D)
    base_centers = center_img_idx[base_blocks]
    all_centers = base_centers[None, None, :].expand(H, 1, base_count)
    block_to_base_pos = torch.full((num_blocks,), -1, dtype=torch.long, device=device)
    block_to_base_pos[base_blocks] = torch.arange(base_count, device=device, dtype=torch.long)

    for start in range(0, num_img_tokens, query_chunk_size):
        end = min(start + query_chunk_size, num_img_tokens)
        q_count = end - start
        q_tok = q_img[:, start:end, :]

        z = torch.einsum("hqd,hcad->hqca", q_tok, cand_k) * scale
        z = z.masked_fill(~cand_valid[:, None, :, :], -math.inf)
        g = torch.logsumexp(z, dim=-1)
        center_z = z.gather(
            -1,
            cand_center_pos[:, None, :, None].expand(H, q_count, candidate_count, 1),
        ).squeeze(-1)
        g_hat = center_z + torch.log(cand_counts[:, None, :])
        mass_error = (g - g_hat).abs()

        pi = torch.softmax(z, dim=-1)
        mu = torch.einsum("hqca,hcad->hqcd", pi, cand_v)
        value_error = (
            cand_center_v[:, None, :, :] - mu
        ).norm(dim=-1) / mu.norm(dim=-1).clamp_min(eps)

        if refinement_strategy == "random":
            violation_score = torch.rand(
                (H, q_count, candidate_count),
                device=device,
                generator=generator,
            )
        else:
            mass_norm = _zscore_last_dim(mass_error, eps=eps)
            value_norm = _zscore_last_dim(value_error, eps=eps)
            violation_score = violation_alpha * mass_norm + (1.0 - violation_alpha) * value_norm

        refine_pos = torch.topk(
            violation_score,
            k=refine_count,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        refine_blocks = candidate_blocks[:, None, :].expand(H, q_count, -1).gather(2, refine_pos)

        ref_z = z.gather(
            2,
            refine_pos[..., None].expand(H, q_count, refine_count, block_area),
        )
        ref_valid = cand_valid[:, None, :, :].expand(H, q_count, -1, block_area).gather(
            2,
            refine_pos[..., None].expand(H, q_count, refine_count, block_area),
        )
        ref_v = cand_v[:, None, :, :, :].expand(
            H, q_count, candidate_count, block_area, v_img.shape[-1]
        ).gather(
            2,
            refine_pos[..., None, None].expand(
                H, q_count, refine_count, block_area, v_img.shape[-1]
            ),
        )
        g_ref = g.gather(2, refine_pos)
        mu_ref = mu.gather(
            2,
            refine_pos[..., None].expand(H, q_count, refine_count, v_img.shape[-1]),
        )

        pair_valid = ref_valid[..., pair_i] & ref_valid[..., pair_j]
        if pair_selection == "random":
            best_pair = _select_random_pairs(pair_valid, generator)
        else:
            best_pair = _select_reconstruction_pairs(
                ref_z,
                ref_v,
                ref_valid,
                g_ref,
                mu_ref,
                violation_alpha,
                eps,
                pair_i,
                pair_j,
            )

        ref_tokens = safe_block_idx[refine_blocks]
        best_i = pair_i[best_pair]
        best_j = pair_j[best_pair]
        pair_tokens = torch.stack(
            [
                ref_tokens.gather(-1, best_i[..., None]).squeeze(-1),
                ref_tokens.gather(-1, best_j[..., None]).squeeze(-1),
            ],
            dim=-1,
        )

        center_keep = torch.ones((H, q_count, base_count), dtype=torch.bool, device=device)
        refine_base_pos = block_to_base_pos[refine_blocks]
        center_keep.scatter_(2, refine_base_pos, False)
        center_keys = all_centers.expand(H, q_count, -1)[center_keep].reshape(
            H, q_count, base_count - refine_count
        )
        final_img_idx = torch.cat(
            [center_keys, pair_tokens.reshape(H, q_count, 2 * refine_count)],
            dim=-1,
        )
        final_key_idx = img_tok_idx[final_img_idx]
        if include_special_keys:
            spec_idx = spec_tok_idx[None, None, :].expand(H, q_count, -1)
            final_key_idx = torch.cat([spec_idx, final_key_idx], dim=-1)

        k_sel = _gather_query_kv(k_heads, final_key_idx)
        v_sel = _gather_query_kv(v_heads, final_key_idx)
        scores = (q_tok[:, :, None, :] * k_sel).sum(dim=-1) * scale
        attn = torch.softmax(scores, dim=-1)
        out_tok = (attn[..., None] * v_sel).sum(dim=-2)
        out[:, :, img_tok_idx[start:end], :] = out_tok.unsqueeze(0).to(dtype=v.dtype)

        del z, g, center_z, g_hat, mass_error, pi, mu, value_error
        del violation_score, refine_pos, refine_blocks, ref_z, ref_valid, ref_v
        del g_ref, mu_ref, pair_valid, best_pair, ref_tokens, pair_tokens
        del center_keep, refine_base_pos, center_keys, final_img_idx, final_key_idx, k_sel, v_sel
        del scores, attn, out_tok

    return out


def query_topk_key_attention(q, k, v, cfg):
    """
    Exact per-query top-k attention over all keys.

    The top-k set is found from the raw QK logits, which has the same ordering
    as softmax attention weights. The computation is chunked over queries so it
    never materializes the full N x N attention matrix.
    """
    assert q.shape[0] == 1

    keep_ratio = float(cfg.get("keep_ratio", 1.0))
    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    chunk_size = int(cfg.get("query_chunk_size", 128))
    if chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {chunk_size}")

    B, H, N, D = q.shape
    keep_count = max(1, min(N, int(round(N * keep_ratio))))
    if keep_count >= N:
        return F.scaled_dot_product_attention(q, k, v)

    q_heads = q.squeeze(0).float()
    k_heads = k.squeeze(0).float()
    v_heads = v.squeeze(0).float()
    k_t = k_heads.transpose(-1, -2)
    scale = 1.0 / math.sqrt(D)

    out = torch.empty((H, N, v.shape[-1]), device=q.device, dtype=torch.float32)
    kth_smallest = N - keep_count + 1
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        scores = torch.matmul(q_heads[:, start:end, :], k_t) * scale
        threshold = torch.kthvalue(scores, kth_smallest, dim=-1).values
        scores.masked_fill_(scores < threshold[..., None], -math.inf)
        attn = torch.softmax(scores, dim=-1)
        out[:, start:end, :] = torch.matmul(attn, v_heads)

    return out.to(dtype=v.dtype).unsqueeze(0)


def fixed_key_attention(q, k, v, cfg, head_idx=0):
    """
    Query-independent global key subset attention.

    `uniform` keeps evenly spaced keys from the full token sequence. `random`
    keeps a random key subset per head, with deterministic layer/head seeding.
    """
    assert q.shape[0] == 1

    keep_ratio = float(cfg.get("keep_ratio", 1.0))
    if keep_ratio <= 0 or keep_ratio > 1:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

    selection = cfg.get("selection", "uniform")
    B, H, N, _ = q.shape
    keep_count = max(1, min(N, int(round(N * keep_ratio))))
    if keep_count >= N:
        return F.scaled_dot_product_attention(q, k, v)

    device = q.device
    if selection == "uniform":
        selected_idx = torch.linspace(
            0,
            N - 1,
            steps=keep_count,
            device=device,
        ).round().long()
        selected_idx = selected_idx[None, :].expand(H, -1)
    elif selection == "random":
        seed = int(cfg.get("seed", 0))
        layer_idx = int(cfg.get("layer_idx", 0))
        seed = seed + 1000003 * layer_idx + 1009 * int(head_idx) + 37 * keep_count + 17 * N
        generator = _make_generator(device, seed)
        scores = torch.rand((H, N), device=device, generator=generator)
        selected_idx = torch.topk(scores, k=keep_count, dim=1, largest=True, sorted=False).indices
        selected_idx = torch.sort(selected_idx, dim=1).values
    else:
        raise ValueError(f"Unsupported fixed key selection: {selection}")

    key_idx = selected_idx[None, :, :].expand(B, -1, -1)
    k_sel, v_sel = _gather_kv_per_head(k, v, key_idx)
    return F.scaled_dot_product_attention(q, k_sel, v_sel)


def _oracle_region_ids(num_imgs, p_h, p_w, block_size, num_special_tokens, device):
    """Return one fixed, non-overlapping region id for every sequence token."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    patches_per_img = p_h * p_w
    blocks_w = (p_w + block_size - 1) // block_size
    blocks_h = (p_h + block_size - 1) // block_size
    regions_per_img = num_special_tokens + blocks_h * blocks_w

    ids = torch.empty(
        num_imgs * (num_special_tokens + patches_per_img),
        device=device,
        dtype=torch.long,
    )
    for frame_idx in range(num_imgs):
        seq_base = frame_idx * (num_special_tokens + patches_per_img)
        region_base = frame_idx * regions_per_img
        ids[seq_base : seq_base + num_special_tokens] = torch.arange(
            region_base, region_base + num_special_tokens, device=device
        )
        patch_ids = torch.arange(patches_per_img, device=device).reshape(p_h, p_w)
        block_ids = (
            (torch.arange(p_h, device=device)[:, None] // block_size) * blocks_w
            + (torch.arange(p_w, device=device)[None, :] // block_size)
        )
        ids[
            seq_base + num_special_tokens : seq_base + num_special_tokens + patches_per_img
        ] = (region_base + num_special_tokens + block_ids.reshape(-1))[patch_ids.reshape(-1)]
    return ids


def _oracle_piecewise_abs_prepare(s, m, weights):
    """Prepare the piecewise-linear absolute-sum state for one greedy round."""
    eps = torch.finfo(s.dtype).tiny
    breakpoints = s / m.clamp_min(eps)
    sorted_breakpoints, order = torch.sort(breakpoints, dim=1)
    sorted_wm = weights.gather(1, order) * m.gather(1, order)
    sorted_ws = weights.gather(1, order) * s.gather(1, order)
    prefix_wm = torch.cumsum(sorted_wm, dim=1)
    prefix_ws = torch.cumsum(sorted_ws, dim=1)

    return (
        sorted_breakpoints,
        F.pad(prefix_wm, (1, 0)),
        F.pad(prefix_ws, (1, 0)),
        sorted_wm.sum(dim=1, keepdim=True),
        sorted_ws.sum(dim=1, keepdim=True),
    )


def _oracle_piecewise_abs_eval(state, z_values):
    """Evaluate a prepared piecewise-linear absolute-sum state."""
    (
        sorted_breakpoints,
        prefix_wm_pad,
        prefix_ws_pad,
        total_wm,
        total_ws,
    ) = state

    # searchsorted supports one sorted boundary row per query row.
    right_count = torch.searchsorted(sorted_breakpoints, z_values, right=True)
    right_count = right_count.clamp_max(sorted_breakpoints.shape[1])
    left_wm = prefix_wm_pad.gather(1, right_count)
    left_ws = prefix_ws_pad.gather(1, right_count)
    return z_values * (2.0 * left_wm - total_wm) + (total_ws - 2.0 * left_ws)


def oracle_region_attention(q, k, v, cfg, p_h, p_w, num_special_tokens, head_idx=0):
    """Three-round batched greedy region-wise Oracle attention for all heads.

    Dense attention weights are computed in FP32. For every query and head, the
    selected set starts empty. Each round scores all candidates against the
    current selected set, selects a batch of the best keys, and updates the
    region statistics before the next round. No first-frame key or special token
    is protected; special tokens are singleton regions.
    """
    if q.shape[0] != 1:
        raise ValueError("oracle_region_keys currently expects batch size 1")

    keep_ratio = float(cfg.get("keep_ratio", 1.0))
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    query_chunk_size = int(cfg.get("query_chunk_size", 1))
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    candidate_chunk_size = int(cfg.get("candidate_chunk_size", 256))
    if candidate_chunk_size <= 0:
        raise ValueError(
            f"candidate_chunk_size must be positive, got {candidate_chunk_size}"
        )
    greedy_rounds = int(cfg.get("greedy_rounds", 3))
    if greedy_rounds <= 0:
        raise ValueError(f"greedy_rounds must be positive, got {greedy_rounds}")

    _, num_heads, num_tokens, query_dim = q.shape
    keep_count = max(1, min(num_tokens, int(round(keep_ratio * num_tokens))))
    if keep_count >= num_tokens:
        return F.scaled_dot_product_attention(q, k, v)

    num_toks_per_img = p_h * p_w + num_special_tokens
    if num_toks_per_img <= num_special_tokens or num_tokens % num_toks_per_img != 0:
        raise ValueError(
            "oracle_region_keys requires tokens laid out as complete VGGT frames: "
            f"num_tokens={num_tokens}, tokens_per_frame={num_toks_per_img}"
        )
    num_imgs = num_tokens // num_toks_per_img
    block_size = int(cfg.get("block_size", 4))
    region_ids = _oracle_region_ids(
        num_imgs, p_h, p_w, block_size, num_special_tokens, q.device
    )
    num_regions = int(region_ids.max().item()) + 1

    # Keep the complete dense QK computation, but let the native model dtype
    # use Tensor Cores. Softmax and all Oracle statistics remain float32.
    qf = q[0]
    kf = k[0]
    vf = v[0].float()
    value_dim = vf.shape[-1]
    scale = 1.0 / math.sqrt(query_dim)
    output = torch.empty((num_heads, num_tokens, value_dim), device=q.device, dtype=torch.float32)
    stats_sum = torch.zeros(3, device=q.device, dtype=torch.float64)

    # Dense attention is recomputed from the current layer input for every call;
    # this is intentional because each retention ratio has its own forward pass.
    with torch.no_grad(), torch.autocast(device_type=q.device.type, enabled=False):
        for query_start in range(0, num_tokens, query_chunk_size):
            query_end = min(query_start + query_chunk_size, num_tokens)
            q_chunk = qf[:, query_start:query_end]
            logits = torch.matmul(q_chunk, kf.transpose(-1, -2)) * scale
            dense_weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
            q_count = query_end - query_start
            flat_q_count = num_heads * q_count
            flat_weights = dense_weights.reshape(flat_q_count, num_tokens)
            flat_head_ids = torch.arange(
                flat_q_count, device=q.device, dtype=torch.long
            ) // q_count
            token_indices = torch.arange(num_tokens, device=q.device, dtype=torch.long)

            dense_mass = torch.zeros((flat_q_count, num_regions), device=q.device, dtype=torch.float32)
            dense_mass.scatter_add_(
                1,
                region_ids[None, :].expand(flat_q_count, -1),
                flat_weights,
            )
            dense_num = torch.zeros(
                (flat_q_count, num_regions, value_dim), device=q.device, dtype=torch.float32
            )
            # Do not expand values to [query, key, value_dim]. At 128 frames that
            # intermediate is several GiB even though the final region statistics
            # are much smaller. Accumulate dense region numerators in key chunks.
            for candidate_start in range(0, num_tokens, candidate_chunk_size):
                candidate_end = min(candidate_start + candidate_chunk_size, num_tokens)
                candidate_indices = token_indices[candidate_start:candidate_end]
                candidate_values = vf[
                    flat_head_ids[:, None], candidate_indices[None, :], :
                ]
                candidate_weights = flat_weights[:, candidate_start:candidate_end]
                dense_num.scatter_add_(
                    1,
                    region_ids[candidate_indices][None, :, None].expand(
                        flat_q_count, -1, value_dim
                    ),
                    candidate_weights[..., None] * candidate_values,
                )
            dense_mu = dense_num / dense_mass.clamp_min(torch.finfo(torch.float32).tiny)[..., None]
            mu_norm = torch.linalg.vector_norm(dense_mu, dim=-1)

            selected_mask = torch.zeros((flat_q_count, num_tokens), device=q.device, dtype=torch.bool)
            selected_mass = torch.zeros_like(dense_mass)
            selected_num = torch.zeros_like(dense_num)
            selected_total = torch.zeros(flat_q_count, device=q.device, dtype=torch.float32)

            remaining_keys = keep_count
            for round_idx in range(greedy_rounds):
                rounds_left = greedy_rounds - round_idx
                batch_size = max(1, (remaining_keys + rounds_left - 1) // rounds_left)
                residual = selected_num - selected_mass[..., None] * dense_mu
                residual_norm = torch.linalg.vector_norm(residual, dim=-1)
                piecewise_state = _oracle_piecewise_abs_prepare(
                    selected_mass, dense_mass, mu_norm
                )
                objective_all = torch.empty_like(flat_weights)

                # Score all candidates against the current S_t, then retain
                # the best batch. The batch is applied together before the
                # next round recomputes scores from the updated sparse state.
                for candidate_start in range(0, num_tokens, candidate_chunk_size):
                    candidate_end = min(candidate_start + candidate_chunk_size, num_tokens)
                    candidate_indices = token_indices[candidate_start:candidate_end]
                    candidate_regions = region_ids[candidate_indices][None, :].expand(
                        flat_q_count, -1
                    )
                    candidate_weights = flat_weights[:, candidate_start:candidate_end]
                    candidate_total = selected_total[:, None] + candidate_weights
                    mass_before = _oracle_piecewise_abs_eval(
                        piecewise_state, candidate_total
                    )

                    candidate_mass = selected_mass.gather(1, candidate_regions) + candidate_weights
                    candidate_dense_mass = dense_mass.gather(1, candidate_regions)
                    candidate_mu_norm = mu_norm.gather(1, candidate_regions)
                    before_region_mass = selected_mass.gather(1, candidate_regions)
                    before_region_term = candidate_mu_norm * (
                        before_region_mass - candidate_total * candidate_dense_mass
                    ).abs()
                    after_region_term = candidate_mu_norm * (
                        candidate_mass - candidate_total * candidate_dense_mass
                    ).abs()
                    mass_num = mass_before - before_region_term + after_region_term

                    candidate_residual = residual.gather(
                        1,
                        candidate_regions[..., None].expand(
                            flat_q_count, candidate_end - candidate_start, value_dim
                        ),
                    )
                    candidate_mu = dense_mu.gather(
                        1,
                        candidate_regions[..., None].expand(
                            flat_q_count, candidate_end - candidate_start, value_dim
                        ),
                    )
                    candidate_values = vf[
                        flat_head_ids[:, None], candidate_indices[None, :], :
                    ]
                    after_residual = candidate_residual + candidate_weights[..., None] * (
                        candidate_values - candidate_mu
                    )
                    rep_num = (
                        residual_norm.sum(dim=1, keepdim=True)
                        - residual_norm.gather(1, candidate_regions)
                        + torch.linalg.vector_norm(after_residual, dim=-1)
                    )

                    objective = (mass_num + rep_num) / candidate_total.clamp_min(
                        torch.finfo(torch.float32).tiny
                    )
                    objective = objective.masked_fill(
                        selected_mask[:, candidate_start:candidate_end], math.inf
                    )
                    objective_all[:, candidate_start:candidate_end] = objective

                # One global top-k per round, after all candidate chunks have
                # been scored. This avoids repeatedly top-k'ing the growing
                # shortlist once per candidate chunk.
                _, chosen = torch.topk(
                    objective_all,
                    k=batch_size,
                    dim=1,
                    largest=False,
                    sorted=False,
                )
                chosen_mass = flat_weights.gather(1, chosen)
                chosen_region = region_ids[chosen]
                chosen_value = vf[flat_head_ids[:, None], chosen]

                selected_mask.scatter_(1, chosen, True)
                selected_mass.scatter_add_(1, chosen_region, chosen_mass)
                selected_num.scatter_add_(
                    1,
                    chosen_region[..., None].expand(-1, -1, value_dim),
                    chosen_mass[..., None] * chosen_value,
                )
                selected_total += chosen_mass.sum(dim=1)
                remaining_keys -= batch_size
                if remaining_keys <= 0:
                    break

            sparse_output = selected_num.sum(dim=1) / selected_total[:, None].clamp_min(
                torch.finfo(torch.float32).tiny
            )
            output[:, query_start:query_end] = sparse_output.reshape(
                num_heads, q_count, value_dim
            )

            sparse_mass = selected_mass / selected_total[:, None].clamp_min(
                torch.finfo(torch.float32).tiny
            )
            sparse_mu = selected_num / selected_mass.clamp_min(torch.finfo(torch.float32).tiny)[..., None]
            mass_error = (
                (sparse_mass - dense_mass).abs() * mu_norm
            ).sum(dim=1)
            rep_error = (
                sparse_mass * torch.linalg.vector_norm(sparse_mu - dense_mu, dim=-1)
            ).sum(dim=1)
            # dense_num is already the dense weighted Value sum grouped by
            # region, so a second pass over all keys is unnecessary.
            dense_output = dense_num.sum(dim=1)
            output_error = torch.linalg.vector_norm(sparse_output - dense_output, dim=-1) / (
                torch.linalg.vector_norm(dense_output, dim=-1) + 1e-6
            )
            stats_sum += torch.stack(
                [mass_error.sum(), rep_error.sum(), output_error.sum()]
            ).to(torch.float64)

    oracle_stats = cfg.get("_oracle_stats")
    if oracle_stats is not None:
        oracle_stats["mass_error_sum"] += float(stats_sum[0].item())
        oracle_stats["rep_error_sum"] += float(stats_sum[1].item())
        oracle_stats["output_error_sum"] += float(stats_sum[2].item())
        oracle_stats["queries"] += num_heads * num_tokens
        oracle_stats["selected_keys"] += num_heads * num_tokens * keep_count

    return output.to(dtype=v.dtype).view(1, num_heads, num_tokens, value_dim)


def get_separate_indices(num_imgs, num_toks_per_img, num_special_tokens):
    """
    Returns two 1D tensors of indices for special tokens and image tokens, respectively.
    """
    base = torch.arange(num_imgs) * num_toks_per_img                # [num_imgs]

    # special: base + [0..num_special_tokens-1]
    spec_local = torch.arange(num_special_tokens)                   # [S]
    spec_idx = (base[:, None] + spec_local[None, :]).reshape(-1)    # [num_imgs*S]

    # image: base + [S..num_toks_per_img-1]
    img_local = torch.arange(num_special_tokens, num_toks_per_img)  # [T-S]
    img_idx = (base[:, None] + img_local[None, :]).reshape(-1)      # [num_imgs*(T-S)]

    return spec_idx, img_idx


def broadcast_anchorframe_attention(q, k, v, num_toks_per_img):
    """
    Copy the attention map of the first frame to all frames.
    """
    # q0, k0: (B, H, N, D)
    q0 = q[:, :, :num_toks_per_img, :]
    k0 = k[:, :, :num_toks_per_img, :]
    v = v.reshape(v.shape[0], v.shape[1], -1, num_toks_per_img, v.shape[-1])  # (B, H, S, N, D)

    # attn: (B, H, N, N)
    attn = torch.matmul(q0, k0.transpose(-1, -2)) / (q0.size(-1) ** 0.5)
    attn = torch.softmax(attn, dim=-1)

    # attn: (B, H, 1, N, N)
    # v:    (B, H, S, N, D)
    attn = attn.unsqueeze(2) 

    # output: (B, H, S, N, D)
    output = torch.matmul(attn, v)

    # (B, H, SN, D)
    return output.flatten(2, 3)


def all_to_first(q, k, v, num_toks_per_img):
    """
    Each query attends to all tokens in the first image.
    """
    k = k[:, :, :num_toks_per_img, :]
    v = v[:, :, :num_toks_per_img, :]
    return F.scaled_dot_product_attention(q, k, v)


def q_probe_topk_attention(q, k, v, topk, q_sample_ratio=1, indices=None):
    """
    Using a small subset of queries to probe the keys and select top-k keys for all queries.
    """
    assert q.shape[0] == 1 and q.shape[1] == 1
    assert k.shape[0] == 1 and k.shape[1] == 1
    assert v.shape[0] == 1 and v.shape[1] == 1
    _, _, N, D = q.shape

    # 0) Sample queries
    q_sel = q
    if q_sample_ratio < 1:
        num_sampled_q = max(1, int(q.size(2) * q_sample_ratio))
        q_idx = torch.linspace(0, q.size(2)-1, steps=num_sampled_q).long()
        q_sel = q[:, :, q_idx, :]             # (1, 1, num_sampled_q, D)

    # 1) Compute q_bar using ALL queries
    q_bar = q_sel.mean(dim=2, keepdim=True)   # (1, 1, 1, D)

    # 2) Compute scores for all keys
    # use float32 for stability
    scores = torch.matmul(
        q_bar.to(torch.float32), k.transpose(-1, -2).to(torch.float32)
    ) / math.sqrt(D)                          # (1, 1, 1, N)
    scores = scores.squeeze()                 # (N,)

    # 3) Select topk keys
    topk_idx = torch.topk(scores, k=topk, dim=-1, sorted=False).indices  # (topk,)

    # 3.5) If there are additional indices to attend to, combine them with the topk_idx
    if indices is not None:
        indices = indices.to(device=topk_idx.device, dtype=torch.long)
        combined_idx = torch.cat([topk_idx, indices])
        topk_idx = torch.unique(combined_idx)
    else:
        topk_idx = torch.sort(topk_idx).values

    # 4) Direct advanced indexing (NO gather)
    k_topk = k[:, :, topk_idx, :]             # (1, 1, topk, D)
    v_topk = v[:, :, topk_idx, :]             # (1, 1, topk, Dv)

    # 5) Final attention over reduced KV
    out = F.scaled_dot_product_attention(q, k_topk, v_topk)
    return out


def dino_topk_attention(q, k, v, spec_tok_idx, img_tok_idx, est_topk_idx, dino_sp2sp_stride=1):
    """
    Separate special and image tokens: special tokens only attend to each other / all tokens,
    while image tokens attend to a subset of keys/values.
    Selected keys/values of each query are given by est_topk_idx.
    """
    assert q.shape[0] == 1 and q.shape[1] == 1

    # 1) Process special tokens (attend to all tokens)
    q_spec = q[:, :, spec_tok_idx, :]  # [1, 1, num_special_tokens*num_imgs, D]
    if dino_sp2sp_stride == 1:
        k_spec = k  # attend to all keys
        v_spec = v  # attend to all values
    else:
        k_spec = k[:, :, ::dino_sp2sp_stride, :]
        v_spec = v[:, :, ::dino_sp2sp_stride, :]

    # NOTE: force to use flash or mem-eff SDPA
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        out_spec = F.scaled_dot_product_attention(q_spec, k_spec, v_spec)  # [1, 1, num_special_tokens*num_imgs, Dv]

    # 2) Process image tokens (attend only to estimated top-k tokens)
    D = q.shape[-1]
    Dv = v.shape[-1]
    num_img_tokens = img_tok_idx.shape[0]
    q_img = q[:, :, img_tok_idx, :].reshape(-1, D)   # [num_img_tokens, D]
    k_img = k[:, :, img_tok_idx, :].reshape(-1, D)   # [num_img_tokens, D]
    v_img = v[:, :, img_tok_idx, :].reshape(-1, Dv)  # [num_img_tokens, Dv]

    # NOTE: `fused_sparse_topk_attention` fails when topk is large (e.g., 2048)
    topk = est_topk_idx.shape[-1]
    if topk <= 2048:
        out_img_flat = fused_sparse_topk_attention(q_img, k_img, v_img, est_topk_idx)
    else:
        out_img_flat = fused_sparse_topk_attention_2(q_img, k_img, v_img, est_topk_idx)
    out_img = out_img_flat.reshape(1, 1, num_img_tokens, Dv)

    # 3) Combine special and image token outputs
    out = torch.empty_like(v)
    out[:, :, spec_tok_idx, :] = out_spec
    out[:, :, img_tok_idx, :] = out_img

    return out


def global_to_frame_attention(q, k, v, num_toks_per_img):
    """
    Attend only to tokens within the same image.
    """
    B, H, num_toks, D = q.shape
    _, _, _, Dv = v.shape
    num_imgs = num_toks // num_toks_per_img

    q = q.view(B, H, num_imgs, num_toks_per_img, D)  # [B, 1, num_imgs, toks_per_img, head_dim]
    k = k.view(B, H, num_imgs, num_toks_per_img, D)  # [B, 1, num_imgs, toks_per_img, head_dim]
    v = v.view(B, H, num_imgs, num_toks_per_img, Dv) # [B, 1, num_imgs, toks_per_img, head_dim]
    
    q = q.permute(0, 2, 1, 3, 4).view(-1, 1, num_toks_per_img, D)  # [B*num_imgs, 1, toks_per_img, head_dim]
    k = k.permute(0, 2, 1, 3, 4).view(-1, 1, num_toks_per_img, D)  # [B*num_imgs, 1, toks_per_img, head_dim]
    v = v.permute(0, 2, 1, 3, 4).view(-1, 1, num_toks_per_img, Dv) # [B*num_imgs, 1, toks_per_img, head_dim]

    out = F.scaled_dot_product_attention(q, k, v)
    out = out.view(B, num_imgs, H, num_toks_per_img, Dv).permute(0, 2, 1, 3, 4).view(B, H, num_toks, Dv)  # [B, 1, N, head_dim]

    return out


def stride_attention(q, k, v, stride, start_pos=0, indices=None):
    """
    Each query attends to strided keys/values, optionally combined with additional indices. 
    """
    if indices is None:
        k_strided = k[:, :, start_pos::stride, :]
        v_strided = v[:, :, start_pos::stride, :]
        return F.scaled_dot_product_attention(q, k_strided, v_strided)
    
    _, _, N, _ = k.shape
    strided_idx = torch.arange(start_pos, N, step=stride, device=k.device, dtype=torch.long)
    indices = indices.to(device=k.device, dtype=torch.long)
    combined_idx = torch.cat([strided_idx, indices])
    final_idx = torch.unique(combined_idx)

    k_strided = k[:, :, final_idx, :]
    v_strided = v[:, :, final_idx, :]
    return F.scaled_dot_product_attention(q, k_strided, v_strided)

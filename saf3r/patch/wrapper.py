import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_attention import (
    estimate_block3x3_violation_kept_pairs,
    estimate_hier_split_prune_kept_pairs,
    sparse_attention,
    get_separate_indices,
)
from ..triton.fused_qk_topk import compute_qk_topk_indices_fused


class BaseHeadwiseSparseAttentionWrapper(nn.Module):
    """
    A base sparse-attention wrapper for monkey-patching arbitrary attention modules.
    """
    def __init__(self, orig_module, num_special_tokens, num_heads, **kwargs):
        super().__init__()
        self.orig_module = orig_module
        self.num_special_tokens = num_special_tokens
        self.num_heads = num_heads
        self.layer_idx = kwargs.get("layer_idx", 0)

        # lazy init, to be updated before each forward pass
        self.patch_width = kwargs.get("patch_width", None)
        self.patch_height = kwargs.get("patch_height", None)
        self.est_topk_idx = torch.empty(0) # init as empty, to be updated as [SP, topk]

        self._oracle_stats = {
            "mass_error_sum": 0.0,
            "rep_error_sum": 0.0,
            "output_error_sum": 0.0,
            "queries": 0,
            "selected_keys": 0,
        }

        # for lazy topk computation
        self.lazy_dino_topk = kwargs.get("lazy_dino_topk", False)
        self.dino_topk = kwargs.get("dino_topk", 4)
        self.first_dino_flag = False  # init as False
        self.reset_sparsity_stats()
        
        # parse config
        cfg = kwargs.get("config", [])
        self.head_cfg = [{"mode": "full"} for _ in range(self.num_heads)]  # default to "full"

        # parse group configs and update head configs
        for group_cfg in cfg:
            group_cfg = dict(group_cfg)
            group_head_ids = group_cfg["head_ids"]
            group_cfg.pop("head_ids")
            for head_id in group_head_ids:
                # sanity check: each head can only be assigned to one pattern
                if self.head_cfg[head_id]["mode"] != "full":
                    raise ValueError(f"Head {head_id} assigned to multiple patterns!")
                self.head_cfg[head_id].update(group_cfg)
                self.head_cfg[head_id]["layer_idx"] = self.layer_idx
                if self.head_cfg[head_id].get("mode") == "oracle_region_keys":
                    self.head_cfg[head_id]["_oracle_stats"] = self._oracle_stats

    def reset_sparsity_stats(self):
        self._sparsity_stats = {
            "dense_qk_pairs": 0,
            "kept_qk_pairs": 0,
            "calls": 0,
            "heads": 0,
            "by_mode": {},
            "oracle": {
                "mass_error_sum": 0.0,
                "rep_error_sum": 0.0,
                "output_error_sum": 0.0,
                "queries": 0,
                "selected_keys": 0,
            },
        }
        self._oracle_stats = self._sparsity_stats["oracle"]
        for cfg in getattr(self, "head_cfg", []):
            if cfg.get("mode") == "oracle_region_keys":
                cfg["_oracle_stats"] = self._oracle_stats

    def get_sparsity_stats(self):
        return self._sparsity_stats

    def _record_sparsity(self, q, cfg, head_idx=0):
        mode = cfg.get("mode", "full")
        dense_pairs = int(q.shape[0] * q.shape[1] * q.shape[2] * q.shape[2])
        kept_pairs = int(self._estimate_kept_qk_pairs(q, cfg, head_idx))

        self._sparsity_stats["dense_qk_pairs"] += dense_pairs
        self._sparsity_stats["kept_qk_pairs"] += kept_pairs
        self._sparsity_stats["calls"] += 1
        self._sparsity_stats["heads"] += int(q.shape[1])

        mode_stats = self._sparsity_stats["by_mode"].setdefault(
            mode,
            {
                "dense_qk_pairs": 0,
                "kept_qk_pairs": 0,
                "calls": 0,
                "heads": 0,
            },
        )
        mode_stats["dense_qk_pairs"] += dense_pairs
        mode_stats["kept_qk_pairs"] += kept_pairs
        mode_stats["calls"] += 1
        mode_stats["heads"] += int(q.shape[1])

    def _estimate_kept_qk_pairs(self, q, cfg, head_idx=0):
        batch_size, num_heads, num_toks, _ = q.shape
        mode = cfg.get("mode", "full")

        if mode == "full":
            return batch_size * num_heads * num_toks * num_toks
        if mode == "skip":
            return 0

        num_toks_per_img = self.patch_height * self.patch_width + self.num_special_tokens
        num_imgs = num_toks // num_toks_per_img
        num_special_toks = num_imgs * self.num_special_tokens
        num_img_toks = num_toks - num_special_toks

        if mode == "broadcast_first":
            return batch_size * num_heads * num_toks_per_img * num_toks_per_img

        if mode in {"all_to_first", "global_to_frame"}:
            return batch_size * num_heads * num_toks * num_toks_per_img

        if mode == "q_probe_topk":
            if "stride" in cfg:
                key_count = max(1, num_toks // int(cfg["stride"]))
            else:
                key_count = min(int(cfg.get("topk", 1024)), num_toks)
            if cfg.get("include_as", False):
                key_count = min(
                    num_toks,
                    key_count + self._anchor_special_union_count(num_toks, num_toks_per_img, num_imgs),
                )
            return batch_size * num_heads * num_toks * key_count

        if mode == "dino_topk":
            topk = int(self.est_topk_idx.shape[-1]) if self.est_topk_idx.numel() > 0 else int(cfg.get("topk", num_img_toks))
            topk = min(topk, num_img_toks)
            sp2sp_stride = int(cfg.get("dino_sp2sp_stride", 1))
            special_key_count = (num_toks + sp2sp_stride - 1) // sp2sp_stride
            pairs_per_head = num_special_toks * special_key_count + num_img_toks * topk
            return batch_size * num_heads * pairs_per_head

        if mode == "stride":
            stride = int(cfg.get("stride", 4))
            start_pos = int(head_idx) % stride if cfg.get("shift", False) else 0
            key_count = (num_toks - start_pos + stride - 1) // stride
            if cfg.get("include_as", True):
                key_count = self._stride_anchor_special_union_count(
                    num_toks, num_toks_per_img, num_imgs, stride, start_pos
                )
            return batch_size * num_heads * num_toks * key_count

        if mode == "random_keys":
            keep_ratio = float(cfg.get("keep_ratio", 1.0))
            image_key_count = max(1, min(num_img_toks, int(round(num_img_toks * keep_ratio))))
            key_count = image_key_count + (num_special_toks if cfg.get("include_special_keys", True) else 0)
            pairs_per_head = num_special_toks * num_toks + num_img_toks * key_count
            return batch_size * num_heads * pairs_per_head

        if mode == "block2x2_keys":
            if cfg.get("selection", "all") == "all":
                return batch_size * num_heads * num_toks * num_toks
            block_size = int(cfg.get("block_size", 2))
            blocks_per_img = ((self.patch_height + block_size - 1) // block_size) * (
                (self.patch_width + block_size - 1) // block_size
            )
            image_key_count = num_imgs * blocks_per_img
            key_count = image_key_count + (num_special_toks if cfg.get("include_special_keys", True) else 0)
            pairs_per_head = num_special_toks * num_toks + num_img_toks * key_count
            return batch_size * num_heads * pairs_per_head

        if mode == "ranked_middle_keys":
            final_ratio = float(cfg.get("final_ratio", 0.2))
            image_key_count = max(1, min(num_img_toks, int(round(num_img_toks * final_ratio))))
            key_count = image_key_count + (num_special_toks if cfg.get("include_special_keys", True) else 0)
            pairs_per_head = num_special_toks * num_toks + num_img_toks * key_count
            return batch_size * num_heads * pairs_per_head

        if mode == "block_novelty_keys":
            keep_ratio = float(cfg.get("keep_ratio", 0.2))
            mandatory_total = num_special_toks + self.patch_height * self.patch_width
            key_count = max(mandatory_total, min(num_toks, int(round(num_toks * keep_ratio))))
            pairs_per_head = num_special_toks * num_toks + num_img_toks * key_count
            return batch_size * num_heads * pairs_per_head

        if mode == "block_coverage_redundancy_keys":
            keep_ratio = float(cfg.get("keep_ratio", 0.1))
            block_size = int(cfg.get("block_size", 2))
            blocks_per_img = ((self.patch_height + block_size - 1) // block_size) * (
                (self.patch_width + block_size - 1) // block_size
            )
            num_blocks = num_imgs * blocks_per_img
            final_key_count = max(1, min(num_img_toks, int(round(num_img_toks * keep_ratio))))
            image_key_count = min(num_blocks, final_key_count)
            key_count = image_key_count + (num_special_toks if cfg.get("include_special_keys", False) else 0)
            pairs_per_head = num_special_toks * num_toks + num_img_toks * key_count
            return batch_size * num_heads * pairs_per_head

        if mode == "block3x3_violation_keys":
            pairs_per_head = estimate_block3x3_violation_kept_pairs(
                cfg,
                num_toks,
                num_toks_per_img,
                num_imgs,
                num_special_toks,
                num_img_toks,
                self.num_special_tokens,
                p_h=self.patch_height,
                p_w=self.patch_width,
                head_idx=head_idx,
                est_topk_idx=self.est_topk_idx,
            )
            return batch_size * num_heads * pairs_per_head

        if mode == "hier_split_prune_keys":
            pairs_per_head = estimate_hier_split_prune_kept_pairs(
                cfg,
                num_toks,
                num_toks_per_img,
                num_imgs,
                num_special_toks,
                num_img_toks,
            )
            return batch_size * num_heads * pairs_per_head

        if mode == "oracle_region_keys":
            keep_ratio = float(cfg.get("keep_ratio", 1.0))
            key_count = max(1, min(num_toks, int(round(num_toks * keep_ratio))))
            return batch_size * num_heads * num_toks * key_count

        if mode in {"fixed_keys", "query_topk_keys"}:
            keep_ratio = float(cfg.get("keep_ratio", 1.0))
            key_count = max(1, min(num_toks, int(round(num_toks * keep_ratio))))
            return batch_size * num_heads * num_toks * key_count

        return batch_size * num_heads * num_toks * num_toks

    def _anchor_special_union_count(self, num_toks, num_toks_per_img, num_imgs):
        indices = set(range(num_toks_per_img))
        for img_idx in range(num_imgs):
            base = img_idx * num_toks_per_img
            indices.update(range(base, min(base + self.num_special_tokens, num_toks)))
        return len(indices)

    def _stride_anchor_special_union_count(self, num_toks, num_toks_per_img, num_imgs, stride, start_pos):
        indices = set(range(start_pos, num_toks, stride))
        indices.update(range(num_toks_per_img))
        for img_idx in range(num_imgs):
            base = img_idx * num_toks_per_img
            indices.update(range(base, min(base + self.num_special_tokens, num_toks)))
        return len(indices)

    def sparse_attention(self, q, k, v, cfg, head_idx=0):
        # lazy topk: compute and store QK top-k indices from the first DINO TopK head
        if (
            cfg["mode"] == "dino_topk" and
            self.lazy_dino_topk and cfg.get("is_first_dino", False) and not self.first_dino_flag
        ):
            print(f"Computing QK TopK indices (K={self.dino_topk} each frame) for subsequent DINO TopK heads ...")
            self.first_dino_flag = True  # avoid recomputing for subsequent DINO TopK heads at the same block

            # remove special tokens and extract image tokens
            num_toks = q.shape[2]
            num_toks_per_img = self.patch_width * self.patch_height + self.num_special_tokens
            num_imgs = num_toks // num_toks_per_img
            _, img_idx = get_separate_indices(num_imgs, num_toks_per_img, self.num_special_tokens)
            q_img = q.squeeze(1)[:, img_idx, :]  # (B, N_img, D)
            k_img = k.squeeze(1)[:, img_idx, :]  # (B, N_img, D)

            # compute and update indices
            self.est_topk_idx = compute_qk_topk_indices_fused(
                q_img, k_img, self.dino_topk,
                num_toks_per_img=(self.patch_width * self.patch_height),
            ).squeeze()  # [SP, topk]

        # compute sparse attention
        self._record_sparsity(q, cfg, head_idx)
        return sparse_attention(
            q, k, v, self.est_topk_idx, cfg, self.patch_height, self.patch_width,
            self.num_special_tokens, head_idx,
        )

    def compute_multihead_attention(self, q, k, v):
        batched_cfg = self._get_batched_sparse_cfg()
        if batched_cfg is not None:
            return self.sparse_attention(q, k, v, batched_cfg, head_idx=0)

        attn_outs = []
        for h in range(self.num_heads):
            q_h, k_h, v_h = q[:, h:h+1], k[:, h:h+1], v[:, h:h+1]  # [B, 1, N, D]

            if self.head_cfg[h]["mode"] == "full":
                self._record_sparsity(q_h, self.head_cfg[h], head_idx=h)
                attn_out = F.scaled_dot_product_attention(q_h, k_h, v_h)  # [B, 1, N, D]
            else:
                attn_out = self.sparse_attention(q_h, k_h, v_h, self.head_cfg[h], head_idx=h)  # [B, 1, N, D]

            attn_outs.append(attn_out)

        return torch.cat(attn_outs, dim=1)  # [B, H, N, D]

    def _get_batched_sparse_cfg(self):
        first_cfg = self.head_cfg[0]
        if first_cfg.get("mode") not in {
            "random_keys",
            "block2x2_keys",
            "block_novelty_keys",
            "block_coverage_redundancy_keys",
            "block3x3_violation_keys",
            "hier_split_prune_keys",
            "ranked_middle_keys",
            "fixed_keys",
            "query_topk_keys",
            "oracle_region_keys",
        }:
            return None

        for cfg in self.head_cfg[1:]:
            if cfg != first_cfg:
                return None

        return first_cfg
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a base wrapper and should not be used directly.")


def reset_sparsity_counters(model):
    for module in model.modules():
        if isinstance(module, BaseHeadwiseSparseAttentionWrapper):
            module.reset_sparsity_stats()


def collect_sparsity_counters(model):
    total = {
        "dense_qk_pairs": 0,
        "kept_qk_pairs": 0,
        "calls": 0,
        "heads": 0,
        "by_mode": {},
        "oracle": {
            "mass_error_sum": 0.0,
            "rep_error_sum": 0.0,
            "output_error_sum": 0.0,
            "queries": 0,
            "selected_keys": 0,
        },
    }
    for module in model.modules():
        if not isinstance(module, BaseHeadwiseSparseAttentionWrapper):
            continue
        stats = module.get_sparsity_stats()
        total["dense_qk_pairs"] += int(stats["dense_qk_pairs"])
        total["kept_qk_pairs"] += int(stats["kept_qk_pairs"])
        total["calls"] += int(stats["calls"])
        total["heads"] += int(stats["heads"])
        for key in total["oracle"]:
            total["oracle"][key] += stats.get("oracle", {}).get(key, 0)
        for mode, mode_stats in stats["by_mode"].items():
            dst = total["by_mode"].setdefault(
                mode,
                {
                    "dense_qk_pairs": 0,
                    "kept_qk_pairs": 0,
                    "calls": 0,
                    "heads": 0,
                },
            )
            dst["dense_qk_pairs"] += int(mode_stats["dense_qk_pairs"])
            dst["kept_qk_pairs"] += int(mode_stats["kept_qk_pairs"])
            dst["calls"] += int(mode_stats["calls"])
            dst["heads"] += int(mode_stats["heads"])

    dense_pairs = total["dense_qk_pairs"]
    kept_pairs = total["kept_qk_pairs"]
    total["keep_rate"] = kept_pairs / dense_pairs if dense_pairs else 1.0
    total["equivalent_sparsity"] = 1.0 - total["keep_rate"]
    queries = total["oracle"]["queries"]
    if queries:
        total["oracle"]["mean_mass_error"] = total["oracle"]["mass_error_sum"] / queries
        total["oracle"]["mean_rep_error"] = total["oracle"]["rep_error_sum"] / queries
        total["oracle"]["mean_output_error"] = total["oracle"]["output_error_sum"] / queries
    else:
        total["oracle"]["mean_mass_error"] = 0.0
        total["oracle"]["mean_rep_error"] = 0.0
        total["oracle"]["mean_output_error"] = 0.0
    for mode_stats in total["by_mode"].values():
        dense = mode_stats["dense_qk_pairs"]
        kept = mode_stats["kept_qk_pairs"]
        mode_stats["keep_rate"] = kept / dense if dense else 1.0
        mode_stats["equivalent_sparsity"] = 1.0 - mode_stats["keep_rate"]
    return total


class BaseSparseAttentionProfiler(nn.Module):
    """
    A profiler that selects the best sparse-attention pattern for each head by exhaustively
    searching a predefined pattern set and comparing each output against full attention.

    NOTE: This profiler is for analysis only and is not optimized for speed.
    It can be much slower than the original attention layer because it performs
    multiple attention computations per head.
    """
    def __init__(self, orig_module, num_special_tokens, num_heads, **kwargs):
        super().__init__()
        self.orig_module = orig_module
        self.num_special_tokens = num_special_tokens
        self.num_heads = num_heads

        # lazy init, to be updated before each forward pass
        self.patch_width = kwargs.get("patch_width", None)
        self.patch_height = kwargs.get("patch_height", None)
        self.est_topk_idx = torch.empty(0) # [SP, topk]

        self.verbose = kwargs.get("verbose", True)
        self.layer_idx = kwargs.get("layer_idx", None)
        self.profile_metric = kwargs.get("profile_metric", "cmp_mse")

        # candidate sparse patterns under given budget
        self.upper_stride = kwargs.get("upper_stride", 128)
        self.lower_stride = kwargs.get("lower_stride", 4)
        self.switch_layer = kwargs.get("switch_layer", 9)

        # NOTE: by default, we will use all available sparse attention patterns for profiling
        self.sparse_patterns = kwargs.get(
            "sparse_patterns",
            ["broadcast_first", "all_to_first", "q_probe_topk", "dino_topk", "global_to_frame", "stride"]
        )

        # accumulate per-head errors across multiple scenes in the calibration dataset
        heads = kwargs.get("heads", list(range(self.num_heads)))
        self.head_errors = {
            h: {} for h in range(self.num_heads) if h in heads
        }

    def sparse_attention(self, q, k, v, cfg, head_idx=0):
        return sparse_attention(
            q, k, v, self.est_topk_idx, cfg, self.patch_height, self.patch_width,
            self.num_special_tokens, head_idx,
        )

    def profile_attention_cmp_mse(self, q, k, v, head_idx):
        # compute full attention
        fa_out = F.scaled_dot_product_attention(q, k, v)  # [B, 1, N, D]

        # define cfgs
        if int(self.layer_idx) < self.switch_layer:
            base_stride = self.upper_stride
        else:
            base_stride = self.lower_stride

        cfgs = [
            # base cfg
            {"mode": "stride", "stride": base_stride, "include_as": True, "is_base": True},
            {"mode": "broadcast_first", "is_base": False},
            {"mode": "all_to_first", "is_base": False},
            {"mode": "global_to_frame", "is_base": False}
        ]

        # stride
        for mul in range(1, 6):
            stride = base_stride * (2 ** mul)
            cfgs.append({"mode": "stride", "stride": stride, "is_base": False})

        # q probe topk with different K
        for mul in range(6):
            stride = base_stride * (2 ** mul)
            cfgs.append({"mode": "q_probe_topk", "stride": stride, "is_base": False})

        # dino topk
        # TODO: this `topk` is dummy and will never be used, to be removed
        cfgs.append({"mode": "dino_topk", "topk": 32768, "dino_sp2sp_stride": 1, "is_base": False})

        # loop over different sparse attention configs and find the best one
        for i, cfg in enumerate(cfgs):
            # compute sparse attention output
            sa_out = self.sparse_attention(q, k, v, cfg, head_idx)  # [B, 1, N, D]

            # compute NMSE with full attention output
            mse = F.mse_loss(sa_out, fa_out)
            energy = torch.mean(fa_out ** 2) + 1e-10
            mse = (mse / energy).item()

            # record
            if i not in self.head_errors[head_idx]:
                self.head_errors[head_idx][i] = {"cfg": cfg, "mse": [mse]}
            else:
                self.head_errors[head_idx][i]["mse"].append(mse)

            if self.verbose:
                print(f"[L{self.layer_idx:>2} | H{head_idx:>2}] config {cfg}, MSE={mse:.6f}")

        # output
        return fa_out

    def profile_multihead_attention(self, q, k, v):
        attn_outs = []
        for h in range(self.num_heads):
            q_h, k_h, v_h = q[:, h:h+1], k[:, h:h+1], v[:, h:h+1]  # [B, 1, N, D]
            attn_out = self.profile_attention_cmp_mse(q_h, k_h, v_h, head_idx=h)
            attn_outs.append(attn_out)
        return torch.cat(attn_outs, dim=1)  # [B, H, N, D]

    def forward(self, *args, **kwargs):
        raise NotImplementedError("This is a base profiler and should not be used directly.")

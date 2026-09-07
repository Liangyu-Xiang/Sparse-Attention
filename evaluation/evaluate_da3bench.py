"""
Evaluate F3R models on DA3-Bench across multiple tasks and datasets.

Supported evaluation tasks:
    - Camera pose estimation
    - Video depth estimation
    - 3D point-cloud reconstruction (unposed)

NOTE: Camera pose and video depth are evaluated as separate tasks, whereas 3D point cloud
reconstruction effectively combines both, as its performance depends on both pose and depth.
"""
import argparse
import os
from addict import Dict
from hydra import initialize, compose

import open3d as o3d
import numpy as np
import torch

from benchmarks.da3.bench.registries import MV_REGISTRY
from benchmarks.da3.bench.utils import evaluate_3d_reconstruction, sample_points_from_mesh
from benchmarks.da3.bench.utils import compute_pose
from benchmarks.da3.da3bench_utils import sample_frames, fuse3d, evaluate_reconstruction_dtu
from benchmarks.da3.geometry import as_homogeneous
from benchmarks.pi3.eval_videodepth import evaluate_video_depth
from utils import get_indicator
from saf3r.utils.data_utils import load_depths
from saf3r.utils.model_utils import build_model, infer_model


AVAILABLE_DATASETS = ["7scenes", "eth3d", "scannetpp", "hiroom", "dtu64", "dtu"]
RECON_TASKS = {"recon_unposed"}
METRIC_DISPLAY_NAMES = {
    "auc30": "AUC@30",
    "auc15": "AUC@15",
    "auc05": "AUC@5",
    "auc03": "AUC@3",
    "acc": "Acc",
    "comp": "Comp",
    "overall": "CD",
    "precision": "Prec",
    "recall": "Rec",
    "fscore": "F1",
}


def apply_dataset_config(dataset, data_cfg):
    data_root = data_cfg.get("data_root", None)
    if data_root is not None:
        dataset.data_root = data_root


def validate_scene_requirements(dataset, dataset_name, scene_name, eval_tasks):
    if dataset_name.lower() != "eth3d":
        return

    scene_dir = os.path.join(dataset.data_root, scene_name)
    required_paths = [
        scene_dir,
        os.path.join(scene_dir, "dslr_calibration_jpg", "cameras.txt"),
        os.path.join(scene_dir, "dslr_calibration_jpg", "images.txt"),
        os.path.join(scene_dir, "images"),
    ]

    if "depth" in eval_tasks:
        required_paths.append(os.path.join(scene_dir, "ground_truth_depth", "dslr_images"))

    if RECON_TASKS & set(eval_tasks):
        required_paths.append(os.path.join(scene_dir, "combined_mesh.ply"))

    missing = [path for path in required_paths if not os.path.exists(path)]
    if missing:
        missing_str = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"Dataset [{dataset_name}] scene [{scene_name}] is missing required files:\n"
            f"  - {missing_str}"
        )


def validate_dataset_requirements(dataset, dataset_name, all_scenes, eval_tasks):
    for scene_name in all_scenes:
        validate_scene_requirements(dataset, dataset_name, scene_name, eval_tasks)


def evaluate(dataset, gt_data, pred_data, eval_tasks):
    all_metrics = Dict()

    # Evaluate camera pose estimation
    if "pose" in eval_tasks:
        metrics = compute_pose(
            torch.from_numpy(as_homogeneous(pred_data.extrinsics)),
            torch.from_numpy(as_homogeneous(gt_data.extrinsics)),
        )
        all_metrics.pose = metrics

    # Evaluate depth estimation
    if "depth" in eval_tasks:
        metrics = evaluate_video_depth(
            pred_data.depth, load_depths(gt_data), gt_data.dataset_name
        )
        # remove other metrics
        del metrics["Sq Rel"]
        del metrics["RMSE"]
        del metrics["Log RMSE"]
        del metrics["δ < 1."]
        del metrics["δ < 1.25^2"]
        del metrics["δ < 1.25^3"]
        del metrics["valid_pixels"]
        all_metrics.depth = metrics
 
    # Evaluate 3D reconstruction
    if "recon_unposed" in eval_tasks:
        pred_mesh, pred_pcd, aligned_intrinsics, aligned_extrinsics = fuse3d(
            dataset, gt_data.scene_name, gt_data, pred_data, mode="recon_unposed"
        )

        # Update intrinsics and extrinsics after global alignment
        pred_data.aligned_intrinsics = aligned_intrinsics
        pred_data.aligned_extrinsics = aligned_extrinsics
        pred_data.mesh = pred_mesh
        pred_data.pcd = pred_pcd

        # Compute metrics
        if gt_data.dataset_name == "dtu":
            # NOTE: DTU adopts a separate evaluation protocol
            result = evaluate_reconstruction_dtu(
                dataset, pred_data.pcd, gt_data.pcd,
                mask_file=gt_data.aux.mask_file, plane_file=gt_data.aux.plane_file,
                use_gpu=True
            )
            metrics = Dict({"comp": result[0], "acc": result[1], "overall": result[2]})
        else:
            metrics = evaluate_3d_reconstruction(
                pred_data.pcd,
                gt_data.pcd,
                threshold=dataset.eval_threshold,
                down_sample=dataset.down_sample,
            )
        all_metrics.recon_unposed = metrics


    return all_metrics


def load_scene_data(
    dataset, dataset_name, scene_name, max_frames, sample_mode="uniform",
    seq_multiple=1, seed=0, eval_tasks=None,
):
    # load scene
    scene_data = dataset.get_data(scene_name)

    # sample frames
    frame_ids, scene_data = sample_frames(
        scene_data, scene_name, max_frames, indices=None, sample_mode=sample_mode,
        seq_multiple=seq_multiple,  # force the sampled sequence length to be a multiple of seq_multiple if specified
        seed=seed
    )

    # set GT meta for evaluation
    gt_data = Dict({
        "dataset_name": dataset_name,
        "scene_name": scene_name,
        "extrinsics": scene_data.extrinsics,
        "intrinsics": scene_data.intrinsics,
        "image_files": scene_data.image_files,
        "aux": Dict({
            "gt_mesh_path": scene_data.aux.gt_mesh_path,
            "gt_pcd_path": scene_data.aux.gt_pcd_path,
            "gt_depth_files": scene_data.aux.gt_depth_files,
            "aliasing_mask_files": scene_data.aux.aliasing_mask_files,
            "mask_file": scene_data.aux.mask_file,
            "plane_file": scene_data.aux.plane_file,
        }),
    })

    # get ground truth point-cloud for reconstruction evaluation
    needs_reconstruction = bool(eval_tasks) and bool(RECON_TASKS & set(eval_tasks))
    if needs_reconstruction and gt_data.aux.gt_mesh_path:
        if not os.path.exists(gt_data.aux.gt_mesh_path):
            raise FileNotFoundError(f"GT mesh file not found: {gt_data.aux.gt_mesh_path}")
        gt_data.mesh = o3d.io.read_triangle_mesh(gt_data.aux.gt_mesh_path)
        gt_data.pcd = sample_points_from_mesh(gt_data.mesh, dataset.sampling_number)
    elif needs_reconstruction and gt_data.aux.gt_pcd_path:
        if not os.path.exists(gt_data.aux.gt_pcd_path):
            raise FileNotFoundError(f"GT point cloud file not found: {gt_data.aux.gt_pcd_path}")
        gt_data.mesh = None
        gt_data.pcd = o3d.io.read_point_cloud(gt_data.aux.gt_pcd_path)
    elif needs_reconstruction:
        raise FileNotFoundError(
            f"Dataset [{dataset_name}] scene [{scene_name}] has no GT mesh or point cloud for reconstruction evaluation."
        )

    return scene_data, gt_data


def print_metrics(metrics, all_scenes, dataset_name):
    print(f"\nPer-scene metrics of dataset [{dataset_name}]")

    scene_width = 10
    metric_width = 16

    first_scene = all_scenes[0]

    for task, task_metrics in metrics[first_scene].items():
        header = (
            f"{task:<{metric_width}} "
            + " ".join(f"{scene:>{scene_width}}" for scene in all_scenes)
            + f" {'AVG':>{scene_width}}"
        )
        print(header)
        print("-" * len(header))

        for m in task_metrics:
            values = np.array(
                [metrics[scene][task][m] for scene in all_scenes],
                dtype=float,
            )
            avg = values.mean()

            metric_name = f"{METRIC_DISPLAY_NAMES.get(m, m)}({get_indicator(m)})"
            row = (
                f"{metric_name:<{metric_width}} "
                + " ".join(f"{v:>{scene_width}.4f}" for v in values)
                + f" {avg:>{scene_width}.4f}"
            )
            print(row)
        
        print()


def print_dataset_summary(metrics, all_scenes):
    ordered_metrics = [
        ("pose", "auc03", "AUC@3"),
        ("pose", "auc30", "AUC@30"),
        ("pose", "auc15", "AUC@15"),
        ("pose", "auc05", "AUC@5"),
        ("recon_unposed", "acc", "Acc"),
        ("recon_unposed", "comp", "Comp"),
        ("recon_unposed", "overall", "CD"),
        ("recon_unposed", "precision", "Prec"),
        ("recon_unposed", "recall", "Rec"),
        ("recon_unposed", "fscore", "F1"),
        ("depth", "Abs Rel", "AbsRel"),
        ("depth", "δ < 1.25", "δ<1.25"),
    ]

    results = []
    for task, metric_key, _ in ordered_metrics:
        values = []

        for scene in all_scenes:
            v = metrics.get(scene, {}).get(task, {}).get(metric_key, None)
            if v is not None:
                values.append(float(v))

        if len(values) > 0:
            avg = np.mean(values)
            results.append(f"{avg:.4f}")
        else:
            results.append("-")

    print("Dataset AVG")
    header = "\t".join(name for _, _, name in ordered_metrics)
    print(header)

    row = "\t".join(results)
    print(row)
    print()


def _merge_sparsity_stats(dst, src):
    dst["dense_qk_pairs"] += int(src["dense_qk_pairs"])
    dst["kept_qk_pairs"] += int(src["kept_qk_pairs"])
    dst["calls"] += int(src["calls"])
    dst["heads"] += int(src["heads"])
    for key in dst.get("oracle", {}):
        dst["oracle"][key] += src.get("oracle", {}).get(key, 0)
    for mode, mode_stats in src["by_mode"].items():
        dst_mode = dst["by_mode"].setdefault(
            mode,
            {
                "dense_qk_pairs": 0,
                "kept_qk_pairs": 0,
                "calls": 0,
                "heads": 0,
            },
        )
        dst_mode["dense_qk_pairs"] += int(mode_stats["dense_qk_pairs"])
        dst_mode["kept_qk_pairs"] += int(mode_stats["kept_qk_pairs"])
        dst_mode["calls"] += int(mode_stats["calls"])
        dst_mode["heads"] += int(mode_stats["heads"])


def _new_sparsity_stats():
    return {
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


def print_sparsity_stats(prefix, sparsity_stats):
    dense_pairs = int(sparsity_stats["dense_qk_pairs"])
    kept_pairs = int(sparsity_stats["kept_qk_pairs"])
    keep_rate = kept_pairs / dense_pairs if dense_pairs else 1.0
    equivalent_sparsity = 1.0 - keep_rate
    print(
        f"{prefix} sparsity: kept_qk={kept_pairs} dense_qk={dense_pairs} "
        f"keep_rate={keep_rate:.6f} equivalent_sparsity={equivalent_sparsity:.6f}",
        flush=True,
    )
    oracle = sparsity_stats.get("oracle", {})
    queries = int(oracle.get("queries", 0))
    if queries:
        print(
            f"{prefix} oracle: mass_error={oracle['mass_error_sum'] / queries:.6f} "
            f"rep_error={oracle['rep_error_sum'] / queries:.6f} "
            f"output_error={oracle['output_error_sum'] / queries:.6f}",
            flush=True,
        )


def print_sparsity_summary(sparsity_by_scene):
    if not sparsity_by_scene:
        return

    total = _new_sparsity_stats()
    for stats in sparsity_by_scene.values():
        _merge_sparsity_stats(total, stats)

    print_sparsity_stats("Dataset", total)
    print("Sparsity by mode")
    print("mode\theads\tcalls\tkept_qk\tdense_qk\tkeep_rate\tequivalent_sparsity")
    for mode, mode_stats in sorted(total["by_mode"].items()):
        dense_pairs = int(mode_stats["dense_qk_pairs"])
        kept_pairs = int(mode_stats["kept_qk_pairs"])
        keep_rate = kept_pairs / dense_pairs if dense_pairs else 1.0
        equivalent_sparsity = 1.0 - keep_rate
        print(
            f"{mode}\t{mode_stats['heads']}\t{mode_stats['calls']}\t"
            f"{kept_pairs}\t{dense_pairs}\t{keep_rate:.6f}\t{equivalent_sparsity:.6f}"
        )
    print()


def main(cfg):
    datasets_to_eval = []
    for dataset_name in cfg.selected_datasets:
        if dataset_name not in AVAILABLE_DATASETS:
            continue

        data_cfg = cfg.datasets[dataset_name]
        dataset = MV_REGISTRY.get(dataset_name.lower())()
        apply_dataset_config(dataset, data_cfg)
        all_scenes = data_cfg.scenes if data_cfg.scenes is not None else dataset.SCENES
        validate_dataset_requirements(dataset, dataset_name, all_scenes, data_cfg.eval_tasks)
        datasets_to_eval.append((dataset_name, data_cfg, dataset, all_scenes))

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg.model, device)

    # Iterate over datasets
    for dataset_name, data_cfg, dataset, all_scenes in datasets_to_eval:
        # Iterate over scenes
        metrics = Dict()
        sparsity_by_scene = Dict()
        for scene_name in all_scenes:
            # Load data
            scene_data, gt_data = load_scene_data(
                dataset, dataset_name, scene_name, data_cfg.max_frames, data_cfg.sample_mode,
                seq_multiple=(8 if cfg.model.name.lower() == "litevggt" else 1),
                seed=cfg.seed,
                eval_tasks=data_cfg.eval_tasks,
            )

            # Inference
            old_dataset = os.environ.get("SAF3R_CURRENT_DATASET")
            old_scene = os.environ.get("SAF3R_CURRENT_SCENE")
            os.environ["SAF3R_CURRENT_DATASET"] = str(dataset_name)
            os.environ["SAF3R_CURRENT_SCENE"] = str(scene_name)
            try:
                pred_data, stats = infer_model(model, cfg.model, scene_data)
            finally:
                if old_dataset is None:
                    os.environ.pop("SAF3R_CURRENT_DATASET", None)
                else:
                    os.environ["SAF3R_CURRENT_DATASET"] = old_dataset
                if old_scene is None:
                    os.environ.pop("SAF3R_CURRENT_SCENE", None)
                else:
                    os.environ["SAF3R_CURRENT_SCENE"] = old_scene
            print(f"Latency: {stats.latency:.2f} (s)  |  Max Mem.: {stats.max_mem:.2f} (GB)")
            if stats.get("sparsity", None) is not None:
                sparsity_by_scene[scene_name] = stats.sparsity
                print_sparsity_stats(f"[{dataset_name}] {scene_name}", stats.sparsity)

            # Evaluate
            metrics[scene_name] = evaluate(dataset, gt_data, pred_data, data_cfg.eval_tasks)
            scene_summary = []
            if "pose" in metrics[scene_name]:
                pose_metrics = metrics[scene_name]["pose"]
                scene_summary.append(f"AUC@3={pose_metrics['auc03']:.4f}")
                scene_summary.append(f"AUC@30={pose_metrics['auc30']:.4f}")
            if "recon_unposed" in metrics[scene_name]:
                recon_metrics = metrics[scene_name]["recon_unposed"]
                scene_summary.append(f"Acc={recon_metrics['acc']:.4f}")
                scene_summary.append(f"Comp={recon_metrics['comp']:.4f}")
                scene_summary.append(f"CD={recon_metrics['overall']:.4f}")
                scene_summary.append(f"F1={recon_metrics['fscore']:.4f}")
            if scene_summary:
                print(f"[{dataset_name}] {scene_name} metrics: " + ", ".join(scene_summary), flush=True)

        # print metrics summary
        print_metrics(metrics, all_scenes, dataset_name)
        print_dataset_summary(metrics, all_scenes)
        print_sparsity_summary(sparsity_by_scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate on DA3-Bench across multiple tasks and datasets.")
    parser.add_argument(
        "--config_path", type=str, default="../configs/evaluation",
        help="Path to the config directory"
    )
    parser.add_argument(
        "--config", type=str, default="vggt_eval.yaml",
        help="Name of the config file (with or without .yaml extension, default: vggt_eval)"
    )
    parser.add_argument(
        "overrides", nargs="*",
        help="Optional Hydra overrides forwarded from evaluation/launch.py."
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path=args.config_path):
        cfg = compose(config_name=args.config, overrides=args.overrides)

    main(cfg)

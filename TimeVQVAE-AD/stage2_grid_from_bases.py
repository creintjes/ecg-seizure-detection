#!/usr/bin/env python3
import sys, yaml, subprocess, itertools, time, copy
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, Any, List

def parse_args():
    p = ArgumentParser()
    p.add_argument("--base-configs", nargs="+", required=True,
                   help="List of base YAMLs (each will overwrite --config-target).")
    p.add_argument("--config-target", default="configs/config.yaml",
                   help="Path the training scripts read; will be overwritten per base.")
    p.add_argument("--dataset_ind", default="1",
                   help="Dataset index arg forwarded to stage1.py and stage2.py.")
    p.add_argument("--dry-run", action="store_true")
    # Stage-2 search budget (we do NOT touch Stage-1 budget)
    p.add_argument("--stage2-steps", type=int, default=70000)
    p.add_argument("--stage2-hours", type=int, default=1)
    return p.parse_args()

def load_yaml(p: Path) -> Dict[str, Any]:
    with open(p, "r") as f:
        return yaml.safe_load(f)

def save_yaml(cfg: Dict[str, Any], p: Path):
    with open(p, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, indent=2)

def overwrite_target_with_base(base_path: Path, target_path: Path) -> Dict[str, Any]:
    cfg = load_yaml(base_path)
    save_yaml(cfg, target_path)
    return cfg

def run(script: str, dataset_ind: str) -> bool:
    # GPU device is auto-detected in your setup; we don’t pass it.
    cmd = [sys.executable, script, "--dataset_ind", dataset_ind]
    print(f"[run] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERR] {script} exited {e.returncode}")
        return False

def set_nested(cfg: Dict[str, Any], dotted: str, val: Any):
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = val

def latent_L(cfg: Dict[str, Any]) -> int:
    nfft = int(cfg["VQ-VAE"]["n_fft"])
    W = int(cfg["encoder"]["downsampled_width"])
    F = (nfft // 2) + 1
    return F * W

def suggested_batches(L: int) -> List[int]:
    # Conservative for 24 GB with hidden_dim=384
    if L <= 144:   return [256, 512]
    if L <= 288:   return [128, 256]
    if L <= 528:   return [64, 128]
    return [32, 48, 64]

def main():
    args = parse_args()
    target = Path(args.config_target)
    assert target.parent.exists(), f"Missing folder: {target.parent}"

    # Stage-2-only grid (we never touch Stage-1 keys or steps)
    prior_depths = [8, 10, 12]
    prior_heads  = [6, 8, 12]     # hidden_dim kept at 384; attn_dim_head auto = 384/heads
    prior_drop   = [0.15, 0.20]
    lr2_grid     = [1e-3, 8e-4]

    for base_path_str in args.base_configs:
        base_path = Path(base_path_str)
        if not base_path.exists():
            print(f"[ERR] Base config not found: {base_path}")
            continue

        print("\n" + "="*72)
        print(f"[BASE] {base_path}")
        print("="*72)

        # 1) Overwrite target with base config (complete file)
        cfg0 = overwrite_target_with_base(base_path, target)

        # 2) Delete existing Stage-1 checkpoint to force retraining with new config
        stage1_ckpt = Path(f"saved_models/stage1-{args.dataset_ind}00_window.ckpt")
        if stage1_ckpt.exists():
            if args.dry_run:
                print(f"[dry-run] would delete existing {stage1_ckpt}")
            else:
                stage1_ckpt.unlink()
                print(f"[cleanup] Deleted existing {stage1_ckpt} to retrain with new base config")

        # 3) Train Stage-1 (DO NOT edit Stage-1 settings/steps)
        print("[info] Training Stage-1 for this base config…")
        if args.dry_run:
            print("[dry-run] would run stage1.py")
        else:
            ok = run("stage1.py", args.dataset_ind)
            if not ok:
                print("[warn] Stage-1 failed; skipping Stage-2 sweep for this base.")
                continue

        # 3) Prepare Stage-2-only sweep for this tokenizer
        L = latent_L(cfg0)
        bs2_grid = suggested_batches(L)
        print(f"[info] Token grid length L={L}; Stage-2 batch candidates: {bs2_grid}")

        # Fixed Stage-2 knobs (safe defaults); ONLY Stage-2 keys
        fixed_stage2 = {
            "MaskGIT.choice_temperature": 2,
            "MaskGIT.T": 25,
            "MaskGIT.mask_scheduling_func": "cosine",
            "MaskGIT.prior_model.hidden_dim": 384,
            "MaskGIT.prior_model.use_rmsnorm": True,
            "MaskGIT.prior_model.ff_mult": 4,
            "trainer_params.max_steps.stage2": args.stage2_steps,
            "trainer_params.max_hours.stage2": args.stage2_hours,
            "trainer_params.val_check_interval.stage2": 1000,
        }

        combos = list(itertools.product(prior_depths, prior_heads, prior_drop, lr2_grid, bs2_grid))
        print(f"[info] Stage-2 combos to run: {len(combos)}")
        for depth, heads, drop, lr2, bs2 in combos:
            cfg = copy.deepcopy(cfg0)

            # 4) Stage-2 updates ONLY
            for k, v in fixed_stage2.items():
                set_nested(cfg, k, v)
            set_nested(cfg, "MaskGIT.prior_model.depth", int(depth))
            set_nested(cfg, "MaskGIT.prior_model.heads", int(heads))
            set_nested(cfg, "MaskGIT.prior_model.attn_dim_head", int(384 // int(heads)))  # keep hidden_dim=384
            set_nested(cfg, "MaskGIT.prior_model.dropout", float(drop))
            set_nested(cfg, "exp_params.lr.stage2", float(lr2))
            set_nested(cfg, "dataset.batch_sizes.stage2", int(bs2))

            # Never touch Stage-1: encoder.*, VQ-VAE.*, trainer_params.max_steps.stage1, etc.

            save_yaml(cfg, target)

            print(f"\n[Stage-2] base={base_path.name} | depth={depth} | heads={heads}x{384//heads} "
                  f"| drop={drop} | lr2={lr2} | bs2={bs2}")
            if args.dry_run:
                print("[dry-run] would run stage2.py")
                print("[dry-run] would delete stage2 checkpoint after run")
            else:
                t0 = time.time()
                ok2 = run("stage2.py", args.dataset_ind)
                t1 = time.time()
                print(f"[done] status={'OK' if ok2 else 'FAIL'} in {t1-t0:.1f}s")
                
                # Delete Stage 2 checkpoint to allow next grid search run
                stage2_ckpt = Path(f"saved_models/stage2-{args.dataset_ind}_window.ckpt")
                if stage2_ckpt.exists():
                    stage2_ckpt.unlink()
                    print(f"[cleanup] Deleted {stage2_ckpt} for next grid search run")

    print("\nAll done. Check Weights & Biases for results.")

if __name__ == "__main__":
    main()
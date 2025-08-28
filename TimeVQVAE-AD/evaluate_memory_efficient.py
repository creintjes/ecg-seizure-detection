"""
Modified evaluation script with memory-efficient option
"""

from argparse import ArgumentParser
import pickle
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import gc

import numpy as np

from utils import get_root_dir, load_yaml_param_settings, set_window_size
from evaluation import evaluate_fn, save_final_summarized_figure
from evaluation.memory_efficient_eval import memory_efficient_evaluate_fn, MemoryEfficientConfig

# -- Multiprocessing setup for CUDA safety --
mp.set_start_method('spawn', force=True)
Process = mp.get_context('spawn').Process

def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_ind', default=None, nargs='+', help='e.g., 1 2 or "all"')
    parser.add_argument('--latent_window_size_rates', default=[0.05, 0.1, 0.3], nargs='+')
    parser.add_argument('--rolling_window_stride_rate', default=0.05, type=float)
    parser.add_argument('--q', default=0.90, type=float)
    parser.add_argument('--explainable_sampling', default=False)
    parser.add_argument('--n_explainable_samples', type=int, default=2)
    parser.add_argument('--max_masking_rate_for_explainable_sampling', type=float, default=0.9)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--n_workers', default=0, type=int)
    parser.add_argument('--max_parallel_subjects', default=12, type=int, help='Maximum number of subjects to process in parallel')
    
    # Memory efficiency options
    parser.add_argument('--memory_efficient', action='store_true', help='Use memory-efficient evaluation')
    parser.add_argument('--max_memory_gb', default=30.0, type=float, help='Maximum memory to use (GB)')
    parser.add_argument('--chunk_size_minutes', default=60, type=int, help='Process data in chunks of this many minutes')
    parser.add_argument('--overlap_minutes', default=5, type=int, help='Overlap between chunks (minutes)')
    parser.add_argument('--reduced_parallel_processes', default=24, type=int, help='Reduce parallel processes when using memory-efficient mode')
    
    return parser.parse_args()


def process_list_arg(arg, dtype):
    return np.array(arg, dtype=dtype)


def process_bool_arg(arg):
    return str(arg).lower() == 'true'


def process_single_subject_memory_efficient(args_tuple):
    """Process a single subject with memory-efficient evaluation"""
    sub, config, args = args_tuple
    
    # Set CUDA device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    expand_labels = config['dataset']['expand_labels']
    window_suffix = "_window" if expand_labels else "_no_window"
    
    print(f'\n[MEMORY-EFFICIENT] Evaluating subject: {sub} with{" " if expand_labels else " no "}window expansion')
    
    # Check for checkpoints
    stage1_ckpt = Path('saved_models') / f'stage1-{sub}{window_suffix}.ckpt'
    stage2_ckpt = Path('saved_models') / f'stage2-{sub}{window_suffix}.ckpt'
    
    if not (stage1_ckpt.exists() and stage2_ckpt.exists()):
        # Try "all" checkpoints as fallback
        stage1_ckpt = Path('saved_models') / f'stage1-all{window_suffix}.ckpt'
        stage2_ckpt = Path('saved_models') / f'stage2-all{window_suffix}.ckpt'
        
        if not (stage1_ckpt.exists() and stage2_ckpt.exists()):
            print(f"Skipping {sub}: Missing checkpoints")
            return f"Skipped {sub}: Missing checkpoints"
        else:
            dataset_id_for_eval = "all"
    else:
        dataset_id_for_eval = sub
    
    # Configure memory-efficient settings
    memory_config = MemoryEfficientConfig(
        max_memory_gb=args.max_memory_gb,
        chunk_size_minutes=args.chunk_size_minutes,
        overlap_minutes=args.overlap_minutes,
        save_intermediate=True
    )
    
    try:
        # Process each latent window size rate
        for wsr in args.latent_window_size_rates:
            print(f"\nProcessing latent window size rate: {wsr}")
            
            memory_efficient_evaluate_fn(
                config=config,
                dataset_idx=dataset_id_for_eval,
                latent_window_size_rate=wsr,
                rolling_window_stride_rate=args.rolling_window_stride_rate,
                q=args.q,
                device=args.device,
                memory_config=memory_config
            )
            
            # Force garbage collection between evaluations
            gc.collect()
        
        # Continue with joint processing (this part uses existing code)
        return process_joint_results(sub, config, args, dataset_id_for_eval, window_suffix)
        
    except Exception as e:
        print(f"Error processing {sub}: {e}")
        return f"Failed {sub}: {e}"


def process_single_subject_original(args_tuple):
    """Process a single subject with original evaluation (for comparison)"""
    sub, config, args = args_tuple
    
    # Set CUDA device for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    
    expand_labels = config['dataset']['expand_labels']
    window_suffix = "_window" if expand_labels else "_no_window"
    
    print(f'\n[ORIGINAL] Evaluating subject: {sub} with{" " if expand_labels else " no "}window expansion')
    
    # Check for individual subject checkpoints first, then fallback to "all" checkpoints
    stage1_ckpt = Path('saved_models') / f'stage1-{sub}{window_suffix}.ckpt'
    stage2_ckpt = Path('saved_models') / f'stage2-{sub}{window_suffix}.ckpt'
    
    if not (stage1_ckpt.exists() and stage2_ckpt.exists()):
        # Try "all" checkpoints as fallback
        stage1_ckpt = Path('saved_models') / f'stage1-all{window_suffix}.ckpt'
        stage2_ckpt = Path('saved_models') / f'stage2-all{window_suffix}.ckpt'
        
        if not (stage1_ckpt.exists() and stage2_ckpt.exists()):
            print(f"Skipping {sub}: Missing checkpoints")
            return f"Skipped {sub}: Missing checkpoints"
        else:
            dataset_id_for_eval = "all"
    else:
        dataset_id_for_eval = sub
    
    # Process latent window size rates in batches with multiprocessing
    for worker_idx in range(len(args.latent_window_size_rates)):
        latent_window_size_rates = args.latent_window_size_rates[worker_idx * args.n_workers:(worker_idx + 1) * args.n_workers]
        if not latent_window_size_rates.size:
            break

        procs = []
        for wsr in latent_window_size_rates:
            proc = Process(target=evaluate_fn, args=(config, dataset_id_for_eval, wsr, args.rolling_window_stride_rate, args.q, args.device))
            procs.append(proc)
            proc.start()
        for p in procs:
            p.join()
    
    return process_joint_results(sub, config, args, dataset_id_for_eval, window_suffix)


def process_joint_results(sub, config, args, dataset_id_for_eval, window_suffix):
    """Process joint anomaly scores - shared by both methods"""
    
    # Aggregate joint anomaly scores
    sr = config['dataset']['downsample_freq']
    n_periods = config['dataset']['n_periods']
    bpm = config['dataset']['heartbeats_per_minute']
    
    a_s_star = 0.
    joint_threshold = 0.
    result = None
    
    for wsr in args.latent_window_size_rates:
        result_fname = get_root_dir().joinpath('evaluation', 'results', 
        f'{dataset_id_for_eval}{window_suffix}-anomaly_score-latent_window_size_rate_{wsr}.pkl')
        
        try:
            with open(result_fname, 'rb') as f:
                result = pickle.load(f)
                a_star = result['a_star']
                a_s_star += a_star
                joint_threshold += result['anom_threshold']
        except FileNotFoundError:
            print(f"Warning: Result file not found for {dataset_id_for_eval}, wsr={wsr}")
            continue

    if result is None:
        return f"No valid results found for {sub}"
        
    a_bar_s_star = a_s_star.mean(axis=0)
    # Smoothed average
    window_size = set_window_size(sr, n_periods=n_periods, bpm=bpm)
    a_2bar_s_star = np.zeros_like(a_bar_s_star)
    for j in range(len(a_2bar_s_star)):
        rng = slice(max(0, j - window_size // 2), j + window_size // 2)
        a_2bar_s_star[j] = np.mean(a_bar_s_star[rng])

    a_final = (a_bar_s_star + a_2bar_s_star) / 2
    final_threshold = joint_threshold.mean()
    anom_ind = a_final > final_threshold

    # Plot
    save_final_summarized_figure(
        sub, result['X_test_unscaled'], result['Y'], result['timestep_rng_test'],
        a_s_star, a_bar_s_star, a_2bar_s_star, a_final,
        joint_threshold, final_threshold, anom_ind, window_size, config, args
    )

    # Save result
    joint_resulting_data = {
        'dataset_index': sub,
        'X_test_unscaled': result['X_test_unscaled'],
        'Y': result['Y'],
        'a_s^*': a_s_star,
        'bar{a}_s^*': a_bar_s_star,
        'doublebar{a}_s^*': a_2bar_s_star,
        'a_final': a_final,
        'joint_threshold': joint_threshold,
        'final_threshold': final_threshold,
    }
    save_path = get_root_dir().joinpath('evaluation', 'results', f'{sub}-joint_anomaly_score.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(joint_resulting_data, f, pickle.HIGHEST_PROTOCOL)
    
    return f"Successfully processed {sub}"


if __name__ == '__main__':
    args = load_args()
    config = load_yaml_param_settings(args.config)

    args.latent_window_size_rates = process_list_arg(args.latent_window_size_rates, float)
    args.explainable_sampling = process_bool_arg(args.explainable_sampling)

    sr = config['dataset']['downsample_freq']
    n_periods = config['dataset']['n_periods']
    bpm = config['dataset']['heartbeats_per_minute']
    expand_labels = config['dataset']['expand_labels']
    window_suffix = "_window" if expand_labels else "_no_window"

    if args.dataset_ind is None or args.dataset_ind == ['all'] or args.dataset_ind == 'all':
        # Process subjects 1 to 125
        subjects = [f"{i:03d}" for i in range(1, 126)]
    elif 'all' in args.dataset_ind:
        # load all subjects from result folder
        result_path = get_root_dir().joinpath("evaluation", "results")
        all_files = list(result_path.glob("sub-*_run-*_preprocessed.pkl"))
        subjects = sorted(set(f.name.split("-")[1].split("_")[0] for f in all_files), reverse=True)
        print(subjects[:10])
    else:
        subjects = [f"{int(x):03d}" for idx in args.dataset_ind for x in idx.split(',')]
    
    # Adjust parallel processing based on memory mode
    if args.memory_efficient:
        max_parallel = args.reduced_parallel_processes
        process_function = process_single_subject_memory_efficient
        print(f"\n🧠 MEMORY-EFFICIENT MODE ENABLED")
        print(f"   Max memory: {args.max_memory_gb}GB")
        print(f"   Chunk size: {args.chunk_size_minutes} minutes")
        print(f"   Overlap: {args.overlap_minutes} minutes")
        print(f"   Reduced parallel processes: {max_parallel}")
    else:
        max_parallel = args.max_parallel_subjects
        process_function = process_single_subject_original
        print(f"\n📈 ORIGINAL MODE")
        print(f"   Standard parallel processes: {max_parallel}")
    
    print(f"\nProcessing {len(subjects)} subjects: {subjects[:10]}{'...' if len(subjects) > 10 else ''}")
    print(f"Using {max_parallel} parallel processes per batch")
    
    # Process subjects in batches
    all_results = []
    total_batches = (len(subjects) + max_parallel - 1) // max_parallel
    
    for batch_idx in range(0, len(subjects), max_parallel):
        batch_subjects = subjects[batch_idx:batch_idx + max_parallel]
        current_batch_num = (batch_idx // max_parallel) + 1
        
        print(f"\n{'='*60}")
        print(f"PROCESSING BATCH {current_batch_num}/{total_batches}")
        print(f"Subjects: {batch_subjects}")
        print(f"{'='*60}")
        
        # Prepare arguments for this batch
        batch_args = [(sub, config, args) for sub in batch_subjects]
        
        # Process this batch in parallel
        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            batch_results = list(executor.map(process_function, batch_args))
        
        all_results.extend(batch_results)
        
        # Print batch summary
        print(f"\nBatch {current_batch_num} completed:")
        for result in batch_results:
            print(f"  {result}")
        
        # Force garbage collection between batches
        gc.collect()
    
    # Print final summary of all results
    print("\n" + "="*60)
    print("FINAL PROCESSING SUMMARY:")
    print("="*60)
    successful = sum(1 for r in all_results if "Successfully processed" in r)
    skipped = sum(1 for r in all_results if "Skipped" in r)
    failed = sum(1 for r in all_results if "Failed" in r)
    
    print(f"Total subjects: {len(subjects)}")
    print(f"Successfully processed: {successful}")
    print(f"Skipped (missing checkpoints): {skipped}")
    print(f"Failed (no valid results): {failed}")
    
    if args.memory_efficient:
        print(f"\n🧠 Memory-efficient mode completed!")
        print(f"   You can now run more parallel processes without memory issues")
    
    print("="*60)
    
    # Print detailed results
    for result in all_results:
        print(result)

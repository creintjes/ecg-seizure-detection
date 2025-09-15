from argparse import ArgumentParser
import pickle
import multiprocessing as mp
from pathlib import Path
import re
import numpy as np

from utils import get_root_dir, load_yaml_param_settings, set_window_size
from evaluation import evaluate_fn, save_final_summarized_figure

# -- Multiprocessing setup for CUDA safety --
mp.set_start_method('spawn', force=True)
Process = mp.get_context('spawn').Process

def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--dataset_ind', default=['all'], nargs='+', help='e.g., 001 002 or "all"')
    parser.add_argument('--latent_window_size_rates', default=[0.005, 0.01, 0.03], nargs='+')
    parser.add_argument('--rolling_window_stride_rate', default=0.25, type=float)
    parser.add_argument('--q', default=0.95, type=float)
    parser.add_argument('--explainable_sampling', default=False)
    parser.add_argument('--n_explainable_samples', type=int, default=2)
    parser.add_argument('--max_masking_rate_for_explainable_sampling', type=float, default=0.9)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--n_workers', default=3, type=int)
    return parser.parse_args()


def process_list_arg(arg, dtype):
    return np.array(arg, dtype=dtype)


def process_bool_arg(arg):
    return str(arg).lower() == 'true'


if __name__ == '__main__':
    args = load_args()
    config = load_yaml_param_settings(args.config)

    args.latent_window_size_rates = process_list_arg(args.latent_window_size_rates, float)
    args.explainable_sampling = process_bool_arg(args.explainable_sampling)

    # Get expand_labels from config for window suffix
    expand_labels = config['dataset']['expand_labels']
    window_suffix = "_window" if expand_labels else "_no_window"

    sr = config['dataset']['downsample_freq']
    n_periods = config['dataset']['n_periods']
    bpm = config['dataset']['heartbeats_per_minute']
    result_path = get_root_dir().joinpath("evaluation", "results")

    if args.dataset_ind[0] == 'all':
        subjects = [f"{idx:03d}" for idx in range(98, 126)]
    else:
        subjects = [f"{int(idx):03d}" for idx in args.dataset_ind]    
    for sub in subjects:
        print("Evaluating subject:", sub)
        for worker_idx in range(len(args.latent_window_size_rates)):
            latent_window_size_rates = args.latent_window_size_rates[worker_idx * args.n_workers:(worker_idx + 1) * args.n_workers]
            if not latent_window_size_rates.size:
                break

            procs = []
            for wsr in latent_window_size_rates:
                proc = Process(target=evaluate_fn, args=(config, sub, wsr, args.rolling_window_stride_rate, args.q, args.device))
                procs.append(proc)
                proc.start()
            for p in procs:
                p.join()
        
        # Aggregate joint anomaly scores
        a_s_star = 0.
        joint_threshold = 0.
        for wsr in args.latent_window_size_rates:
            result_fname = get_root_dir().joinpath('evaluation', 'results', f'{sub}{window_suffix}-anomaly_score-latent_window_size_rate_{wsr}.pkl')
            with open(result_fname, 'rb') as f:
                result = pickle.load(f)
                a_star = result['a_star']
                a_s_star += a_star
                joint_threshold += result['anom_threshold']

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

        # # Plot
        # save_final_summarized_figure(
        #     sub, result['X_test_unscaled'], result['Y'], result['timestep_rng_test'],
        #     a_s_star, a_bar_s_star, a_2bar_s_star, a_final,
        #     joint_threshold, final_threshold, anom_ind, window_size, config, args
        # )

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
        save_path = get_root_dir().joinpath('evaluation', 'results', f'{sub}{window_suffix}-joint_anomaly_score.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(joint_resulting_data, f, pickle.HIGHEST_PROTOCOL)

from argparse import ArgumentParser
import pickle
import multiprocessing as mp

import numpy as np

from utils import get_root_dir, load_yaml_param_settings
from evaluation import evaluate_fn, save_final_summarized_figure
from preprocessing.dataset import PreprocessedSamplesDataset

# -- Multiprocessing setup for CUDA safety --
mp.set_start_method('spawn', force=True)
Process = mp.get_context('spawn').Process

def load_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help="Path to the config file.",
        default=get_root_dir().joinpath('configs', 'config.yaml')
    )
    parser.add_argument(
        '--dataset_ind',
        default=[1],
        nargs='+',
        required=True,
        help='e.g., 1 2 3. Indices of datasets to run experiments on.'
    )
    parser.add_argument(
        '--latent_window_size_rates',
        default=[0.1], #, 0.3, 0.5],
        nargs='+',
        help='Masking rates to sweep over.'
    )
    parser.add_argument(
        '--rolling_window_stride_rate',
        default=0.1,
        type=float,
        help='stride = rolling_window_stride_rate * window_size'
    )
    parser.add_argument('--q', default=0.99, type=float)
    parser.add_argument(
        '--explainable_sampling',
        default=False,
        help='Slower but returns explainable samples if True.'
    )
    parser.add_argument(
        '--n_explainable_samples',
        type=int,
        default=2,
        help='How many explainable samples per window.'
    )
    parser.add_argument(
        '--max_masking_rate_for_explainable_sampling',
        type=float,
        default=0.9,
        help='Prevents complete masking during explainable sampling.'
    )
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument(
        '--n_workers',
        default=4,
        type=int,
        help='Number of parallel processes for the rates sweep.'
    )
    return parser.parse_args()


def process_list_arg(arg, dtype):
    return np.array(arg, dtype=dtype)


def process_bool_arg(arg):
    if str(arg) == 'True':
        return True
    if str(arg) == 'False':
        return False
    raise ValueError(f"Invalid boolean argument {arg!r}")


if __name__ == '__main__':
    args = load_args()
    config = load_yaml_param_settings(args.config)

    args.dataset_ind = process_list_arg(args.dataset_ind, int)
    args.latent_window_size_rates = process_list_arg(args.latent_window_size_rates, float)
    args.explainable_sampling = process_bool_arg(args.explainable_sampling)

    for idx in args.dataset_ind:
        print(f'\n=== Dataset idx: {idx} ===')
        dataset_idx = int(idx)

        # 1) Resolve where your preprocessed files live
        data_dir = get_root_dir().joinpath(config['dataset']['root_dir'])
        max_files = config['dataset']['max_files']

        # 2) Instantiate the train split *only* to infer window size
        train_ds = PreprocessedSamplesDataset(
            data_dir=str(data_dir),
            max_loaded_files=max_files,
            kind='train',
            train_frac=0.8,
        )
        # Grab the first sample (train) to get its length
        first = train_ds[0]
        x = first if not isinstance(first, tuple) else first[0]
        window_size = x.shape[-1]
        print(f"Using inferred window_size = {window_size}")

        # 3) Launch one subprocess per masking rate (batched by n_workers)
        for start in range(0, len(args.latent_window_size_rates), args.n_workers):
            batch_rates = args.latent_window_size_rates[start:start + args.n_workers]
            if len(batch_rates) == 0:
                break
            
            procs = []
            for wsr in batch_rates:
                p = Process(
                    target=evaluate_fn,
                    args=(
                        config,
                        dataset_idx,
                        float(wsr),
                        args.rolling_window_stride_rate,
                        args.q,
                        args.device
                    )
                )
                procs.append(p)
                p.start()

            # wait for this batch to finish
            for p in procs:
                p.join()

        # 4) After all rates done, aggregate their .pkl outputs
        a_s_star = 0.0
        joint_threshold = 0.0
        for wsr in args.latent_window_size_rates:
            fname = get_root_dir().joinpath(
                'evaluation', 'results',
                f'{dataset_idx}-anomaly_score-latent_window_size_rate_{wsr}.pkl'
            )
            with open(str(fname), 'rb') as f:
                result = pickle.load(f)
            a_s_star += result['a_star']             # shape (n_freq, ts_len)
            joint_threshold += result['anom_threshold']

        # Episode‐level aggregation
        a_bar = a_s_star.mean(axis=0)               # (ts_len,)
        # rolling smoothing of half‐windowsize
        a_2bar = np.zeros_like(a_bar)
        half_ws = window_size // 2
        for t in range(len(a_bar)):
            lo = max(0, t - half_ws)
            hi = min(len(a_bar), t + half_ws)
            a_2bar[t] = a_bar[lo:hi].mean()

        a_final = (a_bar + a_2bar) / 2.0
        final_threshold = joint_threshold.mean()
        anom_ind = a_final > final_threshold

        # 5) Plot & save the final summary figure & data
        # We assume the *last* result dict still has the unscaled X_test, Y, and timesteps
        save_final_summarized_figure(
            dataset_idx,
            result['X_test_unscaled'],
            result['Y'],
            result['timestep_rng_test'],
            a_s_star,
            a_bar,
            a_2bar,
            a_final,
            joint_threshold,
            final_threshold,
            anom_ind,
            window_size,
            config,
            args
        )

        joint_data = {
            'dataset_index': dataset_idx,
            'X_test_unscaled': result['X_test_unscaled'],
            'Y': result['Y'],
            'a_s^*': a_s_star,
            'bar{a}_s^*': a_bar,
            'doublebar{a}_s^*': a_2bar,
            'a_final': a_final,
            'joint_threshold': joint_threshold,
            'final_threshold': final_threshold,
        }
        out_fname = get_root_dir().joinpath(
            'evaluation', 'results',
            f'{dataset_idx}-joint_anomaly_score.pkl'
        )
        with open(str(out_fname), 'wb') as f:
            pickle.dump(joint_data, f, pickle.HIGHEST_PROTOCOL)

        print(f"Saved joint results to {out_fname}")
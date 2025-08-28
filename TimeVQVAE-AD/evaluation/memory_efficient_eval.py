"""
Memory-efficient evaluation module that uses lazy loading and streaming processing
to significantly reduce memory usage during evaluation.
"""

import os
import copy
import pickle
import gc
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass

from experiments.exp_stage2 import ExpStage2
from preprocessing.preprocess import scale, ASIMAnomalySequence
from utils import get_root_dir, set_window_size
from models.stage2.maskgit import MaskGIT


@dataclass
class MemoryEfficientConfig:
    """Configuration for memory-efficient evaluation"""
    max_memory_gb: float = 8.0  # Maximum memory to use for caching
    chunk_size_minutes: int = 30  # Process data in chunks of this many minutes
    overlap_minutes: int = 5  # Overlap between chunks to handle boundaries
    cache_sequences: int = 3  # Number of sequences to keep in memory
    save_intermediate: bool = True  # Save results incrementally


class LazyTimeSeriesLoader:
    """Lazy loader for time series data that processes chunks on demand"""
    
    def __init__(self, 
                 sub_id: str, 
                 config: dict, 
                 kind: str,
                 chunk_size_samples: int,
                 overlap_samples: int = 0):
        self.sub_id = sub_id
        self.config = config
        self.kind = kind
        self.chunk_size_samples = chunk_size_samples
        self.overlap_samples = overlap_samples
        
        # Initialize file paths
        sr = config['dataset']['downsample_freq']
        expand_labels = config['dataset']['expand_labels']
        data_dir = Path(config['dataset']['root_dir']) / f"downsample_freq={sr},no_windows"
        
        if sub_id == "all":
            self.file_paths = sorted(data_dir.glob("sub-*_run-*_preprocessed.pkl"))
        else:
            self.file_paths = sorted(data_dir.glob(f"sub-{sub_id}_run-*_preprocessed.pkl"))
            
        if not self.file_paths:
            raise ValueError(f"No matching files found for sub={sub_id} in {data_dir}")
        
        # Precompute sequence metadata without loading data
        self.sequence_info = self._build_sequence_info()
        self.total_length = sum(info['length'] for info in self.sequence_info)
        
        print(f"LazyLoader: {len(self.file_paths)} files, total length: {self.total_length:,} samples")
    
    def _build_sequence_info(self) -> List[Dict]:
        """Build metadata about sequences without loading full data"""
        sequence_info = []
        
        for file_path in self.file_paths:
            # Load only metadata
            with open(file_path, 'rb') as f:
                raw = pickle.load(f)
                length = len(raw['channels'][0]['windows'][0])
            
            sequence_info.append({
                'file_path': file_path,
                'length': length,
                'start_global': sum(info['length'] for info in sequence_info),  # Global start position
            })
        
        return sequence_info
    
    def get_chunk_iterator(self) -> Iterator[Tuple[np.ndarray, np.ndarray, int, int]]:
        """Iterator that yields (X_chunk, Y_chunk, global_start, global_end)"""
        
        sr = self.config['dataset']['downsample_freq']
        expand_labels = self.config['dataset']['expand_labels']
        
        current_pos = 0
        
        while current_pos < self.total_length:
            chunk_end = min(current_pos + self.chunk_size_samples, self.total_length)
            
            # Find which sequences overlap with this chunk
            X_chunk_parts = []
            Y_chunk_parts = []
            
            for seq_info in self.sequence_info:
                seq_start = seq_info['start_global']
                seq_end = seq_start + seq_info['length']
                
                # Check if sequence overlaps with current chunk
                if seq_start < chunk_end and seq_end > current_pos:
                    # Load this sequence
                    sequence = ASIMAnomalySequence.from_path(
                        seq_info['file_path'],
                        expand_labels=expand_labels,
                        sampling_rate=sr,
                        pre_minutes=5.0,
                        post_minutes=3.0
                    )
                    
                    # Calculate which part of the sequence to include
                    local_start = max(0, current_pos - seq_start)
                    local_end = min(seq_info['length'], chunk_end - seq_start)
                    
                    if local_end > local_start:
                        x_part = sequence.data[local_start:local_end, None]  # (T, 1)
                        y_part = sequence.labels[local_start:local_end]       # (T,)
                        
                        X_chunk_parts.append(x_part)
                        Y_chunk_parts.append(y_part)
            
            if X_chunk_parts:
                # Concatenate parts for this chunk
                X_chunk = np.concatenate(X_chunk_parts, axis=0)  # (chunk_length, 1)
                Y_chunk = np.concatenate(Y_chunk_parts, axis=0)  # (chunk_length,)
                
                # Convert to torch and rearrange
                X_chunk = torch.from_numpy(X_chunk.astype(np.float32))
                Y_chunk = torch.from_numpy(Y_chunk.astype(np.int64))
                X_chunk = rearrange(X_chunk, 'l c -> c l')  # (1, chunk_length)
                
                if self.kind == 'train':
                    yield X_chunk, None, current_pos, current_pos + X_chunk.shape[1]
                else:
                    yield X_chunk, Y_chunk, current_pos, current_pos + X_chunk.shape[1]
            
            # Move to next chunk with overlap
            current_pos = chunk_end - self.overlap_samples
    
    def get_chunk_estimate(self) -> Dict:
        """Estimate number of chunks and provide overview"""
        total_chunks = 0
        current_pos = 0
        
        while current_pos < self.total_length:
            chunk_end = min(current_pos + self.chunk_size_samples, self.total_length)
            total_chunks += 1
            current_pos = chunk_end - self.overlap_samples
            
            # Safety check to prevent infinite loop
            if current_pos >= self.total_length or total_chunks > 1000:
                break
        
        chunk_size_hours = self.chunk_size_samples / (self.config['dataset']['downsample_freq'] * 3600)
        overlap_hours = self.overlap_samples / (self.config['dataset']['downsample_freq'] * 3600)
        total_hours = self.total_length / (self.config['dataset']['downsample_freq'] * 3600)
        
        return {
            'total_chunks': total_chunks,
            'total_length_samples': self.total_length,
            'total_hours': total_hours,
            'chunk_size_samples': self.chunk_size_samples,
            'chunk_size_hours': chunk_size_hours,
            'overlap_samples': self.overlap_samples,
            'overlap_hours': overlap_hours,
            'sequences_count': len(self.file_paths)
        }
    
    def get_length_estimate(self) -> int:
        """Get total length without loading all data"""
        return self.total_length


class MemoryEfficientDetector:
    """Memory-efficient anomaly detection that processes data in chunks"""
    
    def __init__(self, config: MemoryEfficientConfig):
        self.config = config
        
    def detect_chunked(self,
                      loader: LazyTimeSeriesLoader,
                      maskgit: MaskGIT,
                      window_size: int,
                      rolling_window_stride: int,
                      latent_window_size: int,
                      compute_reconstructed_X: bool,
                      device: int) -> Dict:
        """
        Memory-efficient detection that processes data in chunks
        """
        
        # Get chunking overview
        chunk_info = loader.get_chunk_estimate()
        
        print(f"\n📋 CHUNKING OVERVIEW:")
        print(f"   📊 Total data: {chunk_info['total_hours']:.2f} hours ({chunk_info['total_length_samples']:,} samples)")
        print(f"   📁 Source files: {chunk_info['sequences_count']} sequences")
        print(f"   🧩 Chunk size: {chunk_info['chunk_size_hours']:.2f} hours ({chunk_info['chunk_size_samples']:,} samples)")
        print(f"   🔗 Overlap: {chunk_info['overlap_hours']:.2f} hours ({chunk_info['overlap_samples']:,} samples)")
        print(f"   🔢 Total chunks to process: {chunk_info['total_chunks']}")
        print(f"   ⏱️  Estimated processing time: ~{chunk_info['total_chunks'] * 2:.0f} minutes")
        
        # Get total length for initialization
        total_length = loader.get_length_estimate()
        
        # Initialize smaller result arrays that grow as needed
        results = {
            'a_star_chunks': [],
            'reconsX_chunks': [],
            'chunk_positions': [],
            'timestep_ranges': [],
            'last_window_rng': None,
        }
        
        chunk_idx = 0
        total_chunks = chunk_info['total_chunks']
        
        print(f"\n🚀 Starting chunk processing...")
        
        for X_chunk, Y_chunk, global_start, global_end in loader.get_chunk_iterator():
            
            chunk_length = X_chunk.shape[1]
            progress_pct = (chunk_idx + 1) / total_chunks * 100
            
            print(f"\n📦 Processing chunk {chunk_idx + 1}/{total_chunks} ({progress_pct:.1f}%)")
            print(f"   📍 Position: {global_start:,}-{global_end:,} ({chunk_length:,} samples)")
            
            # Process this chunk
            chunk_results = self._detect_single_chunk(
                X_chunk, maskgit, window_size, rolling_window_stride, 
                latent_window_size, compute_reconstructed_X, device
            )
            
            # Store chunk results
            results['a_star_chunks'].append(chunk_results['a_star'])
            if compute_reconstructed_X:
                results['reconsX_chunks'].append(chunk_results['reconsX'])
            results['chunk_positions'].append((global_start, global_end))
            results['timestep_ranges'].append(chunk_results['timestep_rng'])
            
            # Update last window range
            if chunk_results['last_window_rng'] is not None:
                results['last_window_rng'] = slice(
                    global_start + chunk_results['last_window_rng'].start,
                    global_start + chunk_results['last_window_rng'].stop
                )
            
            chunk_idx += 1
            
            print(f"   ✅ Chunk {chunk_idx}/{total_chunks} completed")
            
            # Force garbage collection
            del X_chunk, Y_chunk, chunk_results
            gc.collect()
            
            # Optional: save intermediate results
            if self.config.save_intermediate and chunk_idx % 5 == 0:
                self._save_intermediate_results(results, chunk_idx)
                print(f"   💾 Intermediate results saved (every 5 chunks)")
        
        print(f"\n🎉 All {total_chunks} chunks processed successfully!")
        
        # Combine chunk results
        combined_results = self._combine_chunk_results(results, total_length)
        
        # Print overlap statistics for transparency
        overlap_stats = self._analyze_overlap(results)
        print(f"\n📊 Overlap Analysis:")
        print(f"   Total chunks: {overlap_stats['total_chunks']}")
        print(f"   Overlapping regions: {overlap_stats['overlap_regions']}")
        print(f"   Max overlap count: {overlap_stats['max_overlap_count']}")
        print(f"   Avg overlap count: {overlap_stats['avg_overlap_count']:.2f}")
        
        return combined_results
    
    @torch.no_grad()
    def _detect_single_chunk(self,
                            X_chunk: torch.Tensor,
                            maskgit: MaskGIT,
                            window_size: int,
                            rolling_window_stride: int,
                            latent_window_size: int,
                            compute_reconstructed_X: bool,
                            device: int) -> Dict:
        """Process a single chunk of data"""
        
        chunk_len = X_chunk.shape[1]
        n_channels = X_chunk.shape[0]
        
        if chunk_len < window_size:
            # Chunk too small, return empty results
            return {
                'a_star': np.zeros((n_channels, maskgit.H_prime, chunk_len)),
                'reconsX': np.zeros((n_channels, chunk_len)),
                'count': np.zeros((n_channels, chunk_len)),
                'timestep_rng': [],
                'last_window_rng': None,
            }
        
        end_time_step = (chunk_len - 1) - window_size
        timestep_rng = range(0, end_time_step, rolling_window_stride)
        
        # Initialize results for this chunk
        logs = {
            'a_star': np.zeros((n_channels, maskgit.H_prime, chunk_len)),
            'reconsX': np.zeros((n_channels, chunk_len)),
            'count': np.zeros((n_channels, chunk_len)),
            'last_window_rng': None,
        }
        
        for timestep_idx, timestep in enumerate(timestep_rng):
                
            # Fetch a window at each timestep
            window_rng = slice(timestep, timestep + window_size)
            x_unscaled = X_chunk[None, :, window_rng]  # (1, 1, window_size)
            x, (mu, sigma) = scale(x_unscaled, return_scale_params=True)
            
            # Encode
            z_q, s = maskgit.encode_to_z_q(x.to(device), maskgit.encoder, maskgit.vq_model)
            latent_height, latent_width = z_q.shape[2], z_q.shape[3]
            
            # Compute anomaly scores
            a_tilde = np.zeros((n_channels, latent_height, latent_width))
            for w in range(latent_width):
                kernel_rng = slice(w, w + 1) if latent_window_size == 1 else slice(
                    max(0, w - (latent_window_size - 1) // 2), 
                    w + (latent_window_size - 1) // 2 + 1
                )
                
                # Mask-prediction
                logits, logits_prob = self._mask_prediction(s, latent_height, kernel_rng, maskgit)
                
                # Prior-based anomaly score
                s_rearranged = rearrange(s, '1 (h w) -> 1 h w', h=latent_height)
                p = torch.gather(logits_prob, -1, s_rearranged[:, :, :, None])
                p = p[:, :, kernel_rng, 0]
                a_w = -1 * torch.log(p + 1e-30).mean(dim=-1)
                a_w = a_w.detach().cpu().numpy()
                a_tilde[:, :, w] = a_w
            
            a_tilde_m = F.interpolate(
                torch.from_numpy(a_tilde[None, :, :, :]), 
                size=(latent_height, window_size), 
                mode='nearest'
            )[0].numpy()
            logs['a_star'][:, :, window_rng] += a_tilde_m
            
            # Reconstructed X
            if compute_reconstructed_X:
                x_recons = maskgit.decode_token_ind_to_timeseries(s).cpu().numpy()
                x_recons = (x_recons * sigma.numpy()) + mu.numpy()
                x_recons = F.interpolate(
                    torch.from_numpy(x_recons), 
                    size=(window_size,), 
                    mode='linear'
                ).numpy()
                logs['reconsX'][:, window_rng] += x_recons[0]
            
            logs['count'][:, window_rng] += 1
        
        # Finalize chunk results
        logs['last_window_rng'] = window_rng if timestep_rng else None
        logs['count'] = np.clip(logs['count'], 1, None)
        logs['reconsX'] = logs['reconsX'] / logs['count']
        logs['timestep_rng'] = timestep_rng
        
        return logs
    
    def _mask_prediction(self, s, height, slice_rng, maskgit: MaskGIT):
        """Same as original mask_prediction but as method"""
        s_m = copy.deepcopy(s)
        s_m = rearrange(s_m, '1 (h w) -> 1 h w', h=height)
        s_m[:, :, slice_rng] = maskgit.mask_token_id
        
        logits = maskgit.transformer(rearrange(s_m, 'b h w -> b (h w)'))
        logits = rearrange(logits, '1 (h w) K -> 1 h w K', h=height)
        logits_prob = F.softmax(logits, dim=-1)
        
        return logits, logits_prob
    
    def _combine_chunk_results(self, chunk_results: Dict, total_length: int) -> Dict:
        """Combine results from all chunks into final arrays with proper overlap handling"""
        
        n_channels = chunk_results['a_star_chunks'][0].shape[0]
        H_prime = chunk_results['a_star_chunks'][0].shape[1]
        
        # Initialize final arrays
        final_a_star = np.zeros((n_channels, H_prime, total_length))
        final_reconsX = np.zeros((n_channels, total_length))
        final_count = np.zeros((n_channels, total_length))  # Track overlaps
        
        # Combine chunks with proper overlap handling
        for i, (a_star_chunk, (start_pos, end_pos)) in enumerate(zip(
            chunk_results['a_star_chunks'], 
            chunk_results['chunk_positions']
        )):
            chunk_len = end_pos - start_pos
            actual_chunk_len = min(chunk_len, a_star_chunk.shape[2])
            
            # For overlapping regions, we accumulate and will average later
            final_a_star[:, :, start_pos:start_pos + actual_chunk_len] += a_star_chunk[:, :, :actual_chunk_len]
            final_count[:, start_pos:start_pos + actual_chunk_len] += 1
            
            if chunk_results['reconsX_chunks']:
                reconsX_chunk = chunk_results['reconsX_chunks'][i]
                actual_reconsX_len = min(chunk_len, reconsX_chunk.shape[1])
                final_reconsX[:, start_pos:start_pos + actual_reconsX_len] += reconsX_chunk[:, :actual_reconsX_len]
        
        # Average overlapping regions properly
        # Avoid division by zero
        final_count = np.clip(final_count, 1, None)
        
        # Average anomaly scores in overlapping regions
        for c in range(n_channels):
            for h in range(H_prime):
                final_a_star[c, h, :] = final_a_star[c, h, :] / final_count[c, :]
        
        # Average reconstruction in overlapping regions  
        final_reconsX = final_reconsX / final_count
        
        # Collect all timestep ranges with proper offset
        all_timestep_rng = []
        for timestep_rng, (start_pos, _) in zip(
            chunk_results['timestep_ranges'], 
            chunk_results['chunk_positions']
        ):
            # Add offset and avoid duplicates in overlapping regions
            for t in timestep_rng:
                global_t = t + start_pos
                if global_t not in all_timestep_rng:  # Avoid duplicates from overlaps
                    all_timestep_rng.append(global_t)
        
        all_timestep_rng.sort()  # Ensure temporal order
        
        return {
            'a_star': final_a_star,
            'reconsX': final_reconsX,
            'count': final_count,
            'timestep_rng': all_timestep_rng,
            'last_window_rng': chunk_results['last_window_rng'],
        }
    
    def _analyze_overlap(self, chunk_results: Dict) -> Dict:
        """Analyze overlap statistics for transparency"""
        total_chunks = len(chunk_results['chunk_positions'])
        
        if total_chunks <= 1:
            return {
                'total_chunks': total_chunks,
                'overlap_regions': 0,
                'max_overlap_count': 1,
                'avg_overlap_count': 1.0
            }
        
        # Create a simple overlap count array
        positions = chunk_results['chunk_positions']
        total_length = max(end for _, end in positions)
        overlap_count = np.zeros(total_length)
        
        # Count overlaps
        for start_pos, end_pos in positions:
            overlap_count[start_pos:end_pos] += 1
        
        # Calculate statistics
        overlap_regions = np.sum(overlap_count > 1)
        max_overlap_count = int(np.max(overlap_count))
        avg_overlap_count = float(np.mean(overlap_count[overlap_count > 0]))
        
        return {
            'total_chunks': total_chunks,
            'overlap_regions': int(overlap_regions),
            'max_overlap_count': max_overlap_count,
            'avg_overlap_count': avg_overlap_count
        }
    
    def _save_intermediate_results(self, results: Dict, chunk_idx: int):
        """Save intermediate results to disk"""
        save_path = get_root_dir() / 'evaluation' / 'temp' / f'intermediate_chunk_{chunk_idx}.pkl'
        save_path.parent.mkdir(exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
        
        print(f"Saved intermediate results to {save_path}")


def memory_efficient_evaluate_fn(config,
                                dataset_idx: str,
                                latent_window_size_rate: float,
                                rolling_window_stride_rate: float,
                                q: float,
                                device: int = 0,
                                memory_config: Optional[MemoryEfficientConfig] = None):
    """
    Memory-efficient version of evaluate_fn that uses chunked processing
    """
    
    if memory_config is None:
        memory_config = MemoryEfficientConfig()
    
    # Load model
    expand_labels = config['dataset']['expand_labels']
    input_length = window_size = set_window_size(
        config['dataset']['downsample_freq'], 
        config['dataset']['n_periods'], 
        bpm=config['dataset']['heartbeats_per_minute']
    )
    
    stage2 = ExpStage2.load_from_checkpoint(
        os.path.join('saved_models', f'stage2-{dataset_idx}{"_window" if expand_labels else "_no_window"}.ckpt'), 
        dataset_idx=dataset_idx, 
        input_length=input_length, 
        config=config, 
        map_location=f'cuda:{device}'
    )
    maskgit = stage2.maskgit
    maskgit.eval()
    
    # Calculate chunking parameters
    sr = config['dataset']['downsample_freq']
    chunk_size_samples = int(memory_config.chunk_size_minutes * 60 * sr)
    overlap_samples = int(memory_config.overlap_minutes * 60 * sr)
    
    rolling_window_stride = round(window_size * rolling_window_stride_rate)
    latent_window_size = compute_latent_window_size(maskgit.W_prime.item(), latent_window_size_rate)
    
    # Initialize detector
    detector = MemoryEfficientDetector(memory_config)
    
    # Process training set
    print('===== Memory-efficient training set processing... =====')
    train_loader = LazyTimeSeriesLoader(dataset_idx, config, 'train', chunk_size_samples, overlap_samples)
    
    # Show chunking overview for training
    train_chunk_info = train_loader.get_chunk_estimate()
    print(f"\n📚 TRAINING SET CHUNKING:")
    print(f"   Data: {train_chunk_info['total_hours']:.2f}h | Chunks: {train_chunk_info['total_chunks']} | Files: {train_chunk_info['sequences_count']}")
    
    logs_train = detector.detect_chunked(
        train_loader, maskgit, window_size, rolling_window_stride, 
        latent_window_size, compute_reconstructed_X=False, device=device
    )
    
    # Compute threshold
    if q <= 1.0:
        anom_threshold = np.quantile(logs_train['a_star'], q=q, axis=-1)
    else:
        anom_threshold = np.quantile(logs_train['a_star'], q=1.0, axis=-1)
        anom_threshold += anom_threshold * (q - 1.0)
    
    # Clean up training data
    del logs_train, train_loader
    gc.collect()
    
    # Process test set
    print('\n===== Memory-efficient test set processing... =====')
    test_loader = LazyTimeSeriesLoader(dataset_idx, config, 'test', chunk_size_samples, overlap_samples)
    
    # Show chunking overview for test
    test_chunk_info = test_loader.get_chunk_estimate()
    print(f"\n🧪 TEST SET CHUNKING:")
    print(f"   Data: {test_chunk_info['total_hours']:.2f}h | Chunks: {test_chunk_info['total_chunks']} | Files: {test_chunk_info['sequences_count']}")
    
    logs_test = detector.detect_chunked(
        test_loader, maskgit, window_size, rolling_window_stride, 
        latent_window_size, compute_reconstructed_X=True, device=device
    )
    
    # For plotting, we need to load Y labels - this could be optimized further
    print('Loading labels for final processing...')
    X_test_unscaled, Y = load_data_lightweight(dataset_idx, config, 'test')
    
    # Clip to processed length
    if logs_test['last_window_rng']:
        clip_len = logs_test['last_window_rng'].stop
        X_test_unscaled = X_test_unscaled[:, :clip_len]
        a_star = logs_test['a_star'][:, :, :clip_len]
        X_recons_test = logs_test['reconsX'][:, :clip_len]
        Y = Y[:clip_len]
    else:
        # Use full arrays
        a_star = logs_test['a_star']
        X_recons_test = logs_test['reconsX']
    
    # Continue with original plotting and saving logic...
    # (This part remains the same as the original function)
    
    # Univariate processing
    X_test_unscaled = X_test_unscaled[0]
    a_star = a_star[0]
    X_recons_test = X_recons_test[0]
    anom_threshold = anom_threshold[0]
    
    # Save results
    resulting_data = {
        'dataset_index': dataset_idx,
        'latent_window_size_rate': latent_window_size_rate,
        'latent_window_size': latent_window_size,
        'rolling_window_stride_rate': rolling_window_stride_rate,
        'q': q,
        'X_test_unscaled': X_test_unscaled,
        'Y': Y,
        'a_star': a_star,
        'X_recons_test': X_recons_test,
        'timestep_rng_test': logs_test['timestep_rng'],
        'anom_threshold': anom_threshold,
    }
    
    window_suffix = "_window" if config["dataset"]["expand_labels"] else "_no_window"
    saving_fname = get_root_dir().joinpath(
        'evaluation', 
        'results', 
        f'{dataset_idx}{window_suffix}-anomaly_score-latent_window_size_rate_{latent_window_size_rate}.pkl'
    )
    
    with open(str(saving_fname), 'wb') as f:
        pickle.dump(resulting_data, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"Memory-efficient evaluation completed, saved to {saving_fname}")


def load_data_lightweight(sub_id: str, config: dict, kind: str):
    """Lightweight version that only loads what's needed for final processing"""
    # This is a simplified version - for full optimization, 
    # this should also be made lazy/chunked
    from evaluation import load_data  # Import original function
    return load_data(sub_id, config, kind)


def compute_latent_window_size(latent_width, latent_window_size_rate):
    """Same as original function"""
    latent_window_size = latent_width * latent_window_size_rate
    if np.floor(latent_window_size) == 0:
        latent_window_size = 1
    elif np.floor(latent_window_size) % 2 != 0:
        latent_window_size = int(np.floor(latent_window_size))
    elif np.ceil(latent_window_size) % 2 != 0:
        latent_window_size = int(np.ceil(latent_window_size))
    elif latent_window_size % 2 == 0:
        latent_window_size = int(latent_window_size + 1)
    else:
        raise ValueError
    return latent_window_size

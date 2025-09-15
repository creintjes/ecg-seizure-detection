#!/usr/bin/env python3
"""
Simple batch evaluation script that runs evaluate.py for subjects 098-125 in batches.
"""

import subprocess
import sys
import time
import psutil
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse


def check_ram_limit(max_ram_gb):
    """Check if RAM limit is exceeded."""
    current_ram = psutil.virtual_memory().used / (1024**3)
    return current_ram > max_ram_gb


def run_evaluation_subprocess(subject_id, max_ram_gb):
    """Run evaluation for a single subject using subprocess."""
    print(f"Starting evaluation for subject {subject_id}")
    
    try:
        cmd = [
            sys.executable, "evaluate.py",
            "--dataset_ind", subject_id
        ]
        
        # Use Popen for better control - allow real-time output
        process = subprocess.Popen(
            cmd,
            text=True
        )
        
        # Monitor process and RAM
        while process.poll() is None:  # While process is running
            if check_ram_limit(max_ram_gb):
                print(f"🛑 Terminating evaluation for subject {subject_id} due to RAM limit")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                return False, subject_id, "Terminated due to RAM limit exceeded"
            time.sleep(1)
        
        # Process finished normally
        return_code = process.wait()
        
        if return_code == 0:
            print(f"✅ Completed evaluation for subject {subject_id}")
            return True, subject_id, None
        else:
            error_msg = f"Return code: {return_code}"
            print(f"❌ Failed evaluation for subject {subject_id}: {error_msg}")
            return False, subject_id, error_msg
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Failed evaluation for subject {subject_id}: {error_msg}")
        return False, subject_id, error_msg


def main():
    parser = argparse.ArgumentParser(description='Batch evaluation script for subjects 098-125')
    parser.add_argument('--batch_size', type=int, default=5,
                       help='Number of evaluations to run in parallel')
    parser.add_argument('--max_ram_gb', type=float, default=58.0,
                       help='Maximum total system RAM usage in GB before stopping')
    
    args = parser.parse_args()
    
    # Determine which subjects to evaluate
    subjects = [f"{idx:03d}" for idx in range(1, 126)]
    
    print(f"Subjects to evaluate: {subjects}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total system RAM limit: {args.max_ram_gb}GB")
    
    # Show initial RAM usage
    total_ram = psutil.virtual_memory().total / (1024**3)
    initial_ram = psutil.virtual_memory().used / (1024**3)
    print(f"Initial RAM usage: {initial_ram:.2f}GB / {total_ram:.2f}GB")
    
    # Process in batches
    total_subjects = len(subjects)
    completed = 0
    failed = []
    stopped_due_to_ram = False
    
    for i in range(0, total_subjects, args.batch_size):
        if check_ram_limit(args.max_ram_gb):
            print("🛑 Stopping batch processing due to RAM limit")
            stopped_due_to_ram = True
            break
            
        batch = subjects[i:i + args.batch_size]
        current_ram = psutil.virtual_memory().used / (1024**3)
        print(f"\n{'='*50}")
        print(f"Processing batch {i//args.batch_size + 1}: {batch}")
        print(f"Current total system RAM usage: {current_ram:.2f}GB / {total_ram:.2f}GB")
        print(f"{'='*50}")
        
        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=len(batch)) as executor:
            futures = []
            for subject_id in batch:
                if check_ram_limit(args.max_ram_gb):
                    break
                future = executor.submit(
                    run_evaluation_subprocess,
                    subject_id=subject_id,
                    max_ram_gb=args.max_ram_gb
                )
                futures.append(future)
            
            # Wait for batch completion
            for future in as_completed(futures):
                success, subject_id, error = future.result()
                if success:
                    completed += 1
                else:
                    failed.append((subject_id, error))
                
                if check_ram_limit(args.max_ram_gb):
                    print("🛑 RAM limit exceeded during batch execution")
                    stopped_due_to_ram = True
                    break
        
        if stopped_due_to_ram:
            break
            
        print(f"Batch completed. Progress: {completed}/{total_subjects}")
        
        # 10 minute cooldown between batches
        if i + args.batch_size < total_subjects:
            print("⏰ Starting 10-minute cooldown...")
            time.sleep(600)  # 10 minutes = 600 seconds
            print("✅ Cooldown complete, starting next batch...")
    
    # Final RAM check
    final_ram = psutil.virtual_memory().used / (1024**3)
    print(f"\nFinal total system RAM usage: {final_ram:.2f}GB / {total_ram:.2f}GB")
    
    # Summary
    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total subjects: {total_subjects}")
    print(f"Completed: {completed}")
    print(f"Failed: {len(failed)}")
    if stopped_due_to_ram:
        print(f"⚠️  STOPPED DUE TO RAM LIMIT: {args.max_ram_gb}GB")
    
    if failed:
        print(f"\nFailed subjects:")
        for subject_id, error in failed:
            print(f"  - {subject_id}: {error}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create Seizure Type Mapping CSV for Test Set
Generates a CSV file with columns: subject, seizure_idx, seizure_type
- Only includes test subjects (sub-097 to sub-125)
- Excludes saturated runs
- seizure_idx is sequential per subject (across all runs)
- Only includes runs with seizures
"""

import sys
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Set
from collections import defaultdict

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import SeizeIT2 annotation class
try:
    from Information.Data.seizeit2_main.classes.annotation import Annotation
    SEIZEIT_AVAILABLE = True
except ImportError:
    print("ERROR: SeizeIT2 annotation class not found!")
    print("Please ensure Information/Data/seizeit2_main/ is available.")
    sys.exit(1)


# Saturated runs to exclude
SATURATED_TEST_RUNS = {
    ("sub-099", "run-01"), ("sub-114", "run-03"), ("sub-115", "run-11"),
    ("sub-115", "run-32"), ("sub-117", "run-13"), ("sub-118", "run-07"),
    ("sub-119", "run-24"), ("sub-119", "run-36"), ("sub-123", "run-22"),
    ("sub-124", "run-19"), ("sub-124", "run-43"), ("sub-124", "run-63"),
    ("sub-125", "run-36"), ("sub-125", "run-67")
}


class SeizureTypeMappingCreator:
    def __init__(self, seizeit2_data_path: str, output_path: str = "seizure_types.csv"):
        """
        Initialize the mapping creator.

        Args:
            seizeit2_data_path: Path to SeizeIT2 dataset
            output_path: Path to output CSV file
        """
        self.seizeit2_data_path = Path(seizeit2_data_path)
        self.output_path = Path(output_path)
        self.test_subjects = [f"sub-{i:03d}" for i in range(97, 126)]  # sub-097 to sub-125

    def get_available_runs(self, subject_id: str) -> List[str]:
        """
        Get all available run IDs for a subject.

        Args:
            subject_id: Subject ID (e.g., "sub-097")

        Returns:
            List of run IDs (e.g., ["run-01", "run-02", ...])
        """
        subject_dir = self.seizeit2_data_path / subject_id

        if not subject_dir.exists():
            return []

        # Find all run directories
        run_dirs = [d.name for d in subject_dir.iterdir()
                   if d.is_dir() and d.name.startswith('run-')]

        return sorted(run_dirs)

    def load_seizure_types(self, subject_id: str, run_id: str) -> List[Tuple[str, float, float]]:
        """
        Load seizure types and timings for a specific subject and run.

        Args:
            subject_id: Subject ID (e.g., "sub-097")
            run_id: Run ID (e.g., "run-01")

        Returns:
            List of tuples (seizure_type, start_time, end_time)
            Empty list if no seizures or loading fails
        """
        try:
            # Load annotation
            annotation = Annotation.loadAnnotation(
                str(self.seizeit2_data_path),
                [subject_id, run_id]
            )

            # Check if annotation has seizure information
            if not hasattr(annotation, 'types') or not annotation.types:
                return []

            # Collect seizures with types and timings
            seizures = []

            # Get starts and ends if available
            starts = annotation.starts if hasattr(annotation, 'starts') else []
            ends = annotation.ends if hasattr(annotation, 'ends') else []

            # Match types with timings
            for i, seizure_type in enumerate(annotation.types):
                start_time = starts[i] if i < len(starts) else 0.0
                end_time = ends[i] if i < len(ends) else 0.0

                # Only include actual seizures (filter out non-seizure events)
                if seizure_type and (seizure_type.startswith('sz_') or 'seizure' in seizure_type.lower()):
                    seizures.append((seizure_type, start_time, end_time))

            return seizures

        except Exception as e:
            print(f"Warning: Could not load annotations for {subject_id} {run_id}: {e}")
            return []

    def create_mapping(self) -> List[Dict[str, any]]:
        """
        Create complete seizure type mapping for test set.

        Returns:
            List of dictionaries with keys: subject, seizure_idx, seizure_type
        """
        mapping_data = []

        print("Creating seizure type mapping for test subjects (sub-097 to sub-125)...")
        print(f"SeizeIT2 data path: {self.seizeit2_data_path}")
        print(f"Excluding {len(SATURATED_TEST_RUNS)} saturated runs")
        print()

        total_seizures = 0
        subjects_with_seizures = 0

        # Process each test subject
        for subject_id in self.test_subjects:
            print(f"Processing {subject_id}...")

            # Get available runs for this subject
            runs = self.get_available_runs(subject_id)

            if not runs:
                print(f"  No runs found for {subject_id}")
                continue

            # Track seizures per subject (for continuous indexing)
            subject_seizure_idx = 0
            subject_has_seizures = False

            # Process each run
            for run_id in runs:
                # Skip saturated runs
                if (subject_id, run_id) in SATURATED_TEST_RUNS:
                    print(f"  Skipping saturated run: {run_id}")
                    continue

                # Load seizure types for this run
                seizures = self.load_seizure_types(subject_id, run_id)

                if not seizures:
                    continue

                # Add each seizure to mapping
                for seizure_type, start_time, end_time in seizures:
                    mapping_data.append({
                        'subject': subject_id,
                        'seizure_idx': subject_seizure_idx,
                        'seizure_type': seizure_type,
                        'run_id': run_id,  # Extra info (not in output CSV)
                        'start_time': start_time,  # Extra info
                        'end_time': end_time  # Extra info
                    })

                    subject_seizure_idx += 1
                    total_seizures += 1
                    subject_has_seizures = True

                print(f"  {run_id}: {len(seizures)} seizure(s)")

            if subject_has_seizures:
                subjects_with_seizures += 1
                print(f"  Total for {subject_id}: {subject_seizure_idx} seizure(s)")
            else:
                print(f"  No seizures found for {subject_id}")

            print()

        print("="*60)
        print(f"Mapping complete!")
        print(f"Total subjects with seizures: {subjects_with_seizures}/{len(self.test_subjects)}")
        print(f"Total seizures: {total_seizures}")
        print("="*60)

        return mapping_data

    def save_to_csv(self, mapping_data: List[Dict[str, any]]):
        """
        Save mapping data to CSV file.

        Args:
            mapping_data: List of mapping dictionaries
        """
        if not mapping_data:
            print("WARNING: No seizure data to save!")
            return

        # Sort by subject and seizure_idx
        mapping_data.sort(key=lambda x: (x['subject'], x['seizure_idx']))

        # Write to CSV (only required columns)
        with open(self.output_path, 'w', newline='') as csvfile:
            fieldnames = ['subject', 'seizure_idx', 'seizure_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for row in mapping_data:
                writer.writerow({
                    'subject': row['subject'],
                    'seizure_idx': row['seizure_idx'],
                    'seizure_type': row['seizure_type']
                })

        print(f"\nCSV file saved to: {self.output_path.absolute()}")
        print(f"Total rows: {len(mapping_data)}")

        # Print sample rows
        print("\nFirst 10 rows:")
        print("-" * 60)
        print(f"{'subject':<12} {'seizure_idx':<13} {'seizure_type':<30}")
        print("-" * 60)
        for row in mapping_data[:10]:
            print(f"{row['subject']:<12} {row['seizure_idx']:<13} {row['seizure_type']:<30}")

        if len(mapping_data) > 10:
            print("...")

    def run(self):
        """Run the complete mapping creation process."""
        if not self.seizeit2_data_path.exists():
            print(f"ERROR: SeizeIT2 data path not found: {self.seizeit2_data_path}")
            print("Please provide correct path to SeizeIT2 dataset.")
            return False

        # Create mapping
        mapping_data = self.create_mapping()

        if not mapping_data:
            print("ERROR: No seizure data found!")
            return False

        # Save to CSV
        self.save_to_csv(mapping_data)

        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Create seizure type mapping CSV for test set (sub-097 to sub-125)'
    )
    parser.add_argument(
        'seizeit2_path',
        type=str,
        help='Path to SeizeIT2 dataset directory'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='seizure_types.csv',
        help='Output CSV file path (default: seizure_types.csv)'
    )

    args = parser.parse_args()

    # Create mapping
    creator = SeizureTypeMappingCreator(
        seizeit2_data_path=args.seizeit2_path,
        output_path=args.output
    )

    success = creator.run()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

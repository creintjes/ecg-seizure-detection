from pathlib import Path
from typing import List, Any
import pickle
import re
def find_files_with_prefix(directory: str, prefix: str) -> List[Path]:
    """
    Finds all files in the given directory that start with the specified prefix.

    Args:
        directory (str): Path to the directory to search.
        prefix (str): The prefix that filenames should start with.

    Returns:
        List[Path]: List of Path objects for matching files.
    """
    dir_path = Path(directory)
    return [file for file in dir_path.iterdir() if file.is_file() and file.name.startswith(prefix)]
    

def load_mps(preprocessed_data_path: str) -> List[Any]:
    """
    Load matrix profile data from pickled files matching a specific configuration.

    Args:
        preprocessed_data_path (str): Path to the preprocessed data file used to determine the configuration.

    Returns:
        List[Any]: A list of all loaded matrix profile elements from the matching pickled files.
    """
    config = preprocessed_data_path.split("/")[-1]
    MPs_path = "/home/jhagenbe_sw/ASIM/ecg-seizure-detection/MatrixProfile/MPs"
    paths = find_files_with_prefix(directory=MPs_path, prefix=config)

    array_list = []
    for path in paths:
        path=path.__str__()
        with open(path, "rb") as f:
            array_list.extend(pickle.load(f))
    return array_list
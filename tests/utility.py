import pathlib
import random
import torch
import voxel as vx


# set seeds globally for reproducibility
torch.manual_seed(0)
random.seed(0)


# store data in a cache to avoid reloading for each test
datacache = {}


def from_cache(tag : str, loader : callable):
    """
    Load an object from a cache or load it if it is not found.

    Args:
        tag (str): A unique tag for the object.
        loader (callable): A function to load the object if it is not found in the cache.
    
    Returns:
        Any: The loaded object.
    """
    if tag not in datacache:
        datacache[tag] = loader()
    return datacache[tag]


def data_file_path(filename : str) -> pathlib.Path:
    """
    Get the complete filename of a file in the data directory.

    Args:
        filename (str): The name of the file to load.
    
    Returns:
        Path: The complete path to the file.
    """
    return pathlib.Path(__file__).parent / 'data' / filename


def brain_t1w() -> vx.Volume:
    """
    An example 1mm-isotropic, T1-weighted brain volume (stored as uint8).
    """
    return from_cache('brain-t1w', lambda: vx.load_volume(data_file_path('brain-t1w.nii.gz')))

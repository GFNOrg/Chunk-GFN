import multiprocessing as mp
import pickle
from functools import partial
from itertools import product

from polyleven import levenshtein
from tqdm import tqdm


def calculate_distances(mode):
    combinations = product(["A", "C", "G", "U"], repeat=14)
    res = [levenshtein("".join(s), mode) for s in combinations]
    print(len(res))
    return res


def parallel_levenshtein_distances(modes):
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(calculate_distances, modes, chunksize=1),
                total=len(modes),
                desc="Processing modes",
            )
        )

    return dict(zip(modes, results))


# Example usage
if __name__ == "__main__":
    # Sample data - replace with your actual data
    with open("/home/mila/o/oussama.boussif/Chunk-GFN/L14_RNA1_modes.pkl", "rb") as f:
        modes = pickle.load(f)

    distances = parallel_levenshtein_distances(modes[:12])

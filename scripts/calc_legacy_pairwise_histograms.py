from glob import glob
import numpy as np
from sys import argv
from os import path
from scipy.stats import iqr

if __name__ == '__main__':
    assert len(argv) >= 3
    ground_truth_prefix = argv[1] # e.g. nest_seed_1
    comparison_folder = argv[2] 

    # Loop histograms with ground truth
    for s in glob(f"{ground_truth_prefix}_*.npy"):
        # Get name
        name = path.basename(s)
        print(name)

        # Read bin x and ground truth histogram from ground truth file
        with open(s, "rb") as f:
            bin_x = np.load(f)
            ground_truth_hist = np.load(f)

        # Get path to corresponding file in comparison path
        comparison_path = path.join(comparison_folder, s[1 + len(ground_truth_prefix):])
        if path.exists(comparison_path):
            # Load comparison data
            comparison_data = np.load(comparison_path)

            # Calculate histogram
            comparison_hist,_ = np.histogram(comparison_data, bins=bin_x)

            # Write bins and histograms to disk
            with open(ground_truth_prefix + "_" + name, "wb") as f:
                np.save(f, bin_x)
                np.save(f, ground_truth_hist)
                np.save(f, comparison_hist)
        else:
            print("WARNING: Unable to find file to compare %s against" % name)
